"""
subject48 기준, fps(30/25/20/15)에 따라 GREEN/POS/CHROM이 뽑은 신호를
"실제 평가 파이프라인과 똑같이"(detrend -> bandpass[0.6,3.3]Hz -> FFT) 처리해서
FFT 스펙트럼(x축=bpm)을 그리고, 예측 피크(심박수로 인식된 값)와 정답 피크를
동시에 표시한다. -> "정확도가 왜 떨어지는지"를 주파수 영역에서 직접 확인하기 위함.

사용법:
    source .venv/bin/activate
    python logs/fps_experiment/plot_fft_peak_by_fps.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
from scipy.signal import butter

fm.fontManager.addfont("/System/Library/Fonts/Supplemental/AppleGothic.ttf")
matplotlib.rcParams["font.family"] = "AppleGothic"
matplotlib.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = "/Users/birdf00t/Downloads/rPPG-Toolbox-main"
sys.path.insert(0, PROJECT_ROOT)

from unsupervised_methods.methods.GREEN import GREEN
from unsupervised_methods.methods.POS_WANG import POS_WANG
from unsupervised_methods.methods.CHROME_DEHAAN import CHROME_DEHAAN
from evaluation.post_process import _detrend, _next_power_of_2

BASE = os.path.join(PROJECT_ROOT, "PreprocessedData")

FPS_DIRS = {
    30: "UBFC-rPPG_SizeW72_SizeH72_ClipLength180_DataTypeRaw_DataAugNone_LabelTypeRaw_Crop_faceTrue_BackendHC_Large_boxTrue_Large_size1.5_Dyamic_DetFalse_det_len30_Median_face_boxFalse_unsupervised",
    25: "UBFC-rPPG_fps25",
    20: "UBFC-rPPG_fps20",
    15: "UBFC-rPPG_fps15",
}

METHODS = {
    "GREEN": lambda frames, fs: GREEN(frames),
    "POS": lambda frames, fs: POS_WANG(frames, fs),
    "CHROM": lambda frames, fs: CHROME_DEHAAN(frames, fs),
}

SUBJECT = "subject48"
LOW_PASS, HIGH_PASS = 0.6, 3.3  # evaluation/post_process.py 기본값 (36~198bpm)
COLORS = {30: "tab:blue", 25: "tab:green", 20: "tab:orange", 15: "tab:red"}

OUT_DIR = os.path.join(PROJECT_ROOT, "logs", "fps_experiment", "signal_plots")
os.makedirs(OUT_DIR, exist_ok=True)


def process_like_eval(signal_1d, fs):
    """evaluation/post_process.py의 calculate_metric_per_video와 동일한 전처리(detrend -> bandpass)."""
    x = _detrend(signal_1d, 100)
    b, a = butter(1, [LOW_PASS / fs * 2, HIGH_PASS / fs * 2], btype="bandpass")
    x = scipy.signal.filtfilt(b, a, np.double(x))
    return x


def fft_spectrum_bpm(signal_1d, fs):
    """FFT(periodogram) 스펙트럼을 bpm 단위 x축으로 반환하고, [LOW_PASS,HIGH_PASS] 안에서의 피크(bpm)도 같이 반환."""
    sig = np.expand_dims(signal_1d, 0)
    N = _next_power_of_2(sig.shape[1])
    f_hz, pxx = scipy.signal.periodogram(sig, fs=fs, nfft=N, detrend=False)
    pxx = np.squeeze(pxx)
    bpm = f_hz * 60

    mask = (f_hz >= LOW_PASS) & (f_hz <= HIGH_PASS)
    peak_bpm = bpm[mask][np.argmax(pxx[mask])]
    return bpm, pxx, peak_bpm


fig, axes = plt.subplots(1, len(METHODS), figsize=(18, 5), sharey=False)

gt_peaks = {}  # fps -> ground-truth peak bpm (정답도 fps마다 살짝 다를 수 있어 참고용으로 표시)

for col, (name, fn) in enumerate(METHODS.items()):
    ax = axes[col]
    for fps in sorted(FPS_DIRS.keys(), reverse=True):
        input_path = os.path.join(BASE, FPS_DIRS[fps], f"{SUBJECT}_input0.npy")
        label_path = os.path.join(BASE, FPS_DIRS[fps], f"{SUBJECT}_label0.npy")
        frames = np.load(input_path)
        label = np.load(label_path).flatten()

        bvp = np.asarray(fn(frames, fps)).flatten()
        bvp_p = process_like_eval(bvp, fps)
        label_p = process_like_eval(label, fps)

        bpm_axis, pxx, pred_peak = fft_spectrum_bpm(bvp_p, fps)
        _, _, gt_peak = fft_spectrum_bpm(label_p, fps)
        gt_peaks[fps] = gt_peak

        # 표시 구간(36~198bpm)만 잘라서 그리고, 높이는 최대값 1로 정규화(방법/피fs 간 비교용)
        mask = (bpm_axis >= 36) & (bpm_axis <= 198)
        y = pxx[mask]
        y = y / y.max() if y.max() > 0 else y
        ax.plot(bpm_axis[mask], y, color=COLORS[fps], label=f"{fps}fps (예측 {pred_peak:.0f}bpm)")
        ax.axvline(pred_peak, color=COLORS[fps], linestyle="--", alpha=0.6, linewidth=1)

    # 정답(ground truth) 피크는 fps마다 리샘플링돼서 거의 같지만 참고로 30fps 기준 하나만 굵게 표시
    ax.axvline(gt_peaks[30], color="black", linestyle="-", linewidth=2, label=f"정답(GT) ≈ {gt_peaks[30]:.0f}bpm")

    ax.set_title(name, fontsize=13)
    ax.set_xlabel("bpm")
    if col == 0:
        ax.set_ylabel("정규화된 스펙트럼 세기")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)

fig.suptitle(f"{SUBJECT} — fps별 FFT 스펙트럼과 예측/정답 심박수 피크 비교\n"
             "(점선=각 fps의 예측 피크, 검은 실선=정답 피크. 점선이 검은 선에서 멀어질수록 오차가 큰 것)", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.90])

out_path = os.path.join(OUT_DIR, f"{SUBJECT}_fft_peak_by_fps.png")
fig.savefig(out_path, dpi=150)
print("SAVED:", out_path)
for fps in sorted(FPS_DIRS.keys(), reverse=True):
    print(f"  GT peak @ {fps}fps = {gt_peaks[fps]:.1f} bpm")
