"""
GREEN/POS/CHROM 때 만든 plot_fft_peak_by_fps.py와 같은 방식으로,
PhysNet/PhysFormer의 fps별(hold 시뮬레이션) 예측 신호에 detrend->bandpass->FFT를 적용해서
FFT 스펙트럼(x축=bpm)과 fps별 예측 피크, 정답 피크를 비교한다.

신경망 쪽은 hold 방식이라 텐서의 실제 fs/길이가 항상 30으로 고정되고, 정답(label)도
fps 시뮬레이션과 무관하게 동일하므로 정답 피크는 1개만 계산하면 된다.

사용법:
    source .venv/bin/activate
    python logs/fps_experiment/plot_neural_fft_peak_by_fps.py
"""
import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import torch
from scipy.signal import butter

PROJECT_ROOT = "/home/birdfoot/project/rPPG-Toolbox-main"
sys.path.insert(0, PROJECT_ROOT)
from evaluation.post_process import _detrend, _next_power_of_2

# 한글 폰트 (Linux에 나눔고딕 등이 있으면 사용, 없으면 기본 폰트로 대체되어 글자만 깨짐 -> 문제 없으면 생략 가능)
for cand in ["/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
             "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]:
    if os.path.exists(cand):
        fm.fontManager.addfont(cand)
        matplotlib.rcParams["font.family"] = fm.FontProperties(fname=cand).get_name()
        break
matplotlib.rcParams["axes.unicode_minus"] = False

PICKLE_DIR = os.path.join(PROJECT_ROOT, "logs", "fps_experiment", "signal_plots")
OUT_DIR = PICKLE_DIR

FPS_LIST = [30, 25, 20, 15, 10, 5]
MODEL_NAMES = ["PhysNet", "PhysFormer", "PhysMamba"]
SUBJECT = "subject48"
LOW_PASS, HIGH_PASS = 0.6, 3.3
COLORS = {30: "tab:blue", 25: "tab:green", 20: "tab:orange", 15: "tab:red", 10: "tab:purple", 5: "tab:brown"}


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def reconstruct_signal(chunks_dict):
    keys_sorted = sorted(chunks_dict.keys(), key=lambda k: int(k))
    pieces = [to_numpy(chunks_dict[k]).flatten() for k in keys_sorted]
    return np.concatenate(pieces)


def process_like_eval(signal_1d, fs):
    x = _detrend(signal_1d, 100)
    b, a = butter(1, [LOW_PASS / fs * 2, HIGH_PASS / fs * 2], btype="bandpass")
    x = scipy.signal.filtfilt(b, a, np.double(x))
    return x


def fft_spectrum_bpm(signal_1d, fs):
    sig = np.expand_dims(signal_1d, 0)
    N = _next_power_of_2(sig.shape[1])
    f_hz, pxx = scipy.signal.periodogram(sig, fs=fs, nfft=N, detrend=False)
    pxx = np.squeeze(pxx)
    bpm = f_hz * 60
    mask = (f_hz >= LOW_PASS) & (f_hz <= HIGH_PASS)
    peak_bpm = bpm[mask][np.argmax(pxx[mask])]
    return bpm, pxx, peak_bpm


def main():
    fig, axes = plt.subplots(1, len(MODEL_NAMES), figsize=(9 * len(MODEL_NAMES), 5))
    if len(MODEL_NAMES) == 1:
        axes = [axes]

    for col, model in enumerate(MODEL_NAMES):
        ax = axes[col]
        gt_peak = None
        for fps in FPS_LIST:
            pkl_path = os.path.join(PICKLE_DIR, f"{model}_fps{fps}_outputs.pickle")
            if not os.path.exists(pkl_path):
                print(f"[없음] {pkl_path} - 건너뜀")
                continue
            with open(pkl_path, "rb") as f:
                raw = pickle.load(f)
            fs = raw["fs"]  # hold 방식이라 항상 30으로 고정
            pred = reconstruct_signal(raw["predictions"][SUBJECT])
            label = reconstruct_signal(raw["labels"][SUBJECT])

            pred_p = process_like_eval(pred, fs)
            bpm_axis, pxx, pred_peak = fft_spectrum_bpm(pred_p, fs)

            if gt_peak is None:
                label_p = process_like_eval(label, fs)
                _, _, gt_peak = fft_spectrum_bpm(label_p, fs)

            mask = (bpm_axis >= 36) & (bpm_axis <= 198)
            y = pxx[mask]
            y = y / y.max() if y.max() > 0 else y
            ax.plot(bpm_axis[mask], y, color=COLORS[fps], label=f"{fps}fps (pred {pred_peak:.0f}bpm)")
            ax.axvline(pred_peak, color=COLORS[fps], linestyle="--", alpha=0.6, linewidth=1)

        if gt_peak is not None:
            ax.axvline(gt_peak, color="black", linewidth=2, label=f"Ground Truth ≈ {gt_peak:.0f}bpm")

        ax.set_title(model, fontsize=13)
        ax.set_xlabel("bpm")
        if col == 0:
            ax.set_ylabel("Normalized spectrum magnitude")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.3)

    fig.suptitle(f"{SUBJECT} — FFT spectrum & predicted/GT HR peak across simulated fps (PhysNet/PhysFormer)\n"
                 "(dashed = predicted peak per fps, black solid = ground-truth peak. Farther apart = larger error)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.88])

    out_path = os.path.join(OUT_DIR, f"{SUBJECT}_neural_fft_peak_by_fps.png")
    fig.savefig(out_path, dpi=150)
    print("SAVED:", out_path)


if __name__ == "__main__":
    main()
