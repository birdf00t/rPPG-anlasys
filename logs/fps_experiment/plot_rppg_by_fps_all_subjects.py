"""
subject별로, fps(30/25/20/15)에 따라 GREEN/POS/CHROM이 뽑아낸 rPPG 신호(BVP)를
앞 10초 구간만 잘라서 시각화하고 PNG로 저장하는 스크립트.

사용법:
    source .venv/bin/activate
    python logs/fps_experiment/plot_rppg_by_fps_all_subjects.py

subject 목록/캐시 경로는 아래 SUBJECTS, FPS_DIRS 딕셔너리를 수정하면 됨.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

# 한글 폰트 설정 (macOS 기본 AppleGothic)
fm.fontManager.addfont("/System/Library/Fonts/Supplemental/AppleGothic.ttf")
matplotlib.rcParams["font.family"] = "AppleGothic"
matplotlib.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = "/Users/birdf00t/Downloads/rPPG-Toolbox-main"
sys.path.insert(0, PROJECT_ROOT)

from unsupervised_methods.methods.GREEN import GREEN
from unsupervised_methods.methods.POS_WANG import POS_WANG
from unsupervised_methods.methods.CHROME_DEHAAN import CHROME_DEHAAN

BASE = os.path.join(PROJECT_ROOT, "PreprocessedData")

# fps별 캐시(EXP_DATA_NAME) 폴더 이름
FPS_DIRS = {
    30: "UBFC-rPPG_SizeW72_SizeH72_ClipLength180_DataTypeRaw_DataAugNone_LabelTypeRaw_Crop_faceTrue_BackendHC_Large_boxTrue_Large_size1.5_Dyamic_DetFalse_det_len30_Median_face_boxFalse_unsupervised",
    25: "UBFC-rPPG_fps25",
    20: "UBFC-rPPG_fps20",
    15: "UBFC-rPPG_fps15",
}

# 테스트한 subject 목록
SUBJECTS = ["subject34", "subject42", "subject48"]

METHODS = {
    "GREEN": lambda frames, fs: GREEN(frames),
    "POS": lambda frames, fs: POS_WANG(frames, fs),
    "CHROM": lambda frames, fs: CHROME_DEHAAN(frames, fs),
}

SECONDS_TO_SHOW = 10  # 앞 N초 구간만 시각화
OUT_DIR = os.path.join(PROJECT_ROOT, "logs", "fps_experiment", "signal_plots")
os.makedirs(OUT_DIR, exist_ok=True)


def plot_subject(subject: str):
    fig, axes = plt.subplots(len(FPS_DIRS), len(METHODS), figsize=(14, 10), sharex=False)

    for row, fps in enumerate(sorted(FPS_DIRS.keys(), reverse=True)):
        input_path = os.path.join(BASE, FPS_DIRS[fps], f"{subject}_input0.npy")
        frames = np.load(input_path)  # (T, H, W, 3)
        n_show = min(len(frames), SECONDS_TO_SHOW * fps)

        for col, (name, fn) in enumerate(METHODS.items()):
            ax = axes[row, col]
            try:
                bvp = np.asarray(fn(frames, fps)).flatten()
                seg = bvp[:n_show]
                t = np.arange(len(seg)) / fps
                if seg.std() > 0:
                    seg = (seg - seg.mean()) / seg.std()  # z-score 정규화 (방법 간 스케일 비교용)
                ax.plot(t, seg, linewidth=0.9, color="tab:blue")
            except Exception as e:
                ax.text(0.5, 0.5, f"실패:\n{type(e).__name__}", ha="center", va="center",
                        transform=ax.transAxes, fontsize=8, color="red")

            if row == 0:
                ax.set_title(name, fontsize=12)
            if col == 0:
                ax.set_ylabel(f"{fps} fps", fontsize=12)
            if row == len(FPS_DIRS) - 1:
                ax.set_xlabel("time (s)")
            ax.grid(alpha=0.3)

    fig.suptitle(f"{subject} — fps에 따른 rPPG 추출 신호 비교 (앞 {SECONDS_TO_SHOW}초, z-score 정규화)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = os.path.join(OUT_DIR, f"{subject}_rppg_by_fps.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print("SAVED:", out_path)
    return out_path


if __name__ == "__main__":
    saved = [plot_subject(s) for s in SUBJECTS]
    print("\n총", len(saved), "개 저장 완료")
