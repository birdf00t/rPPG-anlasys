"""
fps별로 보존해둔 PhysNet/PhysFormer pickle(logs/fps_experiment/signal_plots/{Model}_fps{N}_outputs.pickle)을
읽어서, subject별로 "fps(행) x 모델(열)" 그리드에 예측 BVP(파란색) vs 정답(검은색)을 그린다.

사용법:
    source .venv/bin/activate
    python logs/fps_experiment/plot_neural_signals_by_fps.py
"""
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = "/home/birdfoot/project/rPPG-Toolbox-main"
PICKLE_DIR = os.path.join(PROJECT_ROOT, "logs", "fps_experiment", "signal_plots")
OUT_DIR = PICKLE_DIR

FPS_LIST = [30, 25, 20, 15, 10, 5]
MODEL_NAMES = ["PhysNet", "PhysFormer", "PhysMamba"]
SECONDS_TO_SHOW = 10


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def reconstruct_signal(chunks_dict):
    keys_sorted = sorted(chunks_dict.keys(), key=lambda k: int(k))
    pieces = [to_numpy(chunks_dict[k]).flatten() for k in keys_sorted]
    return np.concatenate(pieces)


def load_all():
    """data[model][fps] = {"pred": {subj: arr}, "label": {subj: arr}, "fs": fs}"""
    data = {}
    for model in MODEL_NAMES:
        data[model] = {}
        for fps in FPS_LIST:
            pkl_path = os.path.join(PICKLE_DIR, f"{model}_fps{fps}_outputs.pickle")
            if not os.path.exists(pkl_path):
                print(f"[없음] {pkl_path} - 건너뜀")
                continue
            with open(pkl_path, "rb") as f:
                raw = pickle.load(f)
            preds, labels = {}, {}
            for subj_id in raw["predictions"].keys():
                preds[subj_id] = reconstruct_signal(raw["predictions"][subj_id])
                labels[subj_id] = reconstruct_signal(raw["labels"][subj_id])
            data[model][fps] = {"pred": preds, "label": labels, "fs": raw["fs"]}
    return data


def z(x):
    return (x - x.mean()) / x.std() if x.std() > 0 else x


def plot_subject(subject, data):
    fig, axes = plt.subplots(len(FPS_LIST), len(MODEL_NAMES),
                              figsize=(7 * len(MODEL_NAMES), 2.8 * len(FPS_LIST)), squeeze=False)

    for row, fps in enumerate(FPS_LIST):
        for col, model in enumerate(MODEL_NAMES):
            ax = axes[row][col]
            d = data.get(model, {}).get(fps)
            if d is None or subject not in d["pred"]:
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
                continue
            native_fs = d["fs"]  # 텐서 자체는 항상 30fps 기준 (hold 방식이라 프레임 수 불변)
            n_show = SECONDS_TO_SHOW * native_fs
            pred = d["pred"][subject][:n_show]
            t = np.arange(len(pred)) / native_fs

            ax.plot(t, z(pred), linewidth=0.9, color="tab:blue")
            ax.set_title(f"{model} - simulated {fps}fps", fontsize=10)
            ax.grid(alpha=0.3)
            if row == len(FPS_LIST) - 1:
                ax.set_xlabel("time (s)")
            if col == 0:
                ax.set_ylabel(f"{fps}fps", fontsize=10)

    fig.suptitle(f"{subject} - PhysNet vs PhysFormer predicted BVP across simulated fps "
                 f"(first {SECONDS_TO_SHOW}s, z-score normalized)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = os.path.join(OUT_DIR, f"{subject}_neural_signal_by_fps.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print("SAVED:", out_path)


def main():
    data = load_all()
    subjects = set()
    for model in MODEL_NAMES:
        for fps in FPS_LIST:
            if fps in data.get(model, {}):
                subjects.update(data[model][fps]["pred"].keys())
    for subj in sorted(subjects):
        plot_subject(subj, data)


if __name__ == "__main__":
    main()
