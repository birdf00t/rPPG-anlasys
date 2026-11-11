"""
PhysNet / PhysFormer의 only_test 결과 pickle(runs/exp/.../saved_test_outputs/*.pickle)을 읽어서
subject별 예측 BVP(파란색) vs 정답 라벨(검은색) 신호를 시각화하고 PNG로 저장한다.

pickle 구조: predictions[subject_index][chunk_index] = (chunk_len,) 형태의 예측 BVP
             labels[subject_index][chunk_index]      = (chunk_len,) 형태의 정답 BVP
chunk들을 chunk_index 순서대로 이어붙여서 전체 길이 신호로 복원한 뒤 그린다.

사용법:
    source .venv/bin/activate
    python logs/fps_experiment/plot_neural_signals.py
"""
import glob
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def to_numpy(x):
    """torch 텐서(CUDA 포함)든 numpy 배열이든 안전하게 numpy로 변환."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

PROJECT_ROOT = "/home/birdfoot/project/rPPG-Toolbox-main"

# {표시 이름: (EXP_DATA_NAME, 체크포인트 파일명 접두어)}  -- config의 EXP_DATA_NAME / MODEL_PATH와 맞춰야 함
MODEL_RUNS = {
    "PhysNet": ("UBFC-rPPG_test_PhysNet", "UBFC-rPPG_PhysNet_DiffNormalized"),
    "PhysFormer": ("UBFC-rPPG_test_PhysFormer", "UBFC-rPPG_PhysFormer_DiffNormalized"),
    "PhysMamba": ("UBFC-rPPG_test_PhysMamba", "UBFC-rPPG_PhysMamba_DiffNormalized"),
}

SECONDS_TO_SHOW = 10
OUT_DIR = os.path.join(PROJECT_ROOT, "logs", "fps_experiment", "signal_plots")
os.makedirs(OUT_DIR, exist_ok=True)


def find_pickle(exp_data_name, ckpt_prefix):
    pattern = os.path.join(PROJECT_ROOT, "runs", "exp", exp_data_name,
                            "saved_test_outputs", f"{ckpt_prefix}_UBFC-rPPG_outputs.pickle")
    matches = glob.glob(pattern)
    if not matches:
        # 파일명이 정확히 안 맞을 수 있어 폴더 전체에서 찾아봄
        matches = glob.glob(os.path.join(PROJECT_ROOT, "runs", "exp", exp_data_name,
                                          "saved_test_outputs", "*.pickle"))
    if not matches:
        raise FileNotFoundError(f"pickle 못 찾음: {pattern}")
    return matches[0]


def reconstruct_signal(chunks_dict):
    """{chunk_index: array} -> chunk_index 순서대로 이어붙인 1차원 배열."""
    keys_sorted = sorted(chunks_dict.keys(), key=lambda k: int(k))
    pieces = [to_numpy(chunks_dict[k]).flatten() for k in keys_sorted]
    return np.concatenate(pieces)


def main():
    # 모든 모델의 subject 목록을 모아서 공통으로 그림
    all_data = {}  # model_name -> {subject_id: (pred, label, fs)}
    for model_name, (exp_name, ckpt_prefix) in MODEL_RUNS.items():
        pkl_path = find_pickle(exp_name, ckpt_prefix)
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        fs = data["fs"]
        preds, labels = {}, {}
        for subj_id in data["predictions"].keys():
            preds[subj_id] = reconstruct_signal(data["predictions"][subj_id])
            labels[subj_id] = reconstruct_signal(data["labels"][subj_id])
        all_data[model_name] = {"pred": preds, "label": labels, "fs": fs}
        print(f"{model_name}: {pkl_path} 로드, subject {list(preds.keys())}, fs={fs}")

    subjects = sorted(set().union(*[d["pred"].keys() for d in all_data.values()]))
    model_names = list(MODEL_RUNS.keys())

    fig, axes = plt.subplots(len(subjects), len(model_names), figsize=(7 * len(model_names), 3.5 * len(subjects)), squeeze=False)

    for row, subj in enumerate(subjects):
        for col, model_name in enumerate(model_names):
            ax = axes[row][col]
            d = all_data[model_name]
            if subj not in d["pred"]:
                ax.text(0.5, 0.5, "no data for this subject", ha="center", va="center", transform=ax.transAxes)
                continue
            fs = d["fs"]
            pred = d["pred"][subj][: SECONDS_TO_SHOW * fs]
            label = d["label"][subj][: SECONDS_TO_SHOW * fs]
            t = np.arange(len(pred)) / fs

            def z(x):
                return (x - x.mean()) / x.std() if x.std() > 0 else x

            ax.plot(t, z(label), color="black", linewidth=1.2, label="Ground Truth")
            ax.plot(t, z(pred), color="tab:blue", linewidth=1.0, alpha=0.8, label="Predicted")
            ax.set_title(f"{model_name} - {subj}", fontsize=11)
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
            if row == len(subjects) - 1:
                ax.set_xlabel("time (s)")

    fig.suptitle(f"PhysNet vs PhysFormer predicted BVP signal (first {SECONDS_TO_SHOW}s, z-score normalized)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = os.path.join(OUT_DIR, "neural_models_signal_comparison.png")
    fig.savefig(out_path, dpi=150)
    print("SAVED:", out_path)


if __name__ == "__main__":
    main()
