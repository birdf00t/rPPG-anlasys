"""
fps를 낮추면서 영상(vid.avi)은 ffmpeg로 리샘플링했지만, ground_truth.txt(정답 BVP)는
원본(30fps 기준) 그대로 복사해둬서 길이가 안 맞았던 문제를 고친다.

BaseLoader.resample_ppg와 동일한 방식(np.interp)으로 각 fps 영상의 실제 프레임 수에 맞게
정답 신호를 다시 샘플링해서, dataset/UBFC_fps{N}/subjectXX/ground_truth.txt를 덮어쓴다.

사용법:
    source .venv/bin/activate
    python logs/fps_experiment/fix_ground_truth_resample.py
"""
import os
import numpy as np

PROJECT_ROOT = "/Users/birdf00t/Downloads/rPPG-Toolbox-main"
SUBJECTS = ["subject34", "subject42", "subject48"]
FPS_LIST = [25, 20, 15, 10]  # 30fps는 원본 그대로라 손댈 필요 없음


def resample_ppg(input_signal, target_length):
    """dataset/data_loader/BaseLoader.py의 resample_ppg와 동일한 로직."""
    return np.interp(
        np.linspace(1, input_signal.shape[0], target_length),
        np.linspace(1, input_signal.shape[0], input_signal.shape[0]),
        input_signal,
    )


def read_raw_bvp(path):
    with open(path, "r") as f:
        first_line = f.readline()
    return np.asarray([float(x) for x in first_line.split()])


def count_frames(video_path):
    import cv2
    cap = cv2.VideoCapture(video_path)
    n = 0
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        n += 1
    cap.release()
    return n


def main():
    for subject in SUBJECTS:
        raw_gt_path = os.path.join(PROJECT_ROOT, "dataset", "UBFC", subject, "ground_truth.txt")
        raw_bvp = read_raw_bvp(raw_gt_path)
        print(f"{subject}: 원본 ground truth 길이 = {len(raw_bvp)}")

        for fps in FPS_LIST:
            video_path = os.path.join(PROJECT_ROOT, "dataset", f"UBFC_fps{fps}", subject, "vid.avi")
            target_len = count_frames(video_path)
            resampled = resample_ppg(raw_bvp, target_len)

            out_path = os.path.join(PROJECT_ROOT, "dataset", f"UBFC_fps{fps}", subject, "ground_truth.txt")
            with open(out_path, "w") as f:
                f.write(" ".join(f"{v:.7e}" for v in resampled))
                f.write("\n")
            print(f"  -> fps{fps}: {len(raw_bvp)}개 -> {target_len}개로 리샘플링, {out_path} 덮어씀")


if __name__ == "__main__":
    main()
