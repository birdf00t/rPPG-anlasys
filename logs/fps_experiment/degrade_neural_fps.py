"""
신경망(PhysNet/PhysFormer) 캐시된 입력 프레임(.npy)에 "다운샘플 -> hold 업샘플"을 적용해서
텐서 shape(프레임 개수)는 그대로 두고 실제 시간 해상도만 낮은 fps로 시뮬레이션한다.

라벨(.npy)과 파일 리스트(csv)는 전혀 건드리지 않는다 (프레임 개수 자체가 안 바뀌니
GREEN/POS/CHROM 실험 때 겪었던 "정답 길이 불일치" 버그가 애초에 발생하지 않음).

동작:
  1. 최초 1회, 원본(30fps) 캐시 폴더를 <원본>_BACKUP_30fps 로 백업
  2. 매번 실행할 때마다 백업에서 복원 -> 지정한 target_fps로 다운샘플/hold업샘플 적용
     -> 원래 캐시 폴더에 덮어씀 (main.py는 이 폴더를 그대로 읽음)

사용법:
    source .venv/bin/activate
    python logs/fps_experiment/degrade_neural_fps.py --model PhysNet --native_fps 30 --target_fps 15
    python logs/fps_experiment/degrade_neural_fps.py --model PhysFormer --native_fps 30 --target_fps 15
"""
import argparse
import glob
import os
import shutil

import numpy as np

PROJECT_ROOT = "/home/birdfoot/project/rPPG-Toolbox-main"

CACHE_DIRS = {
    "PhysNet": "UBFC-rPPG_test_PhysNet",
    "PhysFormer": "UBFC-rPPG_test_PhysFormer",
    "PhysMamba": "UBFC-rPPG_test_PhysMamba",
}


def downsample_upsample_hold(frames: np.ndarray, native_fps: float, target_fps: float) -> np.ndarray:
    """frames: (T, H, W, C). target_fps로 찍었을 때 보였을 프레임만 골라 hold(복제)해서 원래 T로 되돌림."""
    T = frames.shape[0]
    factor = native_fps / target_fps
    kept_idx = (np.floor(np.arange(T) / factor) * factor).astype(int)
    kept_idx = np.clip(kept_idx, 0, T - 1)
    return frames[kept_idx]


def ensure_backup(cache_dir: str):
    backup_dir = cache_dir.rstrip("/") + "_BACKUP_30fps"
    if not os.path.exists(backup_dir):
        print(f"[백업] {cache_dir} -> {backup_dir}")
        shutil.copytree(cache_dir, backup_dir)
    return backup_dir


def restore_from_backup(cache_dir: str, backup_dir: str):
    # input npy만 복원 (label/파일 목록은 원래 안 바뀌므로 그대로 둬도 무방하지만, 안전하게 전체 복원)
    for f in glob.glob(os.path.join(backup_dir, "*_input*.npy")):
        shutil.copy(f, os.path.join(cache_dir, os.path.basename(f)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(CACHE_DIRS.keys()), required=True)
    parser.add_argument("--native_fps", type=float, default=30)
    parser.add_argument("--target_fps", type=float, required=True)
    args = parser.parse_args()

    cache_dir = os.path.join(PROJECT_ROOT, "PreprocessedData", CACHE_DIRS[args.model])
    if not os.path.exists(cache_dir):
        raise FileNotFoundError(f"캐시 폴더가 없습니다: {cache_dir} (먼저 30fps로 한 번 실행해서 캐시를 만들어두세요)")

    backup_dir = ensure_backup(cache_dir)
    restore_from_backup(cache_dir, backup_dir)

    if args.target_fps >= args.native_fps:
        print("target_fps >= native_fps 이므로 원본(30fps) 그대로 둡니다.")
        return

    input_files = glob.glob(os.path.join(cache_dir, "*_input*.npy"))
    print(f"{len(input_files)}개 입력 파일에 {args.native_fps}->{args.target_fps}fps 다운/업샘플 적용 중...")
    for f in input_files:
        frames = np.load(f)
        degraded = downsample_upsample_hold(frames, args.native_fps, args.target_fps)
        np.save(f, degraded)

    print(f"완료: {cache_dir} 을(를) {args.target_fps}fps 시뮬레이션 데이터로 덮어썼습니다.")
    print("이제 main.py를 DO_PREPROCESS:False 인 채로 실행하면 이 데이터로 테스트됩니다.")


if __name__ == "__main__":
    main()
