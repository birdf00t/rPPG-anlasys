#!/usr/bin/env python3
"""
UBFC / PURE / MMPD 원본 데이터를 fps별(30/25/20/15/10/5)로 실제 리샘플링해서
    <root>/<DATASET>/<fps>/<원본과 같은 하위 구조>
형태로 저장한다. (GREEN/POS/CHROM 같은 신호처리 방법용: 영상 길이(초)는 그대로, 프레임 수만 줄어듦)

- UBFC: ffmpeg -r <fps> 로 vid.avi 재생성 + ground_truth.txt 를 새 프레임 수에 맞게 np.interp 로 리샘플링
        (정답 길이 불일치 버그 방지). 헤더가 선언한 크기보다 작은 vid.avi(다운로드 중 끊긴 파일)는 건너뛰고 보고.
- PURE: PNG 프레임을 fps 비율대로 골라 하드링크(용량 0). json 은 복사 (로더가 정답을 프레임 수에 맞게 자동 리샘플링).
        원본이 <세션>/<세션>/*.png 이든 납작한 <세션>/*.png 이든, 결과는 툴박스 로더가 읽는 표준 구조
        (<fps>/<세션>/<세션>/*.png + <fps>/<세션>/<세션>.json)로 저장한다.
- MMPD: .mat 의 'video' 만 프레임 솎아내서 다시 저장, 나머지 키(GT_ppg, 메타데이터)는 그대로.
- fps == 원본(30) 폴더는 하드링크(없으면 심볼릭 링크)라 용량이 안 듦.
- 이미 끝난 항목은 다시 실행해도 건너뜀(중단 후 재개 가능).

딥러닝 모델(PhysNet/PhysFormer/PhysMamba)은 이 폴더가 아니라 원본 데이터 + 캐시 단계 hold 방식을 쓰므로 여기선 필요 없음.

사용 예:
    python make_fps_datasets.py --root dataset --dataset all --plan          # 개수/예상 용량만 확인
    python make_fps_datasets.py --root dataset --dataset ubfc --limit 1      # 각 fps당 1개만 시험 변환
    python make_fps_datasets.py --root dataset --dataset all --workers 4     # 전체 변환
"""
import argparse
import math
import os
import shutil
import struct
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

ALL_FPS = [30, 25, 20, 15, 10, 5]
NAMES = {"ubfc": "UBFC", "pure": "PURE", "mmpd": "MMPD"}
FPS_DIR_NAMES = {str(f) for f in ALL_FPS}


# ----------------------------------------------------------------- 공통 유틸
def frame_indices(n_frames, native_fps, target_fps):
    """target_fps 로 같은 시간 구간을 찍었다면 남았을 프레임의 원본 인덱스."""
    n_out = int(math.floor(n_frames * target_fps / native_fps))
    idx = np.floor(np.arange(n_out) * (native_fps / target_fps)).astype(np.int64)
    return np.clip(idx, 0, n_frames - 1)


def link_or_copy(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.lexists(dst):
        return
    try:
        os.link(src, dst)
    except OSError:
        os.symlink(os.path.abspath(src), dst)


def is_done(out_dir):
    return os.path.exists(os.path.join(out_dir, ".done"))


def mark_done(out_dir):
    open(os.path.join(out_dir, ".done"), "w").close()


def walk_native(root):
    """root 아래를 훑되, 최상위의 fps 출력 폴더(30,25,...)는 들어가지 않는다."""
    root_abs = os.path.abspath(root)
    for dirpath, dirnames, filenames in os.walk(root):
        if os.path.abspath(dirpath) == root_abs:
            dirnames[:] = sorted(d for d in dirnames if d not in FPS_DIR_NAMES)
        else:
            dirnames.sort()
        yield dirpath, dirnames, filenames


def gb(n):
    return n / 1024 ** 3


# ----------------------------------------------------------------- UBFC
def avi_truncated(path):
    """RIFF 헤더가 선언한 전체 크기보다 파일이 작으면 True(= 다운로드가 중간에 끊긴 파일)."""
    with open(path, "rb") as f:
        h = f.read(12)
    if h[:4] != b"RIFF" or h[8:12] != b"AVI ":
        return False
    declared = struct.unpack("<I", h[4:8])[0] + 8
    return os.path.getsize(path) < declared - 1024


def count_frames(path):
    if shutil.which("ffprobe"):
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
             "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", path],
            capture_output=True, text=True)
        try:
            return int(r.stdout.strip().split(",")[0])
        except ValueError:
            pass
    import cv2
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def discover_ubfc(root):
    items = []
    for dirpath, _, files in walk_native(root):
        if "vid.avi" in files and "ground_truth.txt" in files:
            items.append((dirpath, os.path.relpath(dirpath, root)))
    return items


def ubfc_task(src_dir, out_dir, fps, native_fps):
    if is_done(out_dir):
        return ("skip", src_dir, "")
    src_vid = os.path.join(src_dir, "vid.avi")
    src_gt = os.path.join(src_dir, "ground_truth.txt")
    if avi_truncated(src_vid):
        return ("error", src_dir, "vid.avi 가 헤더 선언 크기보다 작음(다운로드 중 끊김) - 다시 받아야 함")
    os.makedirs(out_dir, exist_ok=True)
    out_vid = os.path.join(out_dir, "vid.avi")
    out_gt = os.path.join(out_dir, "ground_truth.txt")

    if fps == native_fps:
        link_or_copy(src_vid, out_vid)
        shutil.copy(src_gt, out_gt)
        mark_done(out_dir)
        return ("ok", src_dir, "linked")

    r = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", src_vid, "-r", str(fps),
         "-c:v", "rawvideo", "-pix_fmt", "bgr24", out_vid],
        capture_output=True, text=True)
    if r.returncode != 0:
        return ("error", src_dir, "ffmpeg 실패: " + r.stderr[-300:])
    n = count_frames(out_vid)
    if n <= 0:
        return ("error", src_dir, "변환된 영상의 프레임 수를 읽지 못함")

    with open(src_gt, "r") as f:
        bvp = np.array([float(x) for x in f.readline().split()])
    res = np.interp(np.linspace(1, len(bvp), n), np.linspace(1, len(bvp), len(bvp)), bvp)
    with open(out_gt, "w") as f:
        f.write(" ".join(f"{v:.7e}" for v in res) + "\n")
    mark_done(out_dir)
    return ("ok", src_dir, f"{n} frames")


# ----------------------------------------------------------------- PURE
def find_pure_json(png_dir, json_index):
    """<세션>.json 을 찾는다: PNG 폴더 안 -> 부모 폴더 -> root 아래 어디든(예: dataset/PURE/json/<세션>.json,
    이름이 유일할 때). 마지막으로 PNG 폴더 안의 json 이 딱 1개면 그걸 쓴다."""
    base = os.path.basename(png_dir)
    parent = os.path.dirname(png_dir)
    for c in (os.path.join(png_dir, base + ".json"), os.path.join(parent, base + ".json")):
        if os.path.exists(c):
            return c
    cands = json_index.get(base, [])
    if len(cands) == 1:
        return cands[0]
    js = [f for f in os.listdir(png_dir) if f.lower().endswith(".json")]
    if len(js) == 1:
        return os.path.join(png_dir, js[0])
    return None


def discover_pure(root):
    """PURE: 툴박스 표준 <세션>/<세션>/*.png + <세션>/<세션>.json, 납작한 <세션>/*.png, json 이 별도 json/ 폴더에 있는 경우 등.
    결과는 어느 쪽이든 툴박스 로더가 읽는 표준 구조로 정규화해서 저장한다."""
    png_dirs, json_index = [], {}
    for dirpath, _, files in walk_native(root):
        for f in files:
            if f.lower().endswith(".json"):
                json_index.setdefault(os.path.splitext(f)[0], []).append(os.path.join(dirpath, f))
        if any(f.lower().endswith(".png") for f in files):
            png_dirs.append(dirpath)
    items = []
    for dirpath in png_dirs:
        session = os.path.basename(dirpath)
        parent = os.path.dirname(dirpath)
        if os.path.basename(parent) == session:
            outer_rel = os.path.relpath(parent, root)
        else:
            outer_rel = os.path.normpath(os.path.join(os.path.relpath(parent, root), session))
        items.append((dirpath, outer_rel, session, find_pure_json(dirpath, json_index)))
    return items


def pure_task(png_dir, out_outer, session, json_path, fps, native_fps):
    if is_done(out_outer):
        return ("skip", png_dir, "")
    if json_path is None:
        others = [f for f in os.listdir(png_dir) if not f.lower().endswith(".png")][:5]
        return ("error", png_dir, f"json 을 못 찾음 (폴더 안의 png 아닌 파일: {others})")
    pngs = sorted(f for f in os.listdir(png_dir) if f.lower().endswith(".png"))
    if not pngs:
        return ("error", png_dir, "PNG 없음")
    idx = frame_indices(len(pngs), native_fps, fps)
    out_inner = os.path.join(out_outer, session)
    os.makedirs(out_inner, exist_ok=True)
    for i in idx:
        link_or_copy(os.path.join(png_dir, pngs[i]), os.path.join(out_inner, pngs[i]))
    shutil.copy(json_path, os.path.join(out_outer, session + ".json"))
    mark_done(out_outer)
    return ("ok", png_dir, f"{len(pngs)} -> {len(idx)} frames")


# ----------------------------------------------------------------- MMPD
def discover_mmpd(root):
    items = []
    for dirpath, _, files in walk_native(root):
        for f in sorted(files):
            if f.lower().endswith(".mat"):
                p = os.path.join(dirpath, f)
                items.append((p, os.path.relpath(p, root)))
    return items


def mmpd_task(src, outs, native_fps):
    """outs: [(fps, dst), ...]. 원본 .mat 을 한 번만 읽어서 아직 없는 fps 변환본을 모두 만든다."""
    todo = [(f, d) for f, d in outs if not os.path.exists(d)]
    if not todo:
        return ("skip", src, "")
    for f, d in todo:
        if f == native_fps:
            link_or_copy(src, d)
    need = [(f, d) for f, d in todo if f != native_fps]
    if need:
        from scipy.io import loadmat, savemat
        try:
            m = loadmat(src)
        except Exception as e:
            return ("error", src, f"loadmat 실패(v7.3 hdf5 이거나 손상?): {e}")
        if "video" not in m:
            return ("error", src, "'video' 키 없음")
        video = m["video"]
        base = {k: v for k, v in m.items() if k not in ("__header__", "__version__", "__globals__", "video")}
        for f, d in need:
            os.makedirs(os.path.dirname(d), exist_ok=True)
            out = dict(base)
            out["video"] = video[frame_indices(video.shape[0], native_fps, f)]
            tmp = d + ".tmp"
            with open(tmp, "wb") as fh:
                savemat(fh, out, do_compression=True)
            os.replace(tmp, d)
    return ("ok", src, f"fps {[f for f, _ in todo]} 생성")


# ----------------------------------------------------------------- 계획 / 실행
def build_tasks(args):
    tasks = []   # (함수, 인자 튜플, 표시용 이름)
    plan = {}
    for key in args.datasets:
        name = NAMES[key]
        root = os.path.join(args.root, name)
        if not os.path.isdir(root):
            print(f"[경고] {root} 가 없어 {name} 건너뜀")
            continue
        if key == "ubfc":
            items = discover_ubfc(root)
        elif key == "pure":
            items = discover_pure(root)
        else:
            items = discover_mmpd(root)
        if args.limit:
            items = items[: args.limit]
        plan[name] = items

        if key == "mmpd":   # 파일당 1개 작업: 원본을 한 번만 읽고 모든 fps 를 만든다
            for src, rel in items:
                outs = [(fps, os.path.join(root, str(fps), rel)) for fps in args.fps]
                tasks.append((mmpd_task, (src, outs, args.native_fps), f"{name}/*/{rel}"))
            continue

        for fps in args.fps:
            for it in items:
                if key == "ubfc":
                    src, rel = it
                    tasks.append((ubfc_task, (src, os.path.join(root, str(fps), rel), fps, args.native_fps), f"{name}/{fps}/{rel}"))
                else:  # pure
                    png_dir, outer_rel, session, json_path = it
                    tasks.append((pure_task, (png_dir, os.path.join(root, str(fps), outer_rel), session, json_path, fps, args.native_fps), f"{name}/{fps}/{outer_rel}"))
    return tasks, plan


def print_plan(args, plan):
    free = shutil.disk_usage(args.root).free
    print(f"여유 디스크: {gb(free):.1f} GB\n")
    total_new = 0
    for name, items in plan.items():
        if name == "UBFC":
            sizes = [os.path.getsize(os.path.join(s, "vid.avi")) for s, _ in items]
        elif name == "MMPD":
            sizes = [os.path.getsize(p) for p, _ in items]
        else:
            sizes = []
        native = sum(sizes)
        new = sum(native * f / args.native_fps for f in args.fps if f != args.native_fps) if sizes else 0
        total_new += new
        if name == "PURE":
            no_json = sum(1 for it in items if it[3] is None)
            extra = "(PNG는 하드링크라 추가 용량 거의 0)" + (f"  [경고] json 을 못 찾은 세션 {no_json}개" if no_json else "")
        else:
            extra = f"원본 {gb(native):.1f} GB 기준 추가 최대 약 {gb(new):.1f} GB"
        print(f"{name}: 항목 {len(items)}개 x fps {args.fps} -> {extra}")
    print(f"\n추가로 필요한 디스크(대략 상한): {gb(total_new):.1f} GB")


def run_task(fn, a):
    try:
        return fn(*a)
    except Exception as e:  # 한 파일 실패가 전체를 멈추지 않게
        return ("error", str(a[0]), f"예외: {e!r}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="dataset", help="UBFC/PURE/MMPD 폴더가 들어있는 상위 폴더")
    ap.add_argument("--dataset", default="all", choices=["all", "ubfc", "pure", "mmpd"])
    ap.add_argument("--fps", type=int, nargs="+", default=ALL_FPS)
    ap.add_argument("--native_fps", type=float, default=30.0)
    ap.add_argument("--workers", type=int, default=2, help="동시 처리 개수(MMPD는 파일당 메모리를 많이 써서 작게 권장)")
    ap.add_argument("--limit", type=int, default=0, help="각 데이터셋에서 앞 N개 항목만 처리(시험용)")
    ap.add_argument("--plan", action="store_true", help="실제 변환 없이 개수/예상 용량만 출력")
    args = ap.parse_args()
    args.datasets = list(NAMES) if args.dataset == "all" else [args.dataset]

    if "ubfc" in args.datasets and any(f != args.native_fps for f in args.fps) and not shutil.which("ffmpeg"):
        sys.exit("ffmpeg 가 없습니다: sudo apt install ffmpeg")

    tasks, plan = build_tasks(args)
    print_plan(args, plan)
    if args.plan:
        return

    print(f"\n총 {len(tasks)}개 작업 시작 (workers={args.workers})\n")
    t0 = time.time()
    errors, counts = [], {"ok": 0, "skip": 0, "error": 0}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_task, fn, a): label for fn, a, label in tasks}
        for i, fut in enumerate(as_completed(futs), 1):
            status, src, msg = fut.result()
            counts[status] += 1
            label = futs[fut]
            if status == "error":
                errors.append((label, msg))
            print(f"[{i}/{len(tasks)}] {status:5s} {label} {msg}", flush=True)

    print(f"\n완료: ok={counts['ok']} skip={counts['skip']} error={counts['error']} ({time.time()-t0:.0f}s)")
    if errors:
        report = os.path.join(args.root, "_fps_conversion_errors.txt")
        with open(report, "w") as f:
            for label, msg in errors:
                f.write(f"{label}\t{msg}\n")
        print(f"실패 {len(errors)}건 -> {report}")
        for label, msg in errors[:10]:
            print("  -", label, msg)


if __name__ == "__main__":
    main()
