#!/usr/bin/env python3
"""
UBFC / PURE / MMPD 원본 데이터를 해상도별로 줄여서
    <root>/<DATASET>/<해상도 폴더>/<원본과 같은 하위 구조>
형태로 저장한다. 세 데이터셋 모두 OpenCV(cv2.resize, INTER_AREA)로 똑같이 줄인다.
fps(프레임 수/길이)는 건드리지 않는다. 정답(ground truth)도 그대로 복사한다.

해상도 폴더 이름은 UBFC·PURE 기준 가로x세로 (같은 단계는 세 데이터셋에서 같은 폴더 이름):
    640x480(원본)  480x360  320x240  160x120  80x60  40x30
- MMPD 는 원본이 240x320(가로x세로, 세로형)이라 같은 픽셀 수인 320x240 폴더가 원본이고,
  160x120 -> 120x160, 80x60 -> 60x80, 40x30 -> 30x40 으로 저장한다(가로세로를 바꿔 같은 픽셀 수).
  원본보다 큰 단계(640x480, 480x360)는 키우지 않고 건너뛴다.
- 원본과 크기가 같은 단계는 하드링크(없으면 심볼릭 링크)라 용량이 안 듦.
- 이미 끝난 항목은 다시 실행해도 건너뜀(중단 후 재개 가능). 실패는 dataset/_res_conversion_errors_<ds>.txt

저장 방식
- UBFC: vid.avi 를 프레임별로 줄여 무손실 FFV1 로 저장(안 되면 무압축). ground_truth.txt 는 복사. 끊긴 avi 는 건너뛰고 보고.
- PURE: PNG 를 프레임별로 줄여 PNG(무손실)로 저장. json 은 복사. 결과는 툴박스가 읽는 표준 구조
        (<해상도>/<세션>/<세션>/*.png + <해상도>/<세션>/<세션>.json)
- MMPD: .mat 의 'video'(float32)를 프레임별로 줄여서 다시 저장, 나머지 키(GT_ppg 등)는 그대로.

사용 예:
    python make_res_datasets.py --root dataset --dataset all --plan          # 개수만 확인
    python make_res_datasets.py --root dataset --dataset ubfc --limit 1      # 시험 변환
    python make_res_datasets.py --root dataset --dataset all --workers 4     # 전체 변환
"""
import argparse
import os
import shutil
import struct
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

LEVELS = ["640x480", "480x360", "320x240", "160x120", "80x60", "40x30"]
NAMES = {"ubfc": "UBFC", "pure": "PURE", "mmpd": "MMPD"}
LEVEL_NAMES = set(LEVELS)


def level_wh(level):
    w, h = level.split("x")
    return int(w), int(h)


def target_wh(level, native_w, native_h):
    """폴더 단계 -> 이 영상에 적용할 (가로, 세로). 원본이 세로형이면 가로세로를 바꿔 같은 픽셀 수로 만든다."""
    w, h = level_wh(level)
    return (h, w) if native_h > native_w else (w, h)


def resize(img, w, h):
    import cv2
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


# ----------------------------------------------------------------- 공통 유틸
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
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        yield dirpath, dirnames, filenames


# ----------------------------------------------------------------- UBFC
def avi_truncated(path):
    with open(path, "rb") as f:
        h = f.read(12)
    if h[:4] != b"RIFF" or h[8:12] != b"AVI ":
        return False
    declared = struct.unpack("<I", h[4:8])[0] + 8
    return os.path.getsize(path) < declared - 1024


def discover_ubfc(root):
    items = []
    for dirpath, _, files in walk_native(root):
        if "vid.avi" in files and "ground_truth.txt" in files:
            items.append((dirpath, os.path.relpath(dirpath, root)))
    return items


def open_writer(path, fps, w, h):
    import cv2
    for name, fourcc in (("FFV1", cv2.VideoWriter_fourcc(*"FFV1")), ("raw", 0)):
        vw = cv2.VideoWriter(path, fourcc, fps, (w, h))
        if vw.isOpened():
            return vw, name
        vw.release()
        if os.path.exists(path):
            os.remove(path)
    return None, None


def ubfc_task(src_dir, out_dir, level):
    import cv2
    cv2.setNumThreads(1)
    if is_done(out_dir):
        return ("skip", src_dir, "")
    src_vid = os.path.join(src_dir, "vid.avi")
    src_gt = os.path.join(src_dir, "ground_truth.txt")
    if avi_truncated(src_vid):
        return ("error", src_dir, "vid.avi 가 헤더 선언 크기보다 작음(다운로드 중 끊김) - 다시 받아야 함")
    cap = cv2.VideoCapture(src_vid)
    nw, nh = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(5) or 30.0
    w, h = target_wh(level, nw, nh)
    if w > nw or h > nh:
        cap.release()
        return ("error", src_dir, f"원본 {nw}x{nh} 보다 큰 단계({w}x{h}) - 키우지 않음")
    os.makedirs(out_dir, exist_ok=True)
    out_vid = os.path.join(out_dir, "vid.avi")
    if (w, h) == (nw, nh):
        cap.release()
        link_or_copy(src_vid, out_vid)
        shutil.copy(src_gt, os.path.join(out_dir, "ground_truth.txt"))
        mark_done(out_dir)
        return ("ok", src_dir, "linked")
    tmp = out_vid + ".tmp.avi"
    vw, codec = open_writer(tmp, fps, w, h)
    if vw is None:
        cap.release()
        return ("error", src_dir, "VideoWriter 를 열지 못함(FFV1/무압축 모두 실패)")
    n = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        vw.write(resize(fr, w, h))
        n += 1
    cap.release()
    vw.release()
    if n == 0:
        os.remove(tmp)
        return ("error", src_dir, "프레임을 하나도 읽지 못함")
    os.replace(tmp, out_vid)
    shutil.copy(src_gt, os.path.join(out_dir, "ground_truth.txt"))
    mark_done(out_dir)
    return ("ok", src_dir, f"{n} frames {nw}x{nh}->{w}x{h} ({codec})")


# ----------------------------------------------------------------- PURE
def find_pure_json(png_dir):
    base = os.path.basename(png_dir)
    parent = os.path.dirname(png_dir)
    for c in (os.path.join(png_dir, base + ".json"), os.path.join(parent, base + ".json")):
        if os.path.exists(c):
            return c
    js = [f for f in os.listdir(png_dir) if f.lower().endswith(".json")]
    if len(js) == 1:
        return os.path.join(png_dir, js[0])
    return None


def discover_pure(root):
    items = []
    for dirpath, _, files in walk_native(root):
        if any(f.lower().endswith(".png") for f in files):
            session = os.path.basename(dirpath)
            parent = os.path.dirname(dirpath)
            if os.path.basename(parent) == session:
                outer_rel = os.path.relpath(parent, root)
            else:
                outer_rel = os.path.normpath(os.path.join(os.path.relpath(parent, root), session))
            items.append((dirpath, outer_rel, session, find_pure_json(dirpath)))
    return items


def pure_task(png_dir, out_outer, session, json_path, level):
    import cv2
    cv2.setNumThreads(1)
    if is_done(out_outer):
        return ("skip", png_dir, "")
    if json_path is None:
        return ("error", png_dir, "json 을 못 찾음")
    pngs = sorted(f for f in os.listdir(png_dir) if f.lower().endswith(".png"))
    if not pngs:
        return ("error", png_dir, "PNG 없음")
    first = cv2.imread(os.path.join(png_dir, pngs[0]))
    if first is None:
        return ("error", png_dir, f"첫 PNG 를 읽지 못함: {pngs[0]}")
    nh, nw = first.shape[:2]
    w, h = target_wh(level, nw, nh)
    if w > nw or h > nh:
        return ("error", png_dir, f"원본 {nw}x{nh} 보다 큰 단계({w}x{h}) - 키우지 않음")
    out_inner = os.path.join(out_outer, session)
    os.makedirs(out_inner, exist_ok=True)
    same = (w, h) == (nw, nh)
    bad = 0
    for f in pngs:
        dst = os.path.join(out_inner, f)
        src = os.path.join(png_dir, f)
        if same:
            link_or_copy(src, dst)
            continue
        im = cv2.imread(src)
        if im is None:
            bad += 1
            continue
        cv2.imwrite(dst, resize(im, w, h), [cv2.IMWRITE_PNG_COMPRESSION, 1])
    if bad:
        return ("error", png_dir, f"읽지 못한 PNG {bad}개 (check_pure_pngs.py 로 먼저 고치세요)")
    shutil.copy(json_path, os.path.join(out_outer, session + ".json"))
    mark_done(out_outer)
    return ("ok", png_dir, f"{len(pngs)} frames {nw}x{nh}->{w}x{h}")


# ----------------------------------------------------------------- MMPD
def discover_mmpd(root):
    items = []
    for dirpath, _, files in walk_native(root):
        for f in sorted(files):
            if f.lower().endswith(".mat"):
                p = os.path.join(dirpath, f)
                items.append((p, os.path.relpath(p, root)))
    return items


def mmpd_task(src, dst, level):
    import cv2
    from scipy.io import loadmat, savemat
    cv2.setNumThreads(1)
    if os.path.exists(dst):
        return ("skip", src, "")
    try:
        m = loadmat(src)
    except Exception as e:
        return ("error", src, f"loadmat 실패(v7.3 hdf5 이거나 손상?): {e}")
    if "video" not in m:
        return ("error", src, "'video' 키 없음")
    video = m["video"]
    t, nh, nw = video.shape[:3]
    w, h = target_wh(level, nw, nh)
    if w > nw or h > nh:
        return ("skip", src, f"원본 {nw}x{nh} 보다 큰 단계({w}x{h}) - 키우지 않고 건너뜀")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if (w, h) == (nw, nh):
        link_or_copy(src, dst)
        return ("ok", src, "linked")
    out = np.empty((t, h, w) + video.shape[3:], dtype=video.dtype)
    for i in range(t):
        out[i] = resize(video[i], w, h)
    m["video"] = out
    for k in ("__header__", "__version__", "__globals__"):
        m.pop(k, None)
    tmp = dst + ".tmp"
    with open(tmp, "wb") as f:
        savemat(f, m, do_compression=True)
    os.replace(tmp, dst)
    return ("ok", src, f"{t} frames {nw}x{nh}->{w}x{h}")


# ----------------------------------------------------------------- 계획 / 실행
def build_tasks(args):
    tasks, plan = [], {}
    for key in args.datasets:
        name = NAMES[key]
        root = os.path.join(args.root, name)          # 결과를 저장할 폴더(dataset/<DS>)
        src_root = os.path.join(root, args.src)       # 원본 해상도 데이터(기본 dataset/<DS>/30)
        if not os.path.isdir(src_root):
            print(f"[경고] {src_root} 가 없어 {name} 건너뜀")
            continue
        items = {"ubfc": discover_ubfc, "pure": discover_pure, "mmpd": discover_mmpd}[key](src_root)
        if args.limit:
            items = items[: args.limit]
        plan[name] = items
        for level in args.levels:
            for it in items:
                if key == "ubfc":
                    src, rel = it
                    tasks.append((name, ubfc_task, (src, os.path.join(root, level, rel), level), f"{name}/{level}/{rel}"))
                elif key == "pure":
                    png_dir, outer_rel, session, json_path = it
                    tasks.append((name, pure_task, (png_dir, os.path.join(root, level, outer_rel), session, json_path, level), f"{name}/{level}/{outer_rel}"))
                else:
                    src, rel = it
                    tasks.append((name, mmpd_task, (src, os.path.join(root, level, rel), level), f"{name}/{level}/{rel}"))
    return tasks, plan


def run_task(fn, a):
    try:
        return fn(*a)
    except Exception as e:
        return ("error", str(a[0]), f"예외: {e!r}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="dataset")
    ap.add_argument("--dataset", default="all", choices=["all", "ubfc", "pure", "mmpd"])
    ap.add_argument("--src", default="30", help="원본으로 읽을 폴더 이름(dataset/<DS>/<src>). 기본 30(원본 fps 폴더)")
    ap.add_argument("--levels", nargs="+", default=LEVELS, choices=LEVELS)
    ap.add_argument("--workers", type=int, default=2, help="동시 처리 개수(MMPD 는 파일당 메모리를 많이 써서 작게 권장)")
    ap.add_argument("--limit", type=int, default=0, help="각 데이터셋에서 앞 N개 항목만 처리(시험용)")
    ap.add_argument("--plan", action="store_true", help="실제 변환 없이 개수만 출력")
    args = ap.parse_args()
    args.datasets = list(NAMES) if args.dataset == "all" else [args.dataset]

    tasks, plan = build_tasks(args)
    print(f"여유 디스크: {shutil.disk_usage(args.root).free / 1024 ** 3:.1f} GB")
    for name, items in plan.items():
        print(f"{name}: 항목 {len(items)}개 x 해상도 {args.levels}")
    if args.plan:
        return

    print(f"\n총 {len(tasks)}개 작업 시작 (workers={args.workers})\n")
    t0 = time.time()
    errors, counts = {}, {"ok": 0, "skip": 0, "error": 0}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_task, fn, a): (name, label) for name, fn, a, label in tasks}
        for i, fut in enumerate(as_completed(futs), 1):
            status, src, msg = fut.result()
            counts[status] += 1
            name, label = futs[fut]
            if status == "error":
                errors.setdefault(name, []).append((label, msg))
            print(f"[{i}/{len(tasks)}] {status:5s} {label} {msg}", flush=True)

    print(f"\n완료: ok={counts['ok']} skip={counts['skip']} error={counts['error']} ({time.time() - t0:.0f}s)")
    for name, errs in errors.items():
        report = os.path.join(args.root, f"_res_conversion_errors_{name.lower()}.txt")
        with open(report, "w") as f:
            for label, msg in errs:
                f.write(f"{label}\t{msg}\n")
        print(f"{name} 실패 {len(errs)}건 -> {report}")
        for label, msg in errs[:10]:
            print("  -", label, msg)


if __name__ == "__main__":
    sys.exit(main())
