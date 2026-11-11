#!/usr/bin/env python3
"""
PURE 데이터의 깨진 PNG(툴박스가 읽다가 죽는 파일)를 찾고, 필요하면 옆의 정상 프레임으로 대체한다.
툴박스와 같은 cv2.imread 로 읽어서 None 이 나오는 파일을 "깨진 파일"로 본다.

    python check_pure_pngs.py scan   --root dataset/PURE              # 전체 검사 -> dataset/PURE_bad_pngs.txt
    python check_pure_pngs.py repair --root dataset/PURE              # 깨진 파일을 같은 세션의 가까운 정상 프레임으로 대체
    python check_pure_pngs.py scan   --root dataset/PURE/30           # fps 폴더가 잘 고쳐졌는지 재검사

- scan 은 root 바로 아래의 fps 폴더(30,25,20,15,10,5)는 건너뜁니다(원본만 검사).
- repair 는 대체하기 전에 깨진 원본을 <root>_bad_backup/ 에 보관합니다.
  fps 폴더의 PNG 는 원본의 하드링크라 repair 하면 fps 폴더에도 같이 반영됩니다.
  (rclone 으로 다시 받은 경우엔 하드링크가 끊기므로 make_fps_datasets.py 로 fps 폴더를 다시 만드세요)
"""
import argparse
import glob
import os
import shutil
import sys
from collections import Counter
from multiprocessing import Pool

import cv2

FPS_DIRS = {"30", "25", "20", "15", "10", "5"}


def _silence_stderr():
    # libpng 경고/에러가 수만 줄 찍히지 않게
    os.dup2(os.open(os.devnull, os.O_WRONLY), 2)


def is_bad(path):
    try:
        return path if cv2.imread(path) is None else None
    except Exception:
        return path


def list_pngs(root):
    out = []
    for f in glob.glob(os.path.join(root, "**", "*.png"), recursive=True):
        first = os.path.relpath(f, root).split(os.sep)[0]
        if first in FPS_DIRS:
            continue
        out.append(f)
    return sorted(out)


def session_of(root, path):
    parts = os.path.relpath(path, root).split(os.sep)
    return parts[0] if len(parts) > 1 else "."


def scan(root, workers):
    files = list_pngs(root)
    print(f"PNG {len(files)}개 검사 중 (workers={workers}) ...", flush=True)
    bad = []
    with Pool(workers, initializer=_silence_stderr) as pool:
        for i, r in enumerate(pool.imap_unordered(is_bad, files, chunksize=64), 1):
            if r:
                bad.append(r)
            if i % 20000 == 0:
                print(f"  {i}/{len(files)}", flush=True)
    bad.sort()
    out = root.rstrip("/") + "_bad_pngs.txt"
    with open(out, "w") as f:
        for b in bad:
            f.write(os.path.relpath(b, root) + "\n")
    per = Counter(session_of(root, b) for b in bad)
    print(f"\n깨진 PNG {len(bad)}개 / 영향받는 세션 {len(per)}개  -> {out}")
    for s, n in sorted(per.items()):
        print(f"  {s}: {n}개")
    return bad


def repair(root, workers):
    bad = scan(root, workers)
    if not bad:
        print("고칠 파일이 없습니다.")
        return
    backup = root.rstrip("/") + "_bad_backup"
    badset = set(bad)
    fixed = 0
    for b in bad:
        d = os.path.dirname(b)
        names = sorted(f for f in os.listdir(d) if f.lower().endswith(".png"))
        paths = [os.path.join(d, n) for n in names]
        i = paths.index(b)
        donor = None
        for j in range(i - 1, -1, -1):          # 먼저 앞쪽에서 가까운 정상 프레임
            if paths[j] not in badset:
                donor = paths[j]
                break
        if donor is None:
            for j in range(i + 1, len(paths)):  # 없으면 뒤쪽
                if paths[j] not in badset:
                    donor = paths[j]
                    break
        if donor is None:
            print("대체할 정상 프레임이 없음:", b)
            continue
        bk = os.path.join(backup, os.path.relpath(b, root))
        os.makedirs(os.path.dirname(bk), exist_ok=True)
        if not os.path.exists(bk):
            shutil.copyfile(b, bk)
        shutil.copyfile(donor, b)               # 같은 inode 에 덮어써서 하드링크된 fps 폴더에도 반영됨
        fixed += 1
        print(f"  대체: {os.path.relpath(b, root)}  <-  {os.path.basename(donor)}")
    print(f"\n{fixed}개 대체 완료. 원본 백업: {backup}")
    print("다시 검사해서 0개인지 확인하세요: python check_pure_pngs.py scan --root", root)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["scan", "repair"])
    ap.add_argument("--root", default="dataset/PURE")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    if not os.path.isdir(a.root):
        sys.exit(f"폴더가 없습니다: {a.root}")
    (scan if a.mode == "scan" else repair)(a.root, a.workers)


if __name__ == "__main__":
    main()
