#!/usr/bin/env python3
"""
UBFC / PURE / MMPD 원본 영상의 해상도와 fps 를 한 번에 출력한다. (기본: dataset/{DS}/30 에서 각 데이터셋 영상 몇 개)

    python check_video_info.py                    # 세 데이터셋 모두, 데이터셋마다 3개씩
    python check_video_info.py --n 1              # 데이터셋마다 1개씩
    python check_video_info.py --root dataset --base 30

- UBFC : avi 헤더의 fps / 가로 / 세로 / 프레임 수
- PURE : PNG 한 장의 가로 / 세로, 같은 세션의 json 타임스탬프로 계산한 fps (json 을 못 찾으면 표시만 생략)
- MMPD : .mat 의 video 배열 shape(프레임, 세로, 가로, 채널). fps 는 파일에 없어서 툴박스 가정값 30 으로 표시
"""
import argparse
import glob
import json
import os
import sys

import numpy as np


def ubfc(base, n):
    import cv2
    files = sorted(glob.glob(os.path.join(base, "**", "*.avi"), recursive=True))
    print(f"[UBFC] avi {len(files)}개 중 {min(n, len(files))}개")
    for f in files[:n]:
        c = cv2.VideoCapture(f)
        w, h, fr, fps = int(c.get(3)), int(c.get(4)), int(c.get(7)), c.get(5)
        print(f"  {os.path.relpath(f, base)}: {w}x{h}  fps={fps:.2f}  프레임={fr}  길이={fr / fps:.1f}s" if fps else f"  {f}: {w}x{h} (fps 없음)")


def pure(base, n, json_root):
    import cv2
    sessions = sorted(d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d)) and d[:2].isdigit())
    print(f"[PURE] 세션 {len(sessions)}개 중 {min(n, len(sessions))}개")
    for s in sessions[:n]:
        pngs = sorted(glob.glob(os.path.join(base, s, "**", "*.png"), recursive=True))
        if not pngs:
            print(f"  {s}: png 없음")
            continue
        im = cv2.imread(pngs[0])
        h, w = im.shape[:2]
        fps = "json 없음"
        for jp in (os.path.join(json_root, s + ".json"), os.path.join(base, s + ".json"), os.path.join(base, s, s + ".json")):
            if os.path.exists(jp):
                try:
                    d = json.load(open(jp))
                    t = np.array([x["Timestamp"] for x in d["/Image"]], dtype=np.float64) / 1e9
                    fps = f"{1 / np.median(np.diff(t)):.2f}"
                except Exception as e:
                    fps = f"json 읽기 실패({type(e).__name__})"
                break
        print(f"  {s}: {w}x{h}  fps={fps}  프레임={len(pngs)}")


def mmpd(base, n):
    import scipy.io as sio
    files = sorted(glob.glob(os.path.join(base, "**", "*.mat"), recursive=True))
    print(f"[MMPD] mat {len(files)}개 중 {min(n, len(files))}개 (파일이 커서 느릴 수 있음)")
    for f in files[:n]:
        try:
            v = sio.loadmat(f)["video"]
            t, h, w = v.shape[:3]
            print(f"  {os.path.relpath(f, base)}: {w}x{h}  fps=30(툴박스 가정, 파일에 없음)  프레임={t}  채널={v.shape[3:]}  dtype={v.dtype}")
        except Exception as e:
            print(f"  {f}: 읽기 실패 {type(e).__name__}: {e}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="dataset", help="dataset 폴더")
    ap.add_argument("--base", default="30", help="원본 fps 폴더 이름")
    ap.add_argument("--n", type=int, default=3, help="데이터셋마다 볼 영상 수")
    a = ap.parse_args()

    for name, fn in (("UBFC", lambda b: ubfc(b, a.n)),
                     ("PURE", lambda b: pure(b, a.n, os.path.join(a.root, "PURE", "json"))),
                     ("MMPD", lambda b: mmpd(b, a.n))):
        b = os.path.join(a.root, name, a.base)
        if not os.path.isdir(b):
            print(f"[{name}] 폴더 없음: {b}")
            continue
        try:
            fn(b)
        except Exception as e:
            print(f"[{name}] 실패: {type(e).__name__}: {e}")
        print()


if __name__ == "__main__":
    sys.exit(main())
