#!/usr/bin/env python3
"""
해상도별(640x480 ... 40x30) 로 GREEN/POS/CHROM + PhysNet/PhysFormer/PhysMamba 를 UBFC/PURE/MMPD 에서 돌린다.
run_all_models.py 의 실행/재시도/결과기록 코드를 그대로 가져다 쓰고, fps 대신 해상도 폴더(dataset/<DS>/<해상도>)를 순회한다.
(make_res_datasets.py 로 해상도 폴더를 먼저 만들어 두어야 함)

- fps 는 원본(FS=30) 그대로. 딥러닝도 hold 같은 흉내 없이 실제로 줄인 영상을 매 해상도마다 전처리해서 평가한다.
- 기본 프로토콜은 cross: UBFC·PURE·MMPD 모두 PURE 가중치, PURE 는 학습에 안 쓴 뒤 20%(0.8~1.0)만
  (알고리즘도 같은 영상으로 평가). 결과는 '_cross_res' 집합으로 all_results.csv 에 저장된다.
- MMPD 는 원본(240x320)보다 큰 단계(640x480, 480x360)가 없어서 4단계(320x240 ~ 40x30)만 돈다.
- 해상도마다 전처리를 새로 하므로 시간이 걸린다(얼굴 검출이 실패하면 그 영상이 빠지고 n_items 가 줄어듦 -> 요약에서 점검).

    python run_res_models.py --dry-run                              # 설정 파일만 생성
    python run_res_models.py --datasets pure --models PhysNet --levels 320x240 160x120   # 일부만
    python run_res_models.py                                        # 전부
    python run_res_models.py --summarize                            # 표로 보기
"""
import argparse
import csv
import glob
import os
import sys
import time

import run_all_models as R

LEVELS = ["640x480", "480x360", "320x240", "160x120", "80x60", "40x30"]
MMPD_LEVELS = ["320x240", "160x120", "80x60", "40x30"]
SUFFIX = "_res"


def levels_for(ds, wanted):
    return [l for l in wanted if ds != "mmpd" or l in MMPD_LEVELS]


class ResRunner(R.Runner):
    def __init__(self, args):
        super().__init__(args)
        self.sfx += SUFFIX            # 예: '_cross_res' (시험 실행이면 '_e015_cross_res')
        self.sfx += args.tag          # --tag 로 결과 집합 이름 뒤에 꼬리표를 붙임(예: '_cross_res_algall')

    def run_algorithms(self, ds):
        for level in levels_for(ds, self.a.levels):
            todo = [m for m in R.ALGOS if m in self.a.models and not self.is_done(self.key(ds, m, level))]
            if not todo:
                continue
            path = R.data_path(ds, level)
            if path is None:
                print(f"[경고] dataset/{ds.upper()}/{level} 를 못 찾아 알고리즘 {ds} {level} 건너뜀")
                continue
            exp = f"auto_{ds}_alg_{level}{self.sfx}"
            print(f"== {ds} 알고리즘 {level} ({path}) ==", flush=True)
            prepped = os.path.isdir(self.cache_dir(exp)) and bool(glob.glob(
                os.path.join(R.ROOT, "PreprocessedData", "DataFileLists", exp + "_*.csv")))
            for m in todo:
                if not prepped:
                    self.cleanup(exp)
                cfg = R.algo_config(ds, m, R.NATIVE_FPS, path, exp, not prepped, self.a.end, self.info, 0.0)   # 알고리즘은 학습이 없으므로 항상 전체 영상(BEGIN 0.0)
                ok, text = self.execute(self.key(ds, m, level), ds, "algorithm", m, level,
                                        R.write_config(f"{ds}_{m}_{level}{self.sfx}", cfg), R.DATASETS[ds][2])
                if "Preprocessed Dataset Length" in text:
                    prepped = True
            self.cleanup(exp)

    def run_deep(self, ds, model):
        ckpt = self.ckpt(ds, model)
        if not os.path.exists(os.path.join(R.ROOT, ckpt)):
            print(f"[경고] 가중치 없음: {ckpt} -> {model} {ds} 건너뜀")
            return
        printed = False
        for level in levels_for(ds, self.a.levels):
            k = self.key(ds, model, level)
            if self.is_done(k):
                continue
            path = R.data_path(ds, level)
            if path is None:
                print(f"[경고] dataset/{ds.upper()}/{level} 를 못 찾아 {model} {ds} {level} 건너뜀")
                continue
            if not printed:
                print(f"== {ds} {model} (가중치 {os.path.basename(ckpt)}) ==", flush=True)
                printed = True
            exp = f"auto_{ds}_{model}_{level}{self.sfx}"
            name = f"{ds}_{model}_{level}{self.sfx}"
            begin = self.begin(ds)
            test_cfg = R.write_config(name + "_test", R.deep_config(ds, model, path, exp, False, self.a.end, self.info, ckpt, begin))
            prep_done = os.path.isdir(self.cache_dir(exp)) and bool(glob.glob(
                os.path.join(R.ROOT, "PreprocessedData", "DataFileLists", exp + "_*.csv")))
            if prep_done:
                cfg_rel, retry = test_cfg, None
            else:
                self.cleanup(exp)
                cfg_rel = R.write_config(name + "_prep", R.deep_config(ds, model, path, exp, True, self.a.end, self.info, ckpt, begin))
                retry = test_cfg
            self.execute(k, ds, "deep", model, level, cfg_rel, R.DATASETS[ds][2], retry_cfg_rel=retry)
            self.save_pickle(exp, ds, model, level)
            self.cleanup(exp)


def summarize(path, subset):
    if not os.path.exists(path):
        print("결과 파일이 아직 없습니다:", path)
        return
    rows = [r for r in csv.DictReader(open(path)) if r["subset"] == subset]
    if not rows:
        print(f"'{subset}' 결과가 없습니다.")
        return
    latest = {(r["dataset"], r["method"], r["fps"]): r for r in rows}
    print(f"[결과 집합: {subset}]  열 = 해상도(가로x세로, UBFC·PURE 기준. MMPD 는 같은 픽셀 수로 가로세로 바꿈)")
    for metric, arrow in (("MAE", "낮을수록 좋음"), ("Pearson", "1에 가까울수록 좋음"), ("SNR", "높을수록 좋음")):
        print(f"\n### {metric} ({arrow})")
        for ds in ("ubfc", "pure", "mmpd"):
            methods = [m for m in R.ALGOS + list(R.DEEP) if any(k[0] == ds and k[1] == m for k in latest)]
            if not methods:
                continue
            print(f"\n[{ds}]  " + "method".ljust(11) + "".join(f"{l:>9}" for l in LEVELS))
            for m in methods:
                cells = []
                for l in LEVELS:
                    r = latest.get((ds, m, l))
                    cells.append("-" if r is None else "FAIL" if r["status"] != "ok" else f"{float(r[metric]):.2f}")
                print("        " + m.ljust(11) + "".join(f"{c:>9}" for c in cells))
    print("\n### 평가 영상 수(n_items) 점검 - 해상도가 낮아져 얼굴 검출이 실패하면 영상이 빠져 n_items 가 줄어듭니다")
    groups = {}
    for (ds, m, l), r in latest.items():
        if r["n_items"]:
            groups.setdefault((ds, m), {})[l] = int(r["n_items"])
    bad = 0
    for (ds, m), lv in sorted(groups.items()):
        if len(set(lv.values())) > 1:
            bad += 1
            print(f"  [달라짐] {ds} {m}: " + ", ".join(f"{l}={lv[l]}" for l in LEVELS if l in lv))
    print("  모두 일관됨" if not bad else f"  -> {bad}개 그룹에서 n_items 가 해상도마다 다름(빠진 영상이 있다는 뜻)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=list(R.DATASETS), choices=list(R.DATASETS))
    ap.add_argument("--models", nargs="+", default=R.ALGOS + list(R.DEEP), choices=R.ALGOS + list(R.DEEP))
    ap.add_argument("--levels", nargs="+", default=LEVELS, choices=LEVELS)
    ap.add_argument("--end", type=float, default=1.0, help="시험용 subject 비율(0~1). cross 에서 PURE 는 1.0 권장")
    ap.add_argument("--protocol", choices=["same", "cross"], default="cross")
    ap.add_argument("--mmpd_ckpt", default="PURE", choices=["PURE", "UBFC-rPPG"])
    ap.add_argument("--mmpd_standard", action="store_true")
    ap.add_argument("--keep-cache", dest="keep_cache", action="store_true")
    ap.add_argument("--gpu-cycles", dest="gpu_cycles", type=int, default=3)
    ap.add_argument("--gpu-wait", dest="gpu_wait", type=int, default=300)
    ap.add_argument("--cuda-retries", dest="cuda_retries", type=int, default=2)
    ap.add_argument("--retry-failed", dest="retry_failed", action="store_true")
    ap.add_argument("--retry-pattern", dest="retry_pattern", default="")
    ap.add_argument("--dry-run", dest="dry_run", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--tag", default="", help="결과 집합 이름 뒤에 붙일 꼬리표. 예: --tag _algall 이면 _cross_res_algall 로 따로 저장(기존 결과와 섞이지 않음)")
    ap.add_argument("--subset", default="_cross_res", help="--summarize 에서 볼 결과 집합(기본 _cross_res)")
    args = ap.parse_args()

    if args.summarize:
        summarize(os.path.join(R.ROOT, "results", "all_results.csv"), args.subset)
        return
    if args.protocol == "cross" and args.end <= R.PURE_HELDOUT_BEGIN and "pure" in args.datasets:
        sys.exit(f"cross 에서 PURE 는 BEGIN {R.PURE_HELDOUT_BEGIN} 부터라 --end 는 {R.PURE_HELDOUT_BEGIN} 보다 커야 합니다")
    r = ResRunner(args)

    if args.dry_run:
        n = 0
        for ds in args.datasets:
            lv = levels_for(ds, args.levels)
            for m in [m for m in args.models if m in R.ALGOS]:
                for l in lv:
                    p = R.data_path(ds, l)
                    R.write_config(f"{ds}_{m}_{l}{r.sfx}", R.algo_config(ds, m, R.NATIVE_FPS, p or "MISSING", f"auto_{ds}_alg_{l}{r.sfx}", True, args.end, r.info, 0.0))
                    n += 1
                    if p is None:
                        print("  [없음] dataset 경로:", ds, l)
            for m in [m for m in args.models if m in R.DEEP]:
                ck = r.ckpt(ds, m)
                for l in lv:
                    p = R.data_path(ds, l)
                    exp = f"auto_{ds}_{m}_{l}{r.sfx}"
                    R.write_config(f"{ds}_{m}_{l}{r.sfx}_prep", R.deep_config(ds, m, p or "MISSING", exp, True, args.end, r.info, ck, r.begin(ds)))
                    R.write_config(f"{ds}_{m}_{l}{r.sfx}_test", R.deep_config(ds, m, p or "MISSING", exp, False, args.end, r.info, ck, r.begin(ds)))
                    n += 1
                    if p is None:
                        print("  [없음] dataset 경로:", ds, l)
                if not os.path.exists(os.path.join(R.ROOT, ck)):
                    print("  [없음] 가중치:", ck)
        print(f"설정 파일 생성 완료 (configs/infer_configs/auto/). 총 {n}개 실행 예정")
        return

    t0 = time.time()
    try:
        for ds in args.datasets:
            if any(m in R.ALGOS for m in args.models):
                r.run_algorithms(ds)
            for m in [m for m in args.models if m in R.DEEP]:
                r.run_deep(ds, m)
    except R.GpuUnavailable as e:
        print("\n[중단] GPU 오류가 계속됩니다 -> " + str(e))
        print("  실패로 기록하지 않았고 전처리 캐시도 지우지 않았습니다. GPU 를 복구한 뒤 같은 명령어를 다시 실행하면 이어집니다.")
        sys.exit(3)
    print(f"\n전부 끝 ({(time.time()-t0)/60:.0f}분). 표로 보기: python run_res_models.py --summarize")


if __name__ == "__main__":
    main()
