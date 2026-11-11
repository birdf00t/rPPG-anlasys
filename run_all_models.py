#!/usr/bin/env python3
"""
UBFC / PURE / MMPD  x  fps(30/25/20/15/10/5)  x  6개 모델을 자동으로 돌리고 결과를 CSV 로 모은다.
rPPG-Toolbox 프로젝트 루트(main.py 가 있는 폴더)에 두고 실행한다.

  알고리즘(GREEN/POS/CHROM): make_fps_datasets.py 가 만든 dataset/<DS>/<fps>/ 데이터를 쓰고 FS=fps.
                             (데이터셋·fps마다 전처리 캐시를 한 번 만들어 세 방법이 공유. 방법마다 프로세스를 따로 띄워서
                              GREEN 이 5fps 에서 죽어도 POS/CHROM 은 계속 돈다)
  딥러닝(PhysNet/PhysFormer/PhysMamba): 원본(30fps) 전처리 캐시를 한 번 만들고, fps마다 캐시 입력에 "다운샘플->hold 업샘플"을
                             적용한 뒤 사전학습 가중치로 only_test. 정답/텐서 크기/FS(30)는 그대로.
                             (캐시 백업은 하드링크라 용량이 안 듦. 한 (데이터셋,모델)이 끝나면 캐시를 지움)

사용 예:
    python run_all_models.py --dry-run                     # 설정 파일만 만들고 계획 출력
    python run_all_models.py --end 0.15                    # 시험용: 데이터셋마다 일부 subject 만으로 전체 흐름 점검
    python run_all_models.py                               # 전체
    python run_all_models.py --datasets ubfc --models GREEN PhysNet --fps 30 15
    python run_all_models.py --summarize                   # 지금까지 결과를 표로 출력

끊겨도 같은 명령어를 다시 실행하면 끝난 작업은 건너뛴다(results/done/). 실패한 작업도 다시 안 하고 싶지 않으면 --retry-failed.
"""
import argparse
import csv
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.abspath(__file__))
FPS_ALL = [30, 25, 20, 15, 10, 5]
NATIVE_FPS = 30
ALGOS = ["GREEN", "POS", "CHROM"]
METRICS = ["MAE", "RMSE", "MAPE", "Pearson", "SNR"]
PURE_HELDOUT_BEGIN = 0.8   # PURE 가중치는 PURE 0.0~0.8 로 학습, 0.8~1.0 은 검증(툴박스 train_configs)

DATASETS = {
    # key: (toolbox DATASET 이름, 체크포인트 접두어, 전처리 프로세스 수)
    "ubfc": ("UBFC-rPPG", "UBFC-rPPG", 4),
    "pure": ("PURE", "PURE", 6),
    "mmpd": ("MMPD", None, 3),      # MMPD 는 MMPD 로 학습된 공개 가중치가 없어 --mmpd_ckpt 로 고른다(기본 PURE)
}

DEEP = {
    "PhysNet": dict(resize=72, chunk=128, use_last_epoch=True,
                    model={"DROP_RATE": 0.2, "NAME": "Physnet", "PHYSNET": {"FRAME_NUM": 128}}),
    "PhysFormer": dict(resize=128, chunk=160, use_last_epoch=False,
                       model={"DROP_RATE": 0.1, "NAME": "PhysFormer",
                              "PHYSFORMER": {"PATCH_SIZE": 4, "DIM": 96, "FF_DIM": 144, "NUM_HEADS": 4,
                                             "NUM_LAYERS": 12, "THETA": 0.7}}),
    "PhysMamba": dict(resize=128, chunk=128, use_last_epoch=False, model={"NAME": "PhysMamba"}),
}

MMPD_INFO_ALL = {"LIGHT": [1, 2, 3, 4], "MOTION": [1, 2, 3, 4], "EXERCISE": [1, 2], "SKIN_COLOR": [3, 4, 5, 6],
                 "GENDER": [1, 2], "GLASSER": [1, 2], "HAIR_COVER": [1, 2], "MAKEUP": [1, 2]}
MMPD_INFO_STANDARD = {"LIGHT": [1, 2, 3], "MOTION": [1], "EXERCISE": [2], "SKIN_COLOR": [3],
                      "GENDER": [1, 2], "GLASSER": [1, 2], "HAIR_COVER": [1, 2], "MAKEUP": [1, 2]}

WRAPPER = '''"""전처리 병렬 프로세스 수를 환경변수(RPPG_PREPROC_PROCS)로 제한하는 main.py 실행 래퍼 (run_all_models.py 가 생성)."""
import os
import runpy
import sys

sys.path.insert(0, os.getcwd())
import dataset.data_loader.BaseLoader as _bl

_orig = _bl.BaseLoader.multi_process_manager
_quota = int(os.environ.get("RPPG_PREPROC_PROCS", "8"))


def _patched(self, data_dirs, config_preprocess, multi_process_quota=_quota):
    return _orig(self, data_dirs, config_preprocess, multi_process_quota)


_bl.BaseLoader.multi_process_manager = _patched
runpy.run_path("main.py", run_name="__main__")
'''


# ----------------------------------------------------------------- 설정 생성
def crop_face():
    return {"DO_CROP_FACE": True, "BACKEND": "HC", "USE_LARGE_FACE_BOX": True, "LARGE_BOX_COEF": 1.5,
            "DETECTION": {"DO_DYNAMIC_DETECTION": False, "DYNAMIC_DETECTION_FREQUENCY": 30,
                          "USE_MEDIAN_FACE_BOX": False}}


def subject_root(path):
    """path 바로 아래에 subject* 폴더가 있으면 path, 아니면 한 단계 아래(예: DATASET_2)에서 찾는다."""
    if glob.glob(os.path.join(path, "subject*")):
        return path
    for sub in sorted(glob.glob(os.path.join(path, "*"))):
        if os.path.isdir(sub) and glob.glob(os.path.join(sub, "subject*")):
            return sub
    return None


def data_path(ds, fps):
    base = os.path.join("dataset", {"ubfc": "UBFC", "pure": "PURE", "mmpd": "MMPD"}[ds], str(fps))
    full = os.path.join(ROOT, base)
    if not os.path.isdir(full):
        return None
    if ds in ("ubfc", "mmpd"):
        sr = subject_root(full)
        return None if sr is None else "./" + os.path.relpath(sr, ROOT)
    return "./" + base


def algo_config(ds, method, fps, path, exp, do_prep, end, info, begin=0.0):
    data = {"FS": fps, "DATASET": DATASETS[ds][0], "DO_PREPROCESS": do_prep, "DATA_FORMAT": "NDHWC",
            "DATA_PATH": path, "CACHED_PATH": "./PreprocessedData", "EXP_DATA_NAME": exp, "BEGIN": begin, "END": end,
            "PREPROCESS": {"DATA_TYPE": ["Raw"], "DATA_AUG": ["None"], "LABEL_TYPE": "Raw", "DO_CHUNK": False,
                           "CHUNK_LENGTH": 180, "CROP_FACE": crop_face(), "RESIZE": {"H": 72, "W": 72}}}
    if ds == "mmpd":
        data["INFO"] = info
    return {"BASE": [""], "TOOLBOX_MODE": "unsupervised_method",
            "UNSUPERVISED": {"METHOD": [method], "METRICS": METRICS, "DATA": data},
            "INFERENCE": {"EVALUATION_METHOD": "FFT",
                          "EVALUATION_WINDOW": {"USE_SMALLER_WINDOW": False, "WINDOW_SIZE": 10}}}


def deep_config(ds, model, path, exp, do_prep, end, info, ckpt, begin=0.0):
    spec = DEEP[model]
    data = {"FS": NATIVE_FPS, "DATASET": DATASETS[ds][0], "DO_PREPROCESS": do_prep, "DATA_FORMAT": "NCDHW",
            "DATA_PATH": path, "CACHED_PATH": "./PreprocessedData", "EXP_DATA_NAME": exp, "BEGIN": begin, "END": end,
            "PREPROCESS": {"DATA_TYPE": ["DiffNormalized"], "LABEL_TYPE": "DiffNormalized", "DO_CHUNK": True,
                           "CHUNK_LENGTH": spec["chunk"], "CROP_FACE": crop_face(),
                           "RESIZE": {"H": spec["resize"], "W": spec["resize"]}}}
    if ds == "mmpd":
        data["INFO"] = info
    return {"BASE": [""], "TOOLBOX_MODE": "only_test",
            "TEST": {"METRICS": METRICS, "USE_LAST_EPOCH": spec["use_last_epoch"], "DATA": data},
            "DEVICE": "cuda:0", "NUM_OF_GPU_TRAIN": 1, "LOG": {"PATH": "runs/exp"},
            "MODEL": spec["model"],
            "INFERENCE": {"BATCH_SIZE": 4, "EVALUATION_METHOD": "FFT",
                          "EVALUATION_WINDOW": {"USE_SMALLER_WINDOW": False, "WINDOW_SIZE": 10},
                          "MODEL_PATH": ckpt}}


def write_config(name, cfg):
    d = os.path.join(ROOT, "configs", "infer_configs", "auto")
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, name + ".yaml")
    with open(p, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=None, allow_unicode=True)
    return os.path.relpath(p, ROOT)


# ----------------------------------------------------------------- 실행 / 결과
def run_main(cfg_rel, log_path, procs):
    env = os.environ.copy()
    env["RPPG_PREPROC_PROCS"] = str(procs)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as lf:
        return subprocess.run([sys.executable, "run_main_wrapped.py", "--config_file", cfg_rel], cwd=ROOT,
                              stdout=lf, stderr=subprocess.STDOUT, env=env).returncode


NUM = r"([-+]?(?:\d+\.?\d*(?:[eE][-+]?\d+)?|nan|inf))"


def parse_log(text):
    blocks, cur = [], None
    for line in text.splitlines():
        m = re.match(r"Used Unsupervised Method: (\w+)", line)
        if m:
            cur = {"method": m.group(1)}
            blocks.append(cur)
            continue
        m = re.match(rf"FFT (MAE|RMSE|MAPE|Pearson|SNR) \(FFT Label\): {NUM} \+/- {NUM}", line)
        if m:
            if cur is None:
                cur = {"method": None}
                blocks.append(cur)
            cur[m.group(1)] = (float(m.group(2)), float(m.group(3)))
    n = re.search(r"Preprocessed Dataset Length: (\d+)", text)
    return blocks, (int(n.group(1)) if n else None)


class GpuUnavailable(Exception):
    pass


ABORT_ERR = re.compile(r"CUDA error|cuDNN error|CUBLAS_STATUS|CUDNN_STATUS")
GPU_ERR_LOG = re.compile(r"CUDA error|cuDNN error|CUBLAS_STATUS|CUDNN_STATUS|CUDA out of memory|OutOfMemoryError")
GPU_ERR_STATUS = re.compile(r"CUDA|cuDNN|CUDNN|CUBLAS|out of memory", re.I)


def fail_reason(text):
    m = re.findall(r"(CUDA error: [^\n]*|cuDNN error: [^\n]*|[\w.]*OutOfMemoryError[^\n]*|CUDA out of memory[^\n]*)", text)
    if m:
        return m[-1].strip()[:160]
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for l in reversed(lines):
        if "Error" in l or "error" in l:
            return l[:160]
    return (lines[-1][:160] if lines else "로그 없음")


class Runner:
    def __init__(self, args):
        self.a = args
        self.res = os.path.join(ROOT, "results")
        self.done_dir = os.path.join(self.res, "done")
        self.csv_path = os.path.join(self.res, "all_results.csv")
        self.log_dir = os.path.join(ROOT, "logs", "auto")
        self.sfx = "" if args.end >= 1 else f"_e{int(round(args.end * 100)):03d}"
        self.cross = args.protocol == "cross"
        if self.cross:
            self.sfx += "_cross"   # 결과 집합 이름(교차 평가). 기존 full 결과와 섞이지 않음
        self.info = MMPD_INFO_STANDARD if args.mmpd_standard else MMPD_INFO_ALL
        os.makedirs(self.done_dir, exist_ok=True)
        with open(os.path.join(ROOT, "run_main_wrapped.py"), "w") as f:
            f.write(WRAPPER)

    # ---- 완료 기록
    def key(self, ds, method, fps):
        return f"{ds}__{method}__fps{fps}{self.sfx}"

    def is_done(self, k):
        p = os.path.join(self.done_dir, k + ".json")
        if not os.path.exists(p):
            return False
        status = json.load(open(p))["status"]
        if status != "ok":
            if self.a.retry_failed:
                return False
            pat = self.a.retry_pattern
            if pat and ((pat.lower() == "gpu" and GPU_ERR_STATUS.search(status)) or pat in status):
                return False
        return True

    def record(self, k, ds, family, method, fps, status, n, m, log):
        row = {"dataset": ds, "family": family, "method": method, "fps": fps, "subset": self.sfx or "full",
               "status": status, "n_items": n, "log": os.path.relpath(log, ROOT)}
        for name in METRICS:
            mean, std = (m or {}).get(name, (float("nan"), float("nan")))
            row[name] = mean
            row[name + "_std"] = std
        json.dump(row, open(os.path.join(self.done_dir, k + ".json"), "w"))
        new = not os.path.exists(self.csv_path)
        with open(self.csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if new:
                w.writeheader()
            w.writerow(row)

    def execute(self, k, ds, family, method, fps, cfg_rel, procs, retry_cfg_rel=None):
        log = os.path.join(self.log_dir, k + ".log")
        t0 = time.time()
        cur_cfg = cfg_rel
        for cycle in range(self.a.gpu_cycles):
            attempts = 0
            while True:
                rc = run_main(cur_cfg, log, procs)
                text = open(log, errors="replace").read()
                blocks, n = parse_log(text)
                pick = next((b for b in blocks if (family == "deep" or b.get("method") == method)
                             and all(x in b for x in METRICS)), None)
                attempts += 1
                # WSL2 에서 가끔 나는 일시적 GPU 오류(CUDA error: unknown error 등)는 잠깐 기다렸다가 다시 시도
                if pick or not GPU_ERR_LOG.search(text) or attempts > self.a.cuda_retries:
                    break
                print(f"  [재시도 {attempts}/{self.a.cuda_retries}] {ds} {method} {fps}{'fps' if str(fps).isdigit() else ''}: {fail_reason(text)} -> 30초 뒤 다시", flush=True)
                if retry_cfg_rel and "Preprocessed Dataset Length" in text:
                    cur_cfg = retry_cfg_rel      # 전처리는 이미 끝났으니 다시 하지 않고 추론만
                time.sleep(30)
            if pick or not ABORT_ERR.search(text):
                break
            if cycle < self.a.gpu_cycles - 1:
                print(f"  [GPU 오류 지속 {cycle+1}/{self.a.gpu_cycles}] {ds} {method} {fps}{'fps' if str(fps).isdigit() else ''}: {self.a.gpu_wait}초 쉰 뒤 다시", flush=True)
                time.sleep(self.a.gpu_wait)
            else:
                raise GpuUnavailable(f"{ds} {method} {fps}{'fps' if str(fps).isdigit() else ''}: {fail_reason(text)}")
        status = "ok" if pick else "fail: " + fail_reason(text)
        self.record(k, ds, family, method, fps, status, n, pick, log)
        mae = f"MAE={pick['MAE'][0]:.2f}" if pick else status
        print(f"  [{time.strftime('%H:%M:%S')}] {ds} {method} {fps}{'fps' if str(fps).isdigit() else ''} -> {mae} ({time.time()-t0:.0f}s, rc={rc})", flush=True)
        return pick is not None, text

    # ---- 캐시 정리
    def cache_dir(self, exp):
        return os.path.join(ROOT, "PreprocessedData", exp)

    def cleanup(self, exp):
        if self.a.keep_cache:
            return
        for p in (self.cache_dir(exp), self.cache_dir(exp) + "_BACKUP"):
            shutil.rmtree(p, ignore_errors=True)
        for p in glob.glob(os.path.join(ROOT, "PreprocessedData", "DataFileLists", exp + "_*.csv")):
            os.remove(p)

    # ---- 알고리즘
    def run_algorithms(self, ds):
        for fps in self.a.fps:
            todo = [m for m in ALGOS if m in self.a.models and not self.is_done(self.key(ds, m, fps))]
            if not todo:
                continue
            path = data_path(ds, fps)
            if path is None:
                print(f"[경고] dataset/{ds.upper()}/{fps} 를 못 찾아 알고리즘 {ds} {fps}fps 건너뜀")
                continue
            exp = f"auto_{ds}_alg_fps{fps}{self.sfx}"
            print(f"== {ds} 알고리즘 {fps}fps ({path}) ==", flush=True)
            prepped = os.path.isdir(self.cache_dir(exp)) and bool(glob.glob(
                os.path.join(ROOT, "PreprocessedData", "DataFileLists", exp + "_*.csv")))
            for m in todo:
                if not prepped:
                    self.cleanup(exp)
                cfg = algo_config(ds, m, fps, path, exp, not prepped, self.a.end, self.info, self.begin(ds))
                name = f"{ds}_{m}_fps{fps}{self.sfx}"
                ok, text = self.execute(self.key(ds, m, fps), ds, "algorithm", m, fps,
                                        write_config(name, cfg), DATASETS[ds][2])
                if "Preprocessed Dataset Length" in text:
                    prepped = True
            self.cleanup(exp)

    # ---- 딥러닝
    def begin(self, ds):
        """교차 평가(cross)에서 PURE 는 PURE 가중치의 학습 구간(0~0.8)을 피해 뒤 20% 만 평가한다."""
        return PURE_HELDOUT_BEGIN if (self.cross and ds == "pure") else 0.0

    def ckpt(self, ds, model):
        if self.cross and ds in ("ubfc", "pure"):
            src = "PURE"          # UBFC·PURE 모두 PURE 가중치(공식 PURE_UBFC-rPPG_* 평가 방식)
        else:
            src = DATASETS[ds][1] or self.a.mmpd_ckpt
        return f"./final_model_release/{src}_{model}_DiffNormalized.pth"

    def run_deep(self, ds, model):
        fps_list = [f for f in self.a.fps]
        todo = [f for f in fps_list if not self.is_done(self.key(ds, model, f))]
        if not todo:
            return
        path = data_path(ds, NATIVE_FPS)
        if path is None:
            print(f"[경고] dataset/{ds.upper()}/30 를 못 찾아 {model} {ds} 건너뜀")
            return
        ckpt = self.ckpt(ds, model)
        if not os.path.exists(os.path.join(ROOT, ckpt)):
            print(f"[경고] 가중치 없음: {ckpt} -> {model} {ds} 건너뜀")
            return
        exp = f"auto_{ds}_{model}_native{self.sfx}"
        cache = self.cache_dir(exp)
        backup = cache + "_BACKUP"
        print(f"== {ds} {model} (가중치 {os.path.basename(ckpt)}) ==", flush=True)

        # 파일 목록 CSV 는 툴박스가 전처리를 끝낸 뒤에 만든다. CSV 가 있으면 전처리가 끝난 캐시이므로 다시 하지 않는다
        # (GPU 오류로 중단된 뒤 재시작해도 수십 분짜리 전처리를 반복하지 않게)
        prep_done = os.path.isdir(cache) and bool(glob.glob(
            os.path.join(ROOT, "PreprocessedData", "DataFileLists", exp + "_*.csv")))
        name = f"{ds}_{model}{self.sfx}"
        test_cfg_rel = write_config(name + "_test", deep_config(ds, model, path, exp, False, self.a.end, self.info, ckpt, self.begin(ds)))
        if prep_done:
            print(f"  (전처리가 끝난 캐시를 재사용합니다: {os.path.basename(cache)})", flush=True)
            if not os.path.isdir(backup):
                make_backup(cache, backup)
        else:
            self.cleanup(exp)
            cfg = deep_config(ds, model, path, exp, True, self.a.end, self.info, ckpt, self.begin(ds))
            first_fps = NATIVE_FPS
            k = self.key(ds, model, first_fps)
            ok, text = self.execute(k, ds, "deep", model, first_fps, write_config(name + "_prep", cfg),
                                    DATASETS[ds][2], retry_cfg_rel=test_cfg_rel)
            self.save_pickle(exp, ds, model, first_fps)
            if "Preprocessed Dataset Length" not in text:
                print(f"  전처리 실패 -> {ds} {model} 나머지 fps 건너뜀")
                for f in fps_list:
                    if f != first_fps and not self.is_done(self.key(ds, model, f)):
                        self.record(self.key(ds, model, f), ds, "deep", model, f, "fail: 전처리 실패", None, {}, os.path.join(self.log_dir, k + ".log"))
                self.cleanup(exp)
                return
            make_backup(cache, backup)

        cfg_rel = test_cfg_rel
        for f in fps_list:
            if self.is_done(self.key(ds, model, f)):
                continue
            apply_fps(cache, backup, f)
            self.execute(self.key(ds, model, f), ds, "deep", model, f, cfg_rel, DATASETS[ds][2])
            self.save_pickle(exp, ds, model, f)
        apply_fps(cache, backup, NATIVE_FPS)
        self.cleanup(exp)

    def save_pickle(self, exp, ds, model, fps):
        files = glob.glob(os.path.join(ROOT, "runs", "exp", exp, "saved_test_outputs", "*.pickle"))
        if files:
            d = os.path.join(self.res, "pickles")
            os.makedirs(d, exist_ok=True)
            shutil.copy(max(files, key=os.path.getmtime), os.path.join(d, f"{ds}_{model}_fps{fps}{self.sfx}.pickle"))


# ----------------------------------------------------------------- 캐시 fps 시뮬레이션(hold)
def hold(frames, native, target):
    T = frames.shape[0]
    idx = np.clip((np.floor(np.arange(T) / (native / target)) * (native / target)).astype(int), 0, T - 1)
    return frames[idx]


def make_backup(cache, backup):
    os.makedirs(backup, exist_ok=True)
    for f in glob.glob(os.path.join(cache, "*_input*.npy")):
        dst = os.path.join(backup, os.path.basename(f))
        if not os.path.exists(dst):
            os.link(f, dst)


def apply_fps(cache, backup, fps):
    """백업(원본 30fps)에서 읽어 hold 를 적용한 파일로 캐시 입력을 교체한다(원자적 교체라 백업 하드링크는 안 건드림)."""
    for src in glob.glob(os.path.join(backup, "*_input*.npy")):
        live = os.path.join(cache, os.path.basename(src))
        tmp = live + ".tmp"
        if os.path.lexists(tmp):
            os.remove(tmp)
        if fps >= NATIVE_FPS:
            os.link(src, tmp)
        else:
            with open(tmp, "wb") as fh:
                np.save(fh, hold(np.load(src), NATIVE_FPS, fps))
        os.replace(tmp, live)


# ----------------------------------------------------------------- 요약
def summarize(path, subset="full"):
    if not os.path.exists(path):
        print("결과 파일이 아직 없습니다:", path)
        return
    all_rows = list(csv.DictReader(open(path)))
    rows = [r for r in all_rows if r["subset"] == subset]
    if not rows:
        print(f"'{subset}' 결과가 없습니다. 있는 결과 집합:", sorted({r['subset'] for r in all_rows}))
        return
    kind = "전체 실행" if subset == "full" else f"시험 실행({subset})"
    print(f"[결과 집합: {kind}]  (다른 집합은 --subset 으로 선택: {sorted({r['subset'] for r in all_rows})})")
    latest = {}
    for r in rows:
        latest[(r["dataset"], r["method"], r["fps"], r["subset"])] = r
    if subset.endswith("_cross"):
        base = subset[: -len("_cross")] or "full"
        print("교차 평가: UBFC 딥러닝과 PURE 는 이 집합의 결과, 그 외(UBFC 알고리즘·MMPD 전부)는 평가 조건이 같아 "
              f"'{base}' 결과를 가져와 채웁니다. PURE 알고리즘은 같은 뒤 20% 로 따로 돌린 결과가 있을 때만 표시됩니다.")
        for r in all_rows:
            if r["subset"] != base:
                continue
            ds, m = r["dataset"], r["method"]
            if ds == "pure" or (ds == "ubfc" and m in DEEP):
                continue
            latest.setdefault((ds, m, r["fps"], subset), r)
    for metric, arrow in (("MAE", "낮을수록 좋음"), ("Pearson", "1에 가까울수록 좋음"), ("SNR", "높을수록 좋음")):
        print(f"\n### {metric} ({arrow})")
        for ds in ("ubfc", "pure", "mmpd"):
            methods = [m for m in ALGOS + list(DEEP) if any(k[0] == ds and k[1] == m for k in latest)]
            if not methods:
                continue
            fps_cols = [str(f) for f in FPS_ALL]
            print(f"\n[{ds}]  " + "method".ljust(11) + "".join(f"{f+'fps':>9}" for f in fps_cols))
            for m in methods:
                cells = []
                for f in fps_cols:
                    r = next((v for k, v in latest.items() if k[0] == ds and k[1] == m and k[2] == f), None)
                    if r is None:
                        cells.append("-")
                    elif r["status"] != "ok":
                        cells.append("FAIL")
                    else:
                        cells.append(f"{float(r[metric]):.2f}")
                print("        " + m.ljust(11) + "".join(f"{c:>9}" for c in cells))

    print("\n### 평가 영상 수(n_items) 일관성 점검")
    print("같은 방법의 n_items 가 fps마다 다르면, 툴박스가 전처리 중 죽은 영상을 조용히 빼서 fps별 비교가 서로 다른 영상 집합으로 이뤄졌다는 뜻입니다.")
    groups = {}
    for (ds, method, fps, subset), r in latest.items():
        if r["n_items"]:
            groups.setdefault((ds, method, subset), {})[int(fps)] = int(r["n_items"])
    bad = 0
    for (ds, method, subset), fm in sorted(groups.items()):
        if len(set(fm.values())) > 1:
            bad += 1
            print(f"  [불일치] {ds} {method} {subset}: " + ", ".join(f"{f}fps={n}" for f, n in sorted(fm.items(), reverse=True)))
    print("  모두 일관됨" if not bad else f"  -> {bad}개 그룹 불일치 (위 데이터셋의 손상된 영상/파일을 먼저 확인하세요)")


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    ap.add_argument("--models", nargs="+", default=ALGOS + list(DEEP), choices=ALGOS + list(DEEP))
    ap.add_argument("--fps", nargs="+", type=int, default=FPS_ALL)
    ap.add_argument("--end", type=float, default=1.0, help="사용할 subject 비율(0~1). 시험용으로 0.15 정도")
    ap.add_argument("--mmpd_ckpt", default="PURE", choices=["PURE", "UBFC-rPPG"], help="MMPD 에 쓸 사전학습 가중치의 학습 데이터셋")
    ap.add_argument("--mmpd_standard", action="store_true", help="MMPD 를 툴박스 표준 조건(정지·운동 안 함·피부색3)만 평가. 기본은 전체 조건")
    ap.add_argument("--protocol", choices=["same", "cross"], default="same",
                    help="same: 기존(각 데이터셋 가중치). cross: UBFC·PURE 모두 PURE 가중치, PURE 는 뒤 20%%(0.8~1.0)만 평가. 결과는 '_cross' 집합으로 따로 저장")
    ap.add_argument("--keep-cache", dest="keep_cache", action="store_true")
    ap.add_argument("--gpu-cycles", dest="gpu_cycles", type=int, default=3, help="GPU 오류가 이어질 때 쉬었다 다시 하는 라운드 수(다 실패하면 기록/정리 없이 중단)")
    ap.add_argument("--gpu-wait", dest="gpu_wait", type=int, default=300, help="라운드 사이에 쉬는 초")
    ap.add_argument("--cuda-retries", dest="cuda_retries", type=int, default=2, help="CUDA 오류가 나면 자동으로 다시 시도할 횟수")
    ap.add_argument("--retry-failed", dest="retry_failed", action="store_true", help="실패한 작업을 전부 다시 시도")
    ap.add_argument("--retry-pattern", dest="retry_pattern", default="",
                    help="실패 사유에 이 문자열이 들어간 작업만 다시 시도. 'gpu' 는 CUDA/cuDNN/메모리 부족 같은 GPU 오류 전부. 구조적 실패(5fps 필터 등)는 건드리지 않음")
    ap.add_argument("--dry-run", dest="dry_run", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--subset", default="full", help="--summarize 에서 볼 결과 집합: full(전체 실행, 기본) 또는 시험 실행의 _e015")
    args = ap.parse_args()
    if args.protocol == "cross" and args.end <= PURE_HELDOUT_BEGIN and "pure" in args.datasets:
        sys.exit(f"cross 에서 PURE 는 BEGIN {PURE_HELDOUT_BEGIN} 부터라 --end 는 {PURE_HELDOUT_BEGIN} 보다 커야 합니다(시험은 --end 1.0 권장)")

    if args.summarize:
        summarize(os.path.join(ROOT, "results", "all_results.csv"), args.subset)
        return
    r = Runner(args)

    if args.dry_run:
        n = 0
        for ds in args.datasets:
            for m in [m for m in args.models if m in ALGOS]:
                for f in args.fps:
                    p = data_path(ds, f)
                    exp = f"auto_{ds}_alg_fps{f}{r.sfx}"
                    write_config(f"{ds}_{m}_fps{f}{r.sfx}", algo_config(ds, m, f, p or "MISSING", exp, True, args.end, r.info, r.begin(ds)))
                    n += 1
                    if p is None:
                        print("  [없음] dataset 경로:", ds, f)
            for m in [m for m in args.models if m in DEEP]:
                p = data_path(ds, NATIVE_FPS)
                exp = f"auto_{ds}_{m}_native{r.sfx}"
                ck = r.ckpt(ds, m)
                write_config(f"{ds}_{m}{r.sfx}_prep", deep_config(ds, m, p or "MISSING", exp, True, args.end, r.info, ck, r.begin(ds)))
                write_config(f"{ds}_{m}{r.sfx}_test", deep_config(ds, m, p or "MISSING", exp, False, args.end, r.info, ck, r.begin(ds)))
                n += len(args.fps)
                if not os.path.exists(os.path.join(ROOT, ck)):
                    print("  [없음] 가중치:", ck)
        print(f"설정 파일 생성 완료 (configs/infer_configs/auto/). 총 {n}개 실행 예정")
        return

    t0 = time.time()
    try:
        for ds in args.datasets:
            r.run_algorithms(ds) if any(m in ALGOS for m in args.models) else None
            for m in [m for m in args.models if m in DEEP]:
                r.run_deep(ds, m)
    except GpuUnavailable as e:
        print("\n[중단] GPU 오류가 계속됩니다 -> " + str(e))
        print("  실패로 기록하지 않았고 전처리 캐시도 지우지 않았습니다. GPU 를 복구한 뒤 같은 명령어를 다시 실행하면 이어집니다.")
        print("  복구: 윈도우 PowerShell 에서 'wsl --shutdown' 후 WSL 을 다시 열고, nvidia-smi 와 torch.cuda 가 되는지 확인하세요.")
        sys.exit(3)
    print(f"\n전부 끝 ({(time.time()-t0)/60:.0f}분). 결과: results/all_results.csv  |  표로 보기: python run_all_models.py --summarize")


if __name__ == "__main__":
    main()
