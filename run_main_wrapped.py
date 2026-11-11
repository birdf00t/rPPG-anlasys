"""전처리 병렬 프로세스 수를 환경변수(RPPG_PREPROC_PROCS)로 제한하는 main.py 실행 래퍼 (run_all_models.py 가 생성)."""
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
