#!/usr/bin/env python3
"""
rPPG fps / 해상도 실험 엑셀(rPPG_FPS.xlsx, rppg_해상도.xlsx)을 읽어서 그래프를 PNG 로 저장한다.

    python plot_rppg.py                                   # 이 파일과 같은 폴더의 두 엑셀을 읽어 img/ 에 저장
    python plot_rppg.py --fps rPPG_FPS.xlsx --res rppg_res.xlsx --out img

필요: pip install matplotlib openpyxl numpy

만드는 그림
  1) lines   : 지표(MAE/Pearson/SNR) x 데이터셋 작은 선 그래프. 알고리즘(점선·원) vs 딥러닝(실선·네모)
  2) delta   : 원본 조건(30fps / 640x480) 대비 변화량. 출발선을 맞춰 누가 덜 무너지는지 비교
  3) heatmap : 방법 x 조건, 색 = MAE
  4) noface  : (해상도만) 해상도별 얼굴 검출 실패 비율

엑셀 읽는 규칙
  - 시트 이름 MAE / RMSE / MAPE / Pearson / SNR, 표는 위에서부터 UBFC, PURE, MMPD 순서('방법' 헤더 줄로 표를 세고 블록 제목은 보지 않음)
  - 'FAIL' 과 '-' 는 빈 값으로 처리(선이 거기서 끊김)
"""
import argparse
import os
import sys
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.lines import Line2D
from openpyxl import load_workbook

METHODS = ["GREEN", "POS", "CHROM", "PhysNet", "PhysFormer", "PhysMamba"]
ALGOS = METHODS[:3]
DATASETS = ["UBFC", "PURE", "MMPD"]
DS_TITLE = {"UBFC": "UBFC (PURE 가중치)", "PURE": "PURE (PURE 가중치, 딥러닝은 뒤 20%)", "MMPD": "MMPD (PURE 가중치)"}
EVAL_N = {"UBFC": 27, "PURE 알고리즘": 59, "PURE 딥러닝": 12, "MMPD": 132}      # 평가 영상 수(얼굴 검출 실패 비율 계산용)

# 색: 방법마다 고정(검증된 범주형 팔레트 슬롯 1~6 순서). 알고리즘/딥러닝은 선 모양·마커로도 구분(색만으로 구분하지 않음)
COLOR = dict(zip(METHODS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]))
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e3e2dd"
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

METRICS = [("MAE", "MAE (bpm, log축)", True), ("Pearson", "Pearson", False), ("SNR", "SNR (dB)", False)]


def setup_font():
    from matplotlib import font_manager
    have = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("AppleGothic", "Apple SD Gothic Neo", "Malgun Gothic", "NanumGothic", "Noto Sans CJK KR", "Noto Sans KR"):
        if name in have:
            plt.rcParams["font.family"] = name
            break
    else:
        print("[경고] 한글 글꼴을 못 찾았습니다. 글자가 네모로 보이면 한글 글꼴을 설치하세요.")
    plt.rcParams["axes.unicode_minus"] = False


def to_float(v):
    return float(v) if isinstance(v, (int, float)) else np.nan


def read_sheet(wb, sheet):
    """-> (조건 이름 리스트, {데이터셋: {방법: np.array}})"""
    ws = wb[sheet]
    levels, data, ds_idx = None, {d: {} for d in DATASETS}, -1
    for row in ws.iter_rows(values_only=True):
        first = row[0]
        if first == "방법":                       # 표마다 '방법' 헤더 줄이 하나씩 있음 -> 이걸로 표 시작을 센다(블록 제목은 안 봄)
            ds_idx += 1
            levels = [str(c) for c in row[2:] if c is not None]
            continue
        if isinstance(first, str) and first in METHODS and 0 <= ds_idx < len(DATASETS) and levels:
            data[DATASETS[ds_idx]][first] = np.array([to_float(v) for v in row[2:2 + len(levels)]])
    if levels is None or ds_idx != len(DATASETS) - 1:
        sys.exit(f"[오류] '{sheet}' 시트에서 표 {len(DATASETS)}개를 읽지 못했습니다(읽은 표 {ds_idx + 1}개).")
    return levels, data


def read_all(path):
    wb = load_workbook(path, data_only=True)
    out = {}
    for m, _, _ in METRICS:
        out[m] = read_sheet(wb, m)
    levels = out["MAE"][0]
    return levels, {m: out[m][1] for m, _, _ in METRICS}, wb


def read_noface(wb, levels):
    """NoFace 시트에서 데이터셋별 실패 영상 수(알고리즘 줄). 없으면 None"""
    if "NoFace" not in wb.sheetnames:
        return None
    res = {}
    for row in wb["NoFace"].iter_rows(values_only=True):
        if row[0] not in DATASETS or not isinstance(row[1], str):
            continue
        vals = np.array([to_float(v) for v in row[3:3 + len(levels)]])
        if "GREEN" in row[1]:                       # 알고리즘 줄(전처리 공유)
            res["PURE 알고리즘" if row[0] == "PURE" else row[0]] = vals
        elif row[0] == "PURE" and row[1] == "PhysNet":   # PURE 딥러닝은 평가 영상이 달라 따로 표시
            res["PURE 딥러닝"] = vals
    return res or None


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=9)


def plot_line(ax, x, y, m):
    algo = m in ALGOS
    ax.plot(x, y, color=COLOR[m], lw=2, ls="--" if algo else "-", marker="o" if algo else "s",
            ms=6, mec="#fcfcfb", mew=1.2, label=m)


def legend_handles():
    return [Line2D([0], [0], color=COLOR[m], lw=2, ls="--" if m in ALGOS else "-",
                   marker="o" if m in ALGOS else "s", ms=6, label=f"{m} ({'알고리즘' if m in ALGOS else '딥러닝'})")
            for m in METHODS]


def finish(fig, title, foot, path):
    fig.suptitle(title, fontsize=14, color=INK, y=0.995)
    fig.legend(handles=legend_handles(), loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=6, frameon=False, fontsize=10)
    if foot:
        fig.text(0.01, 0.005, "\n".join(textwrap.wrap(foot, 125)), fontsize=8.5, color=INK2, va="bottom", ha="left")
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)
    print("저장:", path)


def fig_lines(levels, data, xlabel, title, foot, path):
    fig, axes = plt.subplots(3, 3, figsize=(15, 11.5), facecolor="#fcfcfb")
    x = np.arange(len(levels))
    for i, (mt, ylabel, log) in enumerate(METRICS):
        for j, ds in enumerate(DATASETS):
            ax = axes[i][j]
            style(ax)
            for m in METHODS:
                plot_line(ax, x, data[mt][ds][m], m)
            if log:
                ax.set_yscale("log")
                lo, hi = ax.get_ylim()
                ticks = [t for t in (0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50) if lo * 0.9 <= t <= hi * 1.1]
                ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
                ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
                ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
                ax.grid(axis="y", color=GRID, lw=0.8)
            if mt == "Pearson":
                ax.set_ylim(-0.25, 1.05)
                ax.axhline(0, color=INK2, lw=0.8)
            if mt == "SNR":
                ax.axhline(0, color=INK2, lw=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(levels, rotation=30 if len(levels[0]) > 6 else 0)
            if i == 0:
                ax.set_title(DS_TITLE[ds], fontsize=11, color=INK)
            if j == 0:
                ax.set_ylabel(ylabel, fontsize=10, color=INK)
            if i == 2:
                ax.set_xlabel(xlabel, fontsize=9, color=INK2)
    fig.tight_layout(rect=(0, 0.0, 1, 0.93))
    finish(fig, title, foot, path)


def fig_delta(levels, data, xlabel, title, foot, path):
    fig, axes = plt.subplots(3, 3, figsize=(15, 11.5), facecolor="#fcfcfb")
    x = np.arange(len(levels))
    for i, (mt, ylabel, _) in enumerate(METRICS):
        unit = {"MAE": "bpm", "Pearson": "", "SNR": "dB"}[mt]
        for j, ds in enumerate(DATASETS):
            ax = axes[i][j]
            style(ax)
            for m in METHODS:
                y = data[mt][ds][m]
                finite = y[np.isfinite(y)]
                if finite.size == 0:
                    continue
                plot_line(ax, x, y - finite[0], m)
            ax.axhline(0, color=INK2, lw=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(levels, rotation=30 if len(levels[0]) > 6 else 0)
            if i == 0:
                ax.set_title(DS_TITLE[ds], fontsize=11, color=INK)
            if j == 0:
                good = "(위로 갈수록 나빠짐)" if mt == "MAE" else "(아래로 갈수록 나빠짐)"
                ax.set_ylabel(f"{mt} 변화량 {('(' + unit + ') ') if unit else ''}{good}", fontsize=9.5, color=INK)
            if i == 2:
                ax.set_xlabel(xlabel, fontsize=9, color=INK2)
    fig.tight_layout(rect=(0, 0.0, 1, 0.93))
    finish(fig, title, foot, path)


def fig_paper(levels, data, xlabel, path, width_in=3.5, height_in=3.3):
    """논문용(단 폭 3.5인치) 변화량 그림: MAE, SNR 2줄 x 데이터셋 3열. 글자를 처음부터 작은 크기로 그린다."""
    rc = {"font.size": 6, "axes.titlesize": 7, "axes.labelsize": 6, "xtick.labelsize": 5.5, "ytick.labelsize": 5.5}
    with plt.rc_context(rc):
        fig, axes = plt.subplots(2, 3, figsize=(width_in, height_in), facecolor="white")
        x = np.arange(len(levels))
        short = [l.replace("fps", "").split("x")[0] for l in levels]     # 30fps -> 30, 640x480 -> 640
        rows = [("MAE", "ΔMAE (bpm)"), ("SNR", "ΔSNR (dB)")]
        for i, (mt, ylabel) in enumerate(rows):
            for j, ds in enumerate(DATASETS):
                ax = axes[i][j]
                style(ax)
                ax.tick_params(labelsize=5.5, length=2, pad=1.5)
                for m in METHODS:
                    y = data[mt][ds][m]
                    finite = y[np.isfinite(y)]
                    if finite.size == 0:
                        continue
                    algo = m in ALGOS
                    ax.plot(x, y - finite[0], color=COLOR[m], lw=1.3, ls="--" if algo else "-",
                            marker="o" if algo else "s", ms=2.6, mec="white", mew=0.4)
                ax.axhline(0, color=INK2, lw=0.6)
                ax.set_xticks(x)
                if max(len(t) for t in short) > 2:        # 해상도처럼 3자리 라벨은 겹치므로 기울인다
                    ax.set_xticklabels(short, rotation=45, ha="right", rotation_mode="anchor")
                else:
                    ax.set_xticklabels(short)
                if i == 0:
                    ax.set_title(ds, fontsize=7, color=INK, pad=2)
                if j == 0:
                    ax.set_ylabel(ylabel, fontsize=6, labelpad=1.5)
                if i == 1:
                    ax.set_xlabel(xlabel, fontsize=5.5, labelpad=1.5)
        order = ["GREEN", "PhysNet", "POS", "PhysFormer", "CHROM", "PhysMamba"]   # 열 우선 채움 -> 윗줄 알고리즘, 아랫줄 딥러닝
        handles = [Line2D([0], [0], color=COLOR[m], lw=1.3, ls="--" if m in ALGOS else "-",
                          marker="o" if m in ALGOS else "s", ms=2.6, label=m) for m in order]
        fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=5.8,
                   handlelength=2.4, columnspacing=1.2, labelspacing=0.3, bbox_to_anchor=(0.5, -0.005))
        fig.tight_layout(rect=(0, 0.095, 1, 1), pad=0.4, w_pad=0.5, h_pad=0.6)
        fig.savefig(path, dpi=300, facecolor="white")
        plt.close(fig)
        print("저장:", path)


def fig_heatmap(levels, data, title, foot, path):
    cmap = LinearSegmentedColormap.from_list("blue", BLUES)
    allv = np.concatenate([data["MAE"][d][m] for d in DATASETS for m in METHODS])
    allv = allv[np.isfinite(allv)]
    norm = LogNorm(vmin=max(allv.min(), 0.1), vmax=allv.max())
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6), facecolor="#fcfcfb")
    fig.subplots_adjust(left=0.06, right=0.93, top=0.86, bottom=0.26, wspace=0.55)
    for ax, ds in zip(axes, DATASETS):
        mat = np.array([data["MAE"][ds][m] for m in METHODS])
        ax.imshow(np.ma.masked_invalid(mat), cmap=cmap, norm=norm, aspect="auto")
        ax.set_xticks(range(len(levels)))
        ax.set_xticklabels(levels, rotation=30 if len(levels[0]) > 6 else 0, fontsize=9, color=INK2)
        ax.set_yticks(range(len(METHODS)))
        ax.set_yticklabels(METHODS, fontsize=10, color=INK)
        ax.set_title(DS_TITLE[ds], fontsize=11, color=INK)
        ax.axhline(2.5, color="#fcfcfb", lw=4)           # 알고리즘 / 딥러닝 경계
        for r in range(mat.shape[0]):
            for c in range(mat.shape[1]):
                v = mat[r, c]
                if np.isfinite(v):
                    dark = norm(v) > 0.55
                    ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=9, color="#fcfcfb" if dark else INK)
                else:
                    ax.add_patch(plt.Rectangle((c - .5, r - .5), 1, 1, fill=True, facecolor="#e3e2dd", hatch="///", edgecolor="#c3c2b7", lw=0))
                    ax.text(c, r, "FAIL/-", ha="center", va="center", fontsize=8, color=INK2)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.01)
    cb.set_label("MAE (bpm, 로그 색 눈금)", fontsize=9, color=INK2)
    fig.suptitle(title, fontsize=14, color=INK)
    if foot:
        fig.text(0.01, 0.01, "\n".join(textwrap.wrap(foot, 150)), fontsize=8.5, color=INK2, va="bottom", ha="left")
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)
    print("저장:", path)


def fig_noface(levels, nf, path):
    colors = {"UBFC": "#6250d6", "PURE 알고리즘": "#e34948", "PURE 딥러닝": "#ec835a", "MMPD": "#52514e"}
    groups = [g for g in ("UBFC", "PURE 알고리즘", "PURE 딥러닝", "MMPD") if g in nf]
    fig, ax = plt.subplots(figsize=(10, 4.8), facecolor="#fcfcfb")
    style(ax)
    x = np.arange(len(levels))
    w = 0.2
    for k, ds in enumerate(groups):
        off = (k - (len(groups) - 1) / 2) * w
        pct = nf[ds] / EVAL_N[ds] * 100
        ax.bar(x + off, np.nan_to_num(pct, nan=0), w * 0.92, color=colors[ds], label=f"{ds} (평가 {EVAL_N[ds]}개)")
        for xi, p in zip(x + off, pct):
            if np.isfinite(p) and p > 0:
                ax.text(xi, p + 1.5, f"{p:.0f}%", ha="center", va="bottom", rotation=90, fontsize=8.5, color=INK)
    ax.set_ylim(0, 125)
    ax.set_xticks(x)
    ax.set_xticklabels(levels, rotation=30 if len(levels[0]) > 6 else 0)
    ax.set_ylabel("얼굴 검출 실패 영상 비율 (%)", color=INK)
    ax.set_xlabel("해상도 (MMPD 는 640x480, 480x360 단계 없음)", fontsize=9, color=INK2)
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    ax.set_title("해상도별 얼굴 검출 실패 비율 (0% 는 막대 없음)", fontsize=13, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)
    print("저장:", path)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fps", default=os.path.join(here, "rPPG_FPS.xlsx"))
    ap.add_argument("--res", default=os.path.join(here, "rppg_res.xlsx"))
    ap.add_argument("--out", default=os.path.join(here, "img"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    setup_font()

    foot_fps = foot_res = ""   # 그림 아래 각주는 넣지 않음

    if os.path.exists(a.fps):
        lv, d, _ = read_all(a.fps)
        fig_lines(lv, d, "fps (원본 30 → 5)", "fps 실험: 알고리즘(점선) vs 딥러닝(실선)", foot_fps, os.path.join(a.out, "fps_1_lines.png"))
        fig_delta(lv, d, "fps (원본 30 → 5)", "fps 실험: 30fps 대비 변화량 (출발선을 맞춘 비교)", foot_fps, os.path.join(a.out, "fps_2_delta.png"))
        fig_heatmap(lv, d, "fps 실험: MAE 히트맵", foot_fps, os.path.join(a.out, "fps_3_heatmap.png"))
        fig_paper(lv, d, "fps", os.path.join(a.out, "paper_fps.png"))
    else:
        print("[건너뜀] fps 엑셀이 없습니다:", a.fps)

    if os.path.exists(a.res):
        lv, d, wb = read_all(a.res)
        fig_lines(lv, d, "해상도 (가로x세로, 큰 것 → 작은 것)", "해상도 실험: 알고리즘(점선) vs 딥러닝(실선)", foot_res, os.path.join(a.out, "res_1_lines.png"))
        fig_delta(lv, d, "해상도 (가로x세로, 큰 것 → 작은 것)", "해상도 실험: 원본 해상도 대비 변화량 (출발선을 맞춘 비교)", foot_res, os.path.join(a.out, "res_2_delta.png"))
        fig_heatmap(lv, d, "해상도 실험: MAE 히트맵", foot_res, os.path.join(a.out, "res_3_heatmap.png"))
        fig_paper(lv, d, "해상도(가로 px)", os.path.join(a.out, "paper_res.png"))
        nf = read_noface(wb, lv)
        if nf:
            fig_noface(lv, nf, os.path.join(a.out, "res_4_noface.png"))
        else:
            print("[건너뜀] NoFace 시트를 못 읽어 얼굴 검출 실패 그림은 만들지 않았습니다.")
    else:
        print("[건너뜀] 해상도 엑셀이 없습니다:", a.res)


if __name__ == "__main__":
    main()
