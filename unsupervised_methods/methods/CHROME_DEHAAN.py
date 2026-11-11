# The Chrominance Method from: De Haan, G., & Jeanne, V. (2013).
# Robust pulse rate from chrominance-based rPPG. IEEE Transactions on Biomedical Engineering, 60(10), 2878-2886.
# DOI: 10.1109/TBME.2013.2266196
#
# [핵심 아이디어]
# POS와 마찬가지로 "조명/피부색 잡음은 RGB 채널에 비슷하게 영향을 주지만,
# 혈류(맥파) 신호는 채널마다 다르게 영향을 준다"는 원리를 이용한다.
#
# 다만 CHROM은 PCA/투영행렬 대신, 생리학적으로 미리 정해둔 두 개의 "색차(chrominance)" 조합
#   Xs = 3R - 2G
#   Ys = 1.5R + G - 1.5B
# 을 사용한다. 두 신호 모두 조명 변화에는 비슷하게 반응하지만 맥파 성분의 크기가 다르기 때문에,
# 표준편차 비율(Alpha)로 스케일을 맞춰 두 신호를 빼주면(Xf - Alpha*Yf) 조명 성분은 상쇄되고
# 맥파 성분만 남는다.
#
# 또한 CHROM은 밴드패스 필터를 1.6초짜리 짧은 구간(window)마다 새로 걸고(overlap-add 방식으로 이어붙임),
# POS는 매 프레임마다 최근 1.6초 구간에 대해 계산을 반복한다는 점에서 구현 방식이 다르다.
# (그래서 낮은 fps에서는 이 1.6초 윈도우 안에 프레임 수가 너무 적어져 CHROM이 먼저 깨진다.)
import numpy as np
import math
from scipy import signal

import unsupervised_methods.utils as utils
def CHROME_DEHAAN(frames, FS):
    LPF = 0.7      # 저역 통과 컷오프 주파수(Hz) 42 bpm
    HPF = 2.5      # 고역 통과 컷오프 주파수(Hz) 150bpm
    # hz 초당 , bpm 분당 , bpm= hz*60

    #정규화 때문에 슬라이딩 윈도우가 필요함
    WinSec = 1.6   # 슬라이딩 윈도우 길이(초)

    RGB = process_video(frames)   # 프레임별 R,G,B 평균 시계열 -> shape (FN, 3)
    FN = RGB.shape[0]             # 전체 프레임 수
    NyquistF = 1/2*FS             # 1/2*FPS, 기존 FPS에 절반이 손실없이 표현 가능한 하한선 진동 횟수
    # 3차 버터워스 밴드패스 필터 설계 (컷오프를 나이퀴스트 주파수로 정규화)
    B, A = signal.butter(3, [LPF/NyquistF, HPF/NyquistF], 'bandpass')

    WinL = math.ceil(WinSec*FS)   # 윈도우 길이를 "초" -> "프레임 개수"로 환산 (올림)
    if(WinL % 2):
        WinL = WinL+1              # 홀수면 짝수로 맞춤 (아래에서 절반씩 나눠 쓰기 때문)
    NWin = math.floor((FN-WinL//2)/(WinL//2))  # 50%씩 겹치며 윈도우를 몇 번 반복할지
    WinS = 0                       # 윈도우 시작 인덱스
    WinM = int(WinS+WinL//2)       # 윈도우 중간 인덱스
    WinE = WinS+WinL               # 윈도우 끝 인덱스
    totallen = (WinL//2)*(NWin+1)  # 최종 출력 신호의 길이
    S = np.zeros(totallen)         # 최종 신호를 담을 배열 (0으로 초기화)

    for i in range(NWin):
        # 이번 윈도우 구간의 채널별 평균값 (밝기 기준선)
        RGBBase = np.mean(RGB[WinS:WinE, :], axis=0)
        RGBNorm = np.zeros((WinE-WinS, 3))
        for temp in range(WinS, WinE):
            # 각 프레임의 RGB를 RGBBase로 나눠서 정규화
            RGBNorm[temp-WinS] = np.true_divide(RGB[temp], RGBBase)

        # CHROM의 색차(chrominance) 신호 계산: 정규화된 R,G,B의 선형결합
        Xs = np.squeeze(3*RGBNorm[:, 0]-2*RGBNorm[:, 1])
        Ys = np.squeeze(1.5*RGBNorm[:, 0]+RGBNorm[:, 1]-1.5*RGBNorm[:, 2])
        # 두 신호에 밴드패스 필터 적용 (심박수 대역만 남김)
        Xf = signal.filtfilt(B, A, Xs, axis=0)
        Yf = signal.filtfilt(B, A, Ys)

        # 두 신호의 표준편차 비율로 가중치 계산 (스케일 보정)
        Alpha = np.std(Xf) / np.std(Yf)
        SWin = Xf-Alpha*Yf   # 이 윈도우 구간의 펄스 신호
        # 윈도우 경계가 급격히 끊기지 않도록 Hann 윈도우(끝이 0에 가깝게 줄어드는 창) 적용
        SWin = np.multiply(SWin, signal.windows.hann(WinL))

        temp = SWin[:int(WinL//2)]  # (사용되지 않는 임시 변수)
        # 이번 윈도우 앞쪽 절반은 이전 윈도우의 뒤쪽 절반과 겹치므로 더해줌 (overlap-add)
        S[WinS:WinM] = S[WinS:WinM] + SWin[:int(WinL//2)]
        # 이번 윈도우 뒤쪽 절반은 아직 채워지지 않았으므로 그대로 대입
        S[WinM:WinE] = SWin[int(WinL//2):]
        # 다음 반복을 위해 윈도우를 절반만큼 이동
        WinS = WinM
        WinM = WinS+WinL//2
        WinE = WinS+WinL
    BVP = S   # 최종 CHROM 기반 BVP(맥파) 신호
    return BVP

def process_video(frames):
    "Calculates the average value of each frame."
    RGB = []
    for frame in frames:
        # H, W 축을 모두 합산 -> 채널별 총합
        sum = np.sum(np.sum(frame, axis=0), axis=0)
        # 총합을 픽셀 개수(H*W)로 나눠서 채널별 평균으로 변환
        RGB.append(sum/(frame.shape[0]*frame.shape[1]))
    return np.asarray(RGB)   # (num_frames, 3) 형태로 변환