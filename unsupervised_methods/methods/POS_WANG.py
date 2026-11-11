"""POS
Wang, W., den Brinker, A. C., Stuijk, S., & de Haan, G. (2017).
Algorithmic principles of remote PPG.
IEEE Transactions on Biomedical Engineering, 64(7), 1479-1491.

[핵심 아이디어]
피부색은 (혈류에 따른 미세한 색 변화) + (피부 고유색) + (조명 변화, 카메라 노이즈 등) 이 섞인 값이다.
POS는 "피부 고유색과 조명 변화는 R/G/B 세 채널에 거의 동일한 비율로 영향을 주지만,
혈류 변화(맥파)는 채널마다 다르게 영향을 준다"는 물리적 사실을 이용한다.

그래서 RGB 3차원 공간에서, "세 채널이 똑같이 변하는 방향(조명 등 잡음)"과 수직인
평면(POS 평면)으로 신호를 투영(projection)하면, 잡음은 없어지고 맥파 성분만 남는다.
이 평면을 정의하는 투영행렬이 코드의 [[0,1,-1],[-2,1,1]]이다.

CHROM(다른 파일)도 같은 원리(RGB를 정규화 후 선형결합)를 쓰지만,
투영에 사용하는 계수/윈도우 처리 방식이 달라 미묘하게 다른 결과를 낸다.
"""

import math

import numpy as np
from scipy import signal
from unsupervised_methods import utils

#GREEN과 비슷한 코드 부분(영상을 처리하는 함수)
def _process_video(frames):
    """각 프레임의 채널별(R,G,B) 평균값을 계산한다."""
    RGB = []
    for frame in frames:
        # frame.shape = (H, W, 3)
        # axis=0(세로,H) 방향으로 먼저 합산 -> (W, 3)
        # 그 결과를 다시 axis=0(이번엔 W) 방향으로 합산 -> (3,)
        # 즉 H, W를 모두 없애고 채널별 총합만 남김
        summation = np.sum(np.sum(frame, axis=0), axis=0)
        # 총합을 전체 픽셀 개수(H*W)로 나눠서 채널별 "평균"으로 변환
        RGB.append(summation / (frame.shape[0] * frame.shape[1]))
    # 리스트(프레임별 [R,G,B] 벡터들)를 numpy 배열로 변환 -> shape (num_frames, 3)
    return np.asarray(RGB)


def POS_WANG(frames, fs):
    # 슬라이딩 윈도우 길이(초 단위). 논문에서 제안하는 기본값 1.6초
    WinSec = 1.6

    # 프레임별 RGB 평균 시계열 계산 -> shape (N, 3)
    RGB = _process_video(frames)

    # 전체 프레임 수
    N = RGB.shape[0]

    # 최종 BVP 신호를 누적(overlap-add)할 배열, 처음엔 전부 0
    H = np.zeros((1, N))

    # WinSec(초)를 fs(FPS)로 곱해서 "윈도우 길이"를 프레임 개수로 환산 (올림)
    l = math.ceil(WinSec * fs)

    # n을 0부터 N-1까지 한 프레임씩 늘려가며,
    # 매 시점마다 "최근 l프레임(윈도우)"에 대해 POS 계산을 수행
    for n in range(N):
        m = n - l  # 현재 윈도우의 시작 인덱스
        if m >= 0:  # 윈도우를 채울 만큼 프레임이 쌓였을 때만 계산

            # ---- 1) 시간적 정규화 (Temporal normalization) ----
            # 윈도우 구간 RGB를 그 구간의 채널별 평균으로 나눔
            # -> 채널별 평균이 1이 되도록 스케일 맞춤 (조명 밝기 영향 축소)
            Cn = np.true_divide(RGB[m:n, :], np.mean(RGB[m:n, :], axis=0))

            # 전치 + 행렬(matrix)로 변환 -> shape (3, l) : 행=채널, 열=시간
            Cn = np.mat(Cn).H

            # ---- 2) POS 평면으로 투영 ----
            # 논문에서 유도된 투영행렬 [[0,1,-1],[-2,1,1]]을 곱해
            # 피부의 혈류 변화가 실리는 평면 성분(S[0], S[1])을 추출
            S = np.matmul(np.array([[0, 1, -1], [-2, 1, 1]]), Cn)

            # ---- 3) 두 투영 성분을 표준편차 비율로 가중합 ----
            # 두 신호의 진폭(스케일) 차이를 보정해 신호 대 잡음비를 높임
            h = S[0, :] + (np.std(S[0, :]) / np.std(S[1, :])) * S[1, :]

            # 이 윈도우 구간 h의 평균값
            mean_h = np.mean(h)

            # h에서 평균을 빼서 0을 중심으로 진동하게 만듦 (DC 성분 제거)
            # (아래 for문은 h = h - mean_h 를 풀어서 쓴 것과 동일)
            for temp in range(h.shape[1]):
                h[0, temp] = h[0, temp] - mean_h

            # ---- 4) Overlap-Add ----
            # 이번 윈도우에서 계산된 h를 전체 신호 H의 해당 구간에 더해서 누적
            # 윈도우가 한 프레임씩 겹치며 이동하므로 같은 시점이 여러 번 더해지고,
            # 이 과정에서 노이즈가 어느 정도 상쇄/평균화됨
            H[0, m:n] = H[0, m:n] + (h[0])

    # 누적이 끝난 H가 아직 다듬어지지 않은 BVP
    BVP = H

    # ---- 5) Detrend ----
    # 천천히 변하는 저주파 추세(조명 변화, 카메라 흔들림 등) 제거
    BVP = utils.detrend(np.mat(BVP).H, 100)

    # 다시 1차원 배열 형태로 변환
    BVP = np.asarray(np.transpose(BVP))[0]

    # ---- 6) 밴드패스 필터 설계 ----
    # 0.75~3Hz 대역만 통과 (심박수로 환산 시 45~180 BPM에 해당하는 범위)
    # 이 범위를 벗어나는 저주파/고주파 잡음을 제거하기 위함
    b, a = signal.butter(1, [0.75 / fs * 2, 3 / fs * 2], btype='bandpass')

    # 설계한 필터를 신호에 적용 (filtfilt: 양방향 필터링으로 위상 지연 없이 처리)
    BVP = signal.filtfilt(b, a, BVP.astype(np.double))

    # 최종 정제된 BVP(맥파) 신호 반환
    return BVP