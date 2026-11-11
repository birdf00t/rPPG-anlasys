""" GREEN
Verkruysse, W., Svaasand, L. O. & Nelson, J. S.
Remote plethysmographic imaging using ambient light.
Optical. Express 16, 21434–21445 (2008).
"""

import numpy as np
import math
from scipy import signal # 신호 모듈
from scipy import linalg # 선형대수 모듈
from unsupervised_methods import utils # 비


def GREEN(frames):
    precessed_data = utils.process_video(frames) # 영상 전처리
    BVP = precessed_data[:, 1, :] # G 채널만 선택 [R,G,B]
    BVP = BVP.reshape(-1) # 1차원 배열로 변환
    return BVP

