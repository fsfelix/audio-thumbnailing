# "A segment-based fitness measure for capturing repetitive structs of music recordings"
# by Meinard Müller, Peter Grosche, Nanzhu Jiang

import numpy as np
import librosa
from ssm import ssm

class audio_thumb_muller:
    def __init__(self, audio_path, t = 'chroma'):
        self.ssm = ssm(audio_path, t)

    def max_path_family(self, alpha):
        (N, N) = self.ssm.s.shape
        scores = []
        #M = alpha[1] - alpha[0] + 1
        M = alpha
        for seg_ini in range(N - M):
            S_a = self.ssm.s[:, seg_ini:seg_ini + alpha + 2]

            D = np.zeros((N, M+1))
            D[N-1, 2:M+1] = -np.inf

            for i in range(N-1, -1, -1):
                for j in range(M + 1):
                    if (j == 0 and i+1 < N):
                        D[i, j] = max(D[i+1,0], D[i+1,M])
                    elif (j == 1):
                        D[i, j] = D[i, 0] + S_a[0, N-1-i]
                    elif (i != N-1 and j != 0):
                        D[i, j] = S_a[j-1, (N-1-i)] + max(D[i+1, j-1] if i+1 < N and j-1 > 0 else 0,
                                        D[i+1, j-2] if i+1 < N and j-2 > 0 else 0 , 
                                        D[i+2, j-1] if i+2 < N and j-1 > 0 else 0)

            scores.append(D[0, M])
