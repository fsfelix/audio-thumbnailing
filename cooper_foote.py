# Audio Thumbnailing based on "Automatic Music Summarization via Similatiry Analysis",
# by Matthew Cooper and Jonathan Foote.

import numpy as np
import librosa
from ssm import ssm

class audio_thumb_cf:
    def __init__(self, audio_path, t = 'chroma'):
        self.ssm = ssm(audio_path, t)

    def score(self, m, n):
        return self.ssm(m, n)

    def score_normalized(self, q, r):
        return np.sum(np.sum(self.ssm.s[:,q : r + 1], axis = 0)/(self.ssm.s.shape[0]*(r - q + 1)))

    def score_Q(self, L, i):
        return self.score_normalized(i, i + L - 1)

    def score_max(self, L):
        N = self.ssm.s.shape[0]
        s = np.array([])

        for i in range(N - L + 1):
            s = np.append(s, self.score_Q(L, i))
        return s.argmax()

    def thumb(self, L):
        q_max = self.score_max(L)
        print("The best thumbnail for this song with length " + str(L) + " starts at frame: " + str(q_max))
