# Audio Thumbnailing based on "Automatic Music Summarization via Similatiry Analysis",
# by Matthew Cooper and Jonathan Foote.

import numpy as np
import librosa
import IPython.display
from ssm import ssm

class audio_thumb_cf:
    def __init__(self, audio_path, t = 'chroma', k = 10, smooth = 1, thresh = 1):
        self.ssm = ssm(audio_path, k, t, smooth, thresh)
        self.y, self.sr  = librosa.load(audio_path)
        self.time = 0

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

    def thumb_alpha(self, L):
        q_max = self.score_max(L)
#        self.thumb_frame = q_max
        print("The best thumbnail for this song with length " + str(round(self.frame_to_time(L), 2)) + " starts at time: " + str(round(self.frame_to_time(q_max), 2)) + "s")


    def thumb_time(self, time):
        L = self.time_to_frame(time)
        q_max = self.score_max(L)
        self.time = self.frame_to_time(q_max)
        print("The best thumbnail for this song with length " + str(round(self.frame_to_time(L), 2)) + " starts at time: " + str(round(self.frame_to_time(q_max), 2)) + "s")

    def frame_to_time(self, f):
        dt = self.ssm.duration/self.ssm.s.shape[0]
        return dt*f

    def time_to_frame(self, time):
        df = self.ssm.s.shape[0]/self.ssm.duration
        return int(df*time)

    def display(self):
        if self.time == 0:
            print("run thumb_time() or thumb_alpha()")
        else:
            frame = int(len(self.y)*self.time_to_frame(self.time)/self.ssm.duration)
            IPython.display.Audio(data = self.y[frame:], rate = self.sr)
