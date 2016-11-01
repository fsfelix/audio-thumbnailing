from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import librosa

class ssm:
    def __init__(self, audio_path, t, normalized = 1):
        self.audio, self.sr = self.read_audio(audio_path)
        self.s = self.create_ssm(self.calculate_feat(t), normalized)
        self.duration = self.duration()

    def read_audio(self, audio_path):
        audio, sr = librosa.load(audio_path)
        return audio, sr

    def calculate_feat(self, t = 'chroma'):
        if t == 'chroma':
            return librosa.feature.chroma_stft(y = self.audio, sr = self.sr, n_fft = 2048)

    def create_ssm(self, feat, normalized):
        if normalized == 1:
            s_norm = np.linalg.norm(feat, axis = 0)
            s_norm[s_norm == 0] = 1
            feat   = np.abs(feat/s_norm)

        s = np.dot(feat.T, feat)
        return s

    def create_ssm_old(self, feat):
        M, N = feat.shape
        s = np.zeros(N * N).reshape(N, N)
        for i in range(N):
            for j in range(N):
                s[i, j] = self.dist(feat[:, i], feat[:, j])
        return s

    def dist(self, f, g):
        return np.dot(f, g)

    def score(self, m, n):
        return self.ssm.s(m, n)

    def visualize(self):
        plt.pcolor(self.s)

    def visualize_img(self):
#        bin_s = self.s[self.s < 100]
        S = Image.fromarray(self.s * 100)
        S.show()

    def duration(self):
        return librosa.core.get_duration(self.audio, self.sr)
