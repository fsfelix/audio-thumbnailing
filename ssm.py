from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import librosa

class ssm:
    def __init__(self, audio_path, t):
        self.audio, self.sr = self.read_audio(audio_path)
        self.s = self.create_ssm(self.calculate_feat(t))
        self.duration = self.duration()

    def read_audio(self, audio_path):
        audio, sr = librosa.load(audio_path)
        return audio, sr

    def calculate_feat(self, t = 'chroma'):
        if t == 'chroma':
            return librosa.feature.chroma_stft(y = self.audio, sr = self.sr, n_fft = 2048)

    def create_ssm(self, feat):
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

    def visualize(self):
        plt.pcolor(self.s)

    def visualize_img(self):
        S = Image.fromarray(ssm.s * 100)
        S.show()

    def duration(self):
        return librosa.core.get_duration(self.audio, self.sr)
