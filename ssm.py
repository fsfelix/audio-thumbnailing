import numpy as np
import matplotlib

class ssm:
    def __init__(self, feat):
        self.s = self.create_ssm(feat)

    def create_ssm(self, feat):
        M, N = feat.shape
        print(M, N)
        s = np.zeros(N * N).reshape(N, N)
        for i in range(N):
            for j in range(N):
                s[i, j] = self.dist(feat[:, i], feat[:, j])
        return s

    def dist(self, f, g):
        return np.dot(f, g)

    def visualize(self):
        matplotlib.pcolor(self.s)
