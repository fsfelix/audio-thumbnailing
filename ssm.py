import numpy as np

class ssm:

    def __init__(self, feat):
        self.s = create_ssm(feat)

    def create_ssm(feat):
        M, N = feat.shape
        s = np.zeros(M * N).reshape(M, N)
        for i in range(M):
            for j in range(N):
                s[i, j] = self.dist(feat[i], feat[j])
        return s

    def dist(f, g):
        return np.dot(f, g)

    def visualize(s):
        pass


