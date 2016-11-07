# "A segment-based fitness measure for capturing repetitive structs of music recordings"
# by Meinard Müller, Peter Grosche, Nanzhu Jiang

import numpy as np
import librosa
from ssm import ssm

class audio_thumb_muller:
    def __init__(self, audio_path, t = 'chroma'):
#        self.ssm = ssm(audio_path, t)
        S = np.array([[0.1, 1, 0.3, 0.4, 0.5],
                      [0.3, 0.3, 1, 0.6, 0.5],
                      [0.6, 0.4, 0.6, 1, 0.3],
                      [0.7, 0.5, 0.5, 0.3, 1],
                      [0.1, 1, 0.3, 0.4, 0.5],
                      [0.3, 0.3, 1, 0.6, 0.5],
                      [0.6, 0.4, 0.6, 1, 0.3],
                      [0.7, 0.5, 0.5, 0.3, 1]])
        self.max_path_family(S, 4)

    def calculate_path(self, pos, D, Sa):
        pass
    def calculate_covarege(self, path, alpha):
        pass
    def calculate_score(self, path_family, score_opt, alpha):
        pass

    def calculate_fitness(gamma, mi):
        return 2*(gamma * mi/(gamma + mi))

    def max_path_family(self, S, alpha):
        [N, M] = S.shape
        D = np.zeros((N, alpha + 1))
        for low in range(0, M - alpha + 1):
            Sa = S[:, low:low + alpha]
            D[N - 1, 2:alpha + 1] = -np.inf

            for i in range(N - 1, -1, -1):
                for j in range(alpha + 1):
                    if (j == 0 and i + 1 < N):
                        D[i, j] = max(D[i + 1,0], D[i + 1, alpha])
                    elif (j == 1):
                        D[i, j] = D[i, 0] + Sa[N - 1 - i, 0]
                    elif (i != N - 1 and j != 0):
                        D[i, j] = Sa[N - 1 - i, j - 1] + max(D[i + 1, j - 1] if i + 1 < N and j - 1 > 0 else 0,
                                                             D[i + 1, j - 2] if i + 1 < N and j - 2 > 0 else 0,
                                                             D[i + 2, j - 1] if i + 2 < N and j - 1 > 0 else 0)
            possible_max = [D[0, alpha], D[N-1,0]]
            arg = np.argmax(possible_max)
            #path_family = self.calculate_path(possible_max[arg], D, Sa)
            #gamma = self.calculate_coverage(path, alpha)
            #mi = self.calculate_score(path_family, score_opt, alpha)
            #fitness = self.calculate_fitness(gamma, mi)
            #fitness_list.append(fitness, low)

            print("Matriz de similaridade:")
            print(Sa)
            print("Matriz de custos:")
            print(D)
            print("Score optimal:")
            #print(score_opt)
            print("Vector scores:")
            #print(scores_alpha)
            print("Thumbnail: ")
            #(max_fit, max_low) = max(fitness_list, key = lambda item:item[0])


oi = audio_thumb_muller("qaluqer coisa")
