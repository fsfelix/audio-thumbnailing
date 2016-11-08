# "A segment-based fitness measure for capturing repetitive structs of music recordings"
# by Meinard Müller, Peter Grosche, Nanzhu Jiang

import numpy as np
import librosa
from ssm import ssm

class audio_thumb_muller:
    def __init__(self, audio_path, t = 'chroma'):
        self.ssm = ssm(audio_path, t)
        # S = np.array([[0.1, 1, 0.3, 0.4, 0.5],
        #               [0.3, 0.3, 1, 0.6, 0.5],
        #               [0.6, 0.4, 0.6, 1, 0.3],
        #               [0.7, 0.5, 0.5, 0.3, 1],
        #               [0.1, 1, 0.3, 0.4, 0.5],
        #               [0.3, 0.3, 1, 0.6, 0.5],
        #               [0.6, 0.4, 0.6, 1, 0.3],
        #               [0.7, 0.5, 0.5, 0.3, 1]])
        # self.max_path_family(S, 4)

    def calculate_path(self, pos, D):
        [N, M] = D.shape
        path = []
        path.append(pos)
        (i, j) = pos
        ok = 1
        while(ok):
            tmp = []
            if j == 0 and i < N - 1:
                path.append((i + 1, j) if D[i + 1, j] > D[i + 1, M - 1] else (i + 1, M - 1))
                (i, j) = (i + 1, j) if D[i + 1, j] > D[i + 1, M - 1] else (i + 1, M - 1)
            elif j == 1 and i < N - 1:
                (i, j) = (i, j - 1)
            else:
                if i + 1 < N and j - 1 >= 0: tmp.append((D[i + 1, j - 1], (i + 1, j - 1)))
                if i + 1 < N and j - 2 >= 0: tmp.append((D[i + 1, j - 2], (i + 1, j - 2)))
                if i + 2 < N and j - 1 >= 0: tmp.append((D[i + 2, j - 1], (i + 2, j - 1)))

                if tmp != []:
                    pos = max(tmp)
                    path.append(pos[1])
                    i, j = pos[1]
                else:
                    ok = 0
            if i == 0 and j == 0:
                ok = 0

        # print("OLHA O CAMINHO AE RAPAZE")
        # print(path)

        new_path = [(N - 1 - x[0], x[1] - 1) for x in path]
        return new_path

    ## Verificar isso aqui!
    def calculate_coverage(self, path_family, alpha, N):
        gamma = len(path_family) - len(path_family)/alpha
        return (gamma - alpha)/N

    def calculate_score(self, path_family, score_opt, alpha):
        return (score_opt - alpha)/len(path_family)

    def calculate_fitness(self, gamma, mi):
        return 2*(gamma * mi/(gamma + mi))

    def max_path_family(self, S, alpha):
        [N, M] = S.shape
        D = np.zeros((N, alpha + 1))
        fitness_list = []
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
            possible_max = [D[0, alpha], D[0, 0]]
            score_opt = max(possible_max)
            arg = np.argmax(possible_max)
            path_family = self.calculate_path((0, 0) if arg else (0, alpha), D)
            gamma = self.calculate_coverage(path_family, alpha, N)
            print("gamma: " + str(gamma))
            mi = self.calculate_score(path_family, score_opt, alpha)
            print("mi: " + str(mi))
            fitness = self.calculate_fitness(gamma, mi)
            print("fitness: " + str(fitness))
            fitness_list.append((fitness, low))

            # print("Matriz de similaridade:")
            # print(Sa)
            # print("Matriz de custos:")
            # print(D)
            #print("Score optimal:")
            #print(score_opt)
            #print("Vector scores:")
            #print(scores_alpha)
        return fitness_list

    def thumb(self, alpha):
        fitness_list = self.max_path_family(self.ssm.s, alpha)
        (max_fit, max_low) = max(fitness_list, key = lambda item:item[0])
        print("Thumbnail init: " + str(self.frame_to_time(max_low)) + " with: " + str(max_fit) + " of fitness value.")

    def frame_to_time(self, f):
        dt = self.ssm.duration/self.ssm.s.shape[0]
        return dt*f
