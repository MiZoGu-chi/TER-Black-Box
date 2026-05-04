

from collections import defaultdict

class NK_Landscape:

    def __init__(self, file, max_nb_turn=-1):

        f = open(file, "r")
        lignes = f.readlines()

        head = lignes[0].split()

        self.N = int(head[0])
        self.K = int(head[1])

        if (len(head) > 2):
            self.D = int(head[2])
        else:
            self.D = 2

        self.links = []
        for n in range(self.N):
            self.links.append([])
            for k in range(self.K + 1):
                self.links[n].append(int(lignes[1 + n * (self.K + 1) + k]))

            self.links[n].append([])
            for k in range(self.D ** (self.K + 1)):
                self.links[n][-1].append(float(lignes[1 + self.N * (self.K + 1) + n * (self.D ** (self.K + 1)) + k]))

        f.close()


    def eval(self,solution):

        fitness = 0

        for link in self.links:
            locus = []
            for i in link[:-1]:
                locus.append(solution[i])
            locus_index = 0
            for i in range(self.K + 1):
                locus_index += (self.D ** (self.K - i)) * locus[i]

            fitness += link[-1][int(locus_index)]

        return fitness/self.N

    def get_interaction(self):

        interactions = defaultdict(list)

        for link in self.links:
            neighborhood = sorted(link[:-1])

        pass