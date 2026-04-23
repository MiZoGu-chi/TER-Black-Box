import numpy as np
import random
from nk import NK_Landscape
class TabuSearchAlgorithm:

    def __init__(self, problem, dim):

        self.dim = dim
        self.problem = problem


    def init(self, params1, params2):

        self.params1 = params1
        self.params2 = params2

        self.current_solution = np.random.randint(0,2,self.dim)

        self.current_fitness = self.problem.eval(self.current_solution)

        self.num_evals = 1


        self.best_fitness = self.current_fitness


        self.tabuTenure = np.zeros((self.dim))

        self.best_solution = None
        self.best_fitness = float("-inf")


    def solve(self, budget):

        iter_ = 0

        while self.num_evals < budget:

            best_delta = -float("inf")
            best_x = -1
            trouve = 1

            for x in range(self.dim):

                # Test de l'inversion du bit x sur la solution actuelle
                self.current_solution[x] = 1 - self.current_solution[x]

                # Calcul de la fitness de la nouvelle solution générée par l'inversion du bit x
                new_fitness = self.problem.eval(self.current_solution)

                # remet la solution à son état original
                self.current_solution[x] = 1 - self.current_solution[x]

                # incrémentation du nb d'utilisation de la boîte noire
                self.num_evals += 1

                # gain de la nouvelle fitness par rapport à l'ancienne
                delta = new_fitness - self.current_fitness

                # 1er condition : on vérifie si le bit x est tabou ou non, c'est à dire si son état d'expiration (tenure) est inférieur ou égale à l'itération courante
                # 2ème condition : si la nouvelle fitness est strictemment supérieur à notre meilleur fitness alors on ignore la 1er condition (critère d'aspiration)
                if ((self.tabuTenure[x] <= iter_) or (new_fitness > self.best_fitness)):

                    if (delta > best_delta):
                        best_x = x # on mémoire l'index du bit pour lequel on a le meilleur delta
                        best_delta = delta # mise à jour du nouveau meilleur delta
                        trouve = 1 # remet le compteur de doublon à 1

                    # Cas d'égalité
                    elif (delta == best_delta):

                        trouve += 1 # incrémentaion du compteur de doublon
                        if (random.randint(0, trouve) == 0): # tire au sort un des bits avec lequel on obtient ce delta, permet d'être non déterministe
                            best_x = x

                if (self.num_evals >= budget):
                    break

            # à la sortie du for : les N voisins ont été analysé ; le meilleur mouvement autorisé (bit inversé) est stocké ds best_x a et son gain ds best_delta    

            self.current_fitness += best_delta # équivalent à faire un appel à la boite noire pour obtenir le nouveau fitness de notre novelle solution (sans le cout)
            self.current_solution[best_x] = 1 - self.current_solution[best_x] # on inverse le bit qui nous déplace à une meilleur position


            if (self.current_fitness > self.best_fitness): # mise à jour de la fitness et coppie de la nouvelle solution
                self.best_fitness = self.current_fitness
                self.best_solution = np.copy(self.current_solution)


            # punition du bit, param1 (constant) + param2 (variable) + iter
            self.tabuTenure[best_x] = int(self.params1) + random.randint(0, int(self.params2)) + iter_

            iter_ += 1


        return self.best_solution, self.best_fitness




budget = 10000
dim = 64

params1 = int(dim // 10)
params2 = 10

problem = NK_Landscape("instances/64/2/nk_64_2_0.txt")

solution = np.random.randint(0,2,dim)

algoTabu = TabuSearchAlgorithm(problem, dim)
algoTabu.init(params1, params2)

best_solution, best_fitness  = algoTabu.solve(budget)

print(best_solution)
print(best_fitness)

