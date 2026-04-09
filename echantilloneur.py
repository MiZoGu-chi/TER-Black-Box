import numpy as np

from walsh import apply_walsh
from walsh import generate_masks

def random_black_box(x):
    return np.random.random() 

def generate_dataset(N, NB, masks, fun):
    X_train = []
    Y_train = []

    for _ in range(nb_exemples):
        random_sol = np.random.randint(2, size=N) # N = 5, [0|1, 0|1, 0|1, 0|1, 0|1]

        score = evalutate(random_sol)
        features_walsh = apply_walsh(random_sol, masks)

        X_train.append(features_walsh)
        Y_train.append(score)

    return np.array(X_train), np.array(Y_train)

if __name__ == "__main__":
    N = 20 # Dimension
    D = 2 # Ordre max, avec D = 2 on vérifie uniquement les duos
    NB = 10 # Budget / nombre de solutions qu'on peut générer
    
    masks = generate_masks(N, D) # 211 masques, 0 parmi 20 (1) + 1 parmi 20 (20) + 2 parmi 20 (190)
    print(f"Nombre de masques générés : {len(mes_masques)}")
    
    X, Y = generate_dataset(N, NB, masks, random_black_box)
    
    print(f"Forme de X (exemples, masques) : {X.shape}") 
    print(f"Premier score : {Y[0]}")