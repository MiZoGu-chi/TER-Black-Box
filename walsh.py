import numpy as np 

from itertools import combinations

# x : vecteurs de bits, la solution à évaluer (numpy.ndarray) -> [0, 1, 1, 0, 0]
# k : le masque binaires (numpy.ndarray)
def walsh(x, k):
    bits_filter = k * x 

    sum_bits = np.sum(bits_filter)

    return (-1) ** sum_bits

def apply_walsh(x, masks):
    res = []

    for k in masks:
        temp = walsh(x, k)
        res.append(temp)

    return np.array(res)

# n : nb de paramêtres de la solution ; d_max ordre maximun (2 poour s'arrêter aux paires)
def generate_masks(n, d_max): 
    masks = []

    m0 = np.zeros(n, dtype=int)
    masks.append(m0) # ajout du masqe 0
    print(f"Masque 0 : {m0}")

    # boucle sur les ordres
    for d in range(1, d_max + 1):
        for pos in combinations(range(n), d):
            mask = np.zeros(n, dtype=int)
            mask[list(pos)] = 1
            masks.append(mask)

            print(f"Masque {len(masks) - 1} : {mask}")

    return masks