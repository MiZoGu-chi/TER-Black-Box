import numpy as np

from walsh import apply_walsh_vectorized
from walsh import generate_masks

# NB : nb de solutions ; N : taille d'une solution
def generate_random_solutions(N, NB):
    return np.random.randint(2, size=(NB, N)) # N = 5, [0|1, 0|1, 0|1, 0|1, 0|1]

def evaluate_and_transform(masks, objective_fun, solutions):
    X_train = []
    Y_train = [] 

    for s in solutions:
        score = objective_fun(s)
        features_walsh = apply_walsh_vectorized(s, masks)

        X_train.append(features_walsh)
        Y_train.append(score)
    
    return np.array(X_train), np.array(Y_train)

# def transform(masks, solutions):
#     X_train = []

#     for s in solutions:
#         features_walsh = apply_walsh_vectorized(s, masks)

#         X_train.append(features_walsh)
    
#     return np.array(X_train)

def transform(masks, solutions):
    sol_array = np.array(solutions) 
    mask_array = np.array(masks)    
    
    intersections = np.dot(sol_array, mask_array.T) 
    
    return np.where(intersections % 2 == 0, 1, -1)

def generate_dataset(N, NB, masks, objective_fun):
    random_sols = generate_random_solutions(N, NB)
    X_train, Y_train = evaluate_and_transform(masks, objective_fun, random_sols)
    return X_train, Y_train

    