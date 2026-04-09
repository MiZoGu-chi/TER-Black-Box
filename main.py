import numpy as np
from data_sampler import generate_dataset
from walsh import generate_masks, apply_walsh
from sklearn.linear_model import LassoLars
from nk import NK_Landscape  

NB_ENTRAINEMENT = 1000 
NB_TEST = 1000

landscape = NK_Landscape("instances/5/2/nk_5_2_0.txt")

N = landscape.N 
K = landscape.K
# D = K + 1
D = 2

masks = generate_masks(N, D)

print("generation...")

X_train, Y_train = generate_dataset(N, NB_ENTRAINEMENT, masks, landscape.eval)


### Consituer le dataset de test de taille NB_TEST
random_sol = np.random.randint(2, size=N)
score = landscape.eval(random_sol)
features = apply_walsh(random_sol, masks)

##
X_test, Y_test = generate_dataset(N, NB_ENTRAINEMENT, masks, landscape.eval)


print(X_train.shape)

print(Y_train.shape)

print("apprentisage...")

for t in range(1000):

    surrogate = LassoLars(alpha=0.001, fit_intercept=False)  # wO constante -> intercept
    surrogate.fit(X_train[:(t+1)], Y_train[:(t+1)])

    print(f"Moyenne : {surrogate.coef_[0]}")

    # test de la fonction surrogate


    score = landscape.eval(random_sol)
    features = apply_walsh(random_sol, masks)

    prediction = surrogate.predict(X_test)

    print("\nResultat du test\n")
    print(f"Score: {score:.4f}")
    print(f"Score Surrogate: {prediction:.4f}")