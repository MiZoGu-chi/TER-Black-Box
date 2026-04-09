import numpy as np
from data_sampler import generate_dataset
from walsh import generate_masks, apply_walsh
from sklearn.linear_model import LassoLars
from nk import NK_Landscape  

NB_ENTRAINEMENT = 1000 

landscape = NK_Landscape("instances/20/2/nk_20_2_0.txt")

N = landscape.N 
K = landscape.K
D = K + 1   

masks = generate_masks(N, D)

print("generation...")

X_train, Y_train = generate_dataset(N, NB_ENTRAINEMENT, masks, landscape.eval)

print("apprentisage...")

surrogate = LassoLars(alpha=0.0001, fit_intercept=False) 
surrogate.fit(X_train, Y_train)

# test de la fonction surrogate

random_sol = np.random.randint(2, size=N)
score = landscape.eval(random_sol) 
features = apply_walsh(random_sol, masks)
prediction = surrogate.predict([features])[0] # car on test sur une unique solution

print("\nResultat du test\n")
print(f"Score: {score:.4f}")
print(f"Score Surrogate: {prediction:.4f}")