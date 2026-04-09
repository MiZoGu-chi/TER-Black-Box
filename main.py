import numpy as np
from echantilloneur import generate_dataset
from walsh import generate_masks, apply_walsh
from sklearn.linear_model import LassoLars
from nk import NK_Landscape  


N = 64    
D = 3     
NB_ENTRAINEMENT = 1000 

machine = NK_Landscape("instances/5/2/nk_5_2_0.txt")

print("Génération des masques...")
mes_masques = generate_masks(N, D)

print("Collecte des données réelles...")

X_train, Y_train = generate_dataset(N, NB_ENTRAINEMENT, mes_masques, machine.eval)

# 3. APPRENTISSAGE
print("L'IA LARS cherche les poids w_k...")
ia_lars = LassoLars(alpha=0.0001, fit_intercept=False) # Alpha plus petit pour plus de précision
ia_lars.fit(X_train, Y_train)

# 4. VÉRIFICATION
poids_non_nuls = np.sum(ia_lars.coef_ != 0)
print(f"LARS a détecté {poids_non_nuls} interactions sur {len(mes_masques)} possibles.")

# Test sur une solution inconnue
test_sol = np.random.randint(2, size=N)
vrai_score = machine.eval(test_sol) # On utilise la vraie machine
phi_test = apply_walsh(test_sol, mes_masques)
prediction = ia_lars.predict([phi_test])[0]

print(f"\nVrai score (Machine) : {vrai_score:.4f}")
print(f"Prédiction (IA)      : {prediction:.4f}")