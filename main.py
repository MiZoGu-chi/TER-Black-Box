import numpy as np

from sklearn.linear_model import Lars
from sklearn.linear_model import LassoLars
from sklearn.linear_model import Ridge

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

import matplotlib.pyplot as plt

from data_sampler import generate_dataset
from walsh import generate_masks, apply_walsh
from nk import NK_Landscape  

from plot import plot_learning_curve
from plot import plot_learning_curve_mae
from plot import plot_compare_mae_lars_cg

NB_ENTRAINEMENT = 400
NB_TEST = 1000

instance_name = "instances/20/1/nk_20_1_0.txt"
landscape = NK_Landscape(instance_name)

N = landscape.N 
K = landscape.K
# D = K + 1
D = 2

print("generation des masques...")
masks = generate_masks(N, D)

print("generation du dataset d'entraînement...")
X_train, Y_train = generate_dataset(N, NB_ENTRAINEMENT, masks, landscape.eval)

print("génération du datast de test...")
X_test, Y_test = generate_dataset(N, NB_TEST, masks, landscape.eval)

print(X_train.shape)

print(Y_train.shape)

mses_lars = []
maes_lars = []

mses_cg = []
maes_cg = []

for t in range(NB_ENTRAINEMENT):

    surrogate_lars = LassoLars(alpha=0.001, fit_intercept=False)  
    surrogate_lars.fit(X_train[:(t+1)], Y_train[:(t+1)])

    #surrogate_lars = Lars(fit_intercept=False)  
    #surrogate_lars.fit(X_train[:(t+1)], Y_train[:(t+1)])

    surrogate_cg = Ridge(alpha=0.001, fit_intercept=False, solver='sparse_cg')
    surrogate_cg.fit(X_train[:(t+1)], Y_train[:(t+1)])

    #print(f"Moyenne : {surrogate.coef_[0]}")
    w0 = surrogate_lars.coef_[0]

    # test des fonctions surrogate

    predictions_lars = surrogate_lars.predict(X_test)
    predictions_cg = surrogate_cg.predict(X_test)

    mse_lars = mean_squared_error(Y_test, predictions_lars)
    mae_lars = mean_absolute_error(Y_test, predictions_lars)

    mse_cg = mean_squared_error(Y_test, predictions_cg)
    mae_cg = mean_absolute_error(Y_test, predictions_cg)

    mses_lars.append(mse_lars)
    maes_lars.append(mae_lars)

    mses_cg.append(mse_cg)
    maes_cg.append(mae_cg)

    print(f"Lars - Itération : {t} ; Moyenne : {w0} ; MSE : {mse_lars} ; MAE : {mae_lars}")
    print(f"CG - Itération : {t} ; MSE : {mse_lars} ; MAE : {mae_lars}")

plot_compare_mae_lars_cg(maes_lars, maes_cg, D, instance_name.split('/')[-1].replace('.txt', ''))
plt.show()  