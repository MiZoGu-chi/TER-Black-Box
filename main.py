import numpy as np

from sklearn.linear_model import Lars
from sklearn.linear_model import LassoLars
from sklearn.linear_model import LassoLarsIC

from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

import matplotlib.pyplot as plt

from data_sampler import generate_dataset
from walsh import generate_masks, apply_walsh
from nk import NK_Landscape  

from plot import plot_learning_curve
from plot import plot_learning_curve_mae
from plot import plot_compare_mae_lars_cg
from plot import plot_compare_models_mae

NB_ENTRAINEMENT = 400
NB_TEST = 1000

instance_name = "instances/20/2/nk_20_2_1.txt"
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

mses_larsIC = []
maes_larsIC = []

mses_cg = []
maes_cg = []

mses_elastic = []
maes_elastic = []

for t in range(NB_ENTRAINEMENT):

    surrogate_lars = LassoLars(alpha=0.001, fit_intercept=False)  
    surrogate_lars.fit(X_train[:(t+1)], Y_train[:(t+1)])

    # criterion{‘aic’, ‘bic’}, default=’aic’
    surrogate_larsIC = LassoLarsIC(criterion='bic', fit_intercept=False, noise_variance=1e-5)  
    surrogate_larsIC.fit(X_train[:(t+1)], Y_train[:(t+1)])

    #surrogate_lars = Lars(fit_intercept=False)  
    #surrogate_lars.fit(X_train[:(t+1)], Y_train[:(t+1)])

    surrogate_cg = Ridge(alpha=0.001, fit_intercept=False, solver='sparse_cg')
    surrogate_cg.fit(X_train[:(t+1)], Y_train[:(t+1)])

    surrogate_elastic = ElasticNet(alpha=0.001, fit_intercept=False)
    surrogate_elastic.fit(X_train[:(t+1)], Y_train[:(t+1)])

    #print(f"Moyenne : {surrogate.coef_[0]}")
    w0 = surrogate_lars.coef_[0]

    # test des fonctions surrogate

    predictions_lars = surrogate_lars.predict(X_test)
    predictions_cg = surrogate_cg.predict(X_test)
    predictions_elastic = surrogate_elastic.predict(X_test)
    predictions_larsIC = surrogate_larsIC.predict(X_test)

    mse_lars = mean_squared_error(Y_test, predictions_lars)
    mae_lars = mean_absolute_error(Y_test, predictions_lars)

    mse_cg = mean_squared_error(Y_test, predictions_cg)
    mae_cg = mean_absolute_error(Y_test, predictions_cg)

    mse_elastic = mean_squared_error(Y_test, predictions_elastic)
    mae_elastic = mean_absolute_error(Y_test, predictions_elastic)

    mse_larsIC = mean_squared_error(Y_test, predictions_larsIC)
    mae_larsIC = mean_absolute_error(Y_test, predictions_larsIC)

    mses_lars.append(mse_lars)
    maes_lars.append(mae_lars)

    mses_cg.append(mse_cg)
    maes_cg.append(mae_cg)

    mses_elastic.append(mse_elastic)
    maes_elastic.append(mae_elastic)

    mses_larsIC.append(mse_larsIC)
    maes_larsIC.append(mae_larsIC)

    print(f"Lars - Itération : {t} ; Moyenne : {w0} ; MSE : {mse_lars} ; MAE : {mae_lars}")
    print(f"CG - Itération : {t} ; MSE : {mse_cg} ; MAE : {mae_cg}")
    print(f"ElasticNet - Itération : {t} ; MSE : {mse_elastic} ; MAE : {mae_elastic}")

maes = {
    'LARS (Lasso)': maes_lars,
    'CG (Ridge)': maes_cg,
    'ElasticNet': maes_elastic,
    'LARS (LassoIC)': maes_larsIC,
}

mses = {
    'LARS (Lasso)': mses_lars,
    'CG (Ridge)': mses_cg,
    'ElasticNet': mses_elastic,
    'LARS (LassoIC)': mses_larsIC,
}

#plot_compare_mae_lars_cg(maes_lars, maes_cg, D, instance_name.split('/')[-1].replace('.txt', ''))
plot_compare_models_mae(maes, D, instance_name.split('/')[-1].replace('.txt', ''))
plt.show()  