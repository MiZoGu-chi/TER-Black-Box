from sklearn.linear_model import Lars, LassoLars, LassoLarsIC, Ridge, ElasticNet

MODELS = {
    'lars': {
        'model': Lars(n_nonzero_coefs=50, fit_intercept=False),
        'name': 'Lars',
        'color': 'tab:cyan'
    },
    'lasso_lars': {
        'model': LassoLars(alpha=0.001, fit_intercept=False, eps=1e-4),
        'name' : 'Lasso with Lars',
        'color': 'tab:green'
    },
    'lasso_lars_ic': {
        'model': LassoLarsIC(criterion='bic', fit_intercept=False, noise_variance=1e-5),
        'name': 'Lasso with Lars using BIC',
        'color': 'tab:orange'
    },
    'cg': {
        'model': Ridge(alpha=0.001, fit_intercept=False, solver='sparse_cg'),
        'name': 'CG (Ridge)',
        'color': 'tab:red'
    },
    'elastic_net': {
        'model': ElasticNet(alpha=0.001, fit_intercept=False),
        'name': 'Elastic Net',
        'color': 'tab:olive'
    }
}

# en faire une @dataclasses ?