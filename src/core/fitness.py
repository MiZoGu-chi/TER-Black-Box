import numpy as np

from src.core.data_sampler import transform

def get_surrogate_fitness_fn(surrogate_model, masks):

    def fitness_fn(solutions):
        X_transformed = transform(masks, solutions)

        fitness = surrogate_model.predict(X_transformed)

        return fitness

    return fitness_fn

def get_black_box_fitness_fn(landscape):

    def fitness_fn(solutions):
        fitness = [landscape.eval(sol) for sol in solutions]

        return np.array(fitness)

    return fitness_fn
