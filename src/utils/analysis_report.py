from src.utils.io_utils import OptimizationResult, load_landscape_and_masks
from src.core.evaluation import get_surrogate_interaction, compare_interaction


def analyse_optimization_file(file):
    opti_res = OptimizationResult(file)

    N = opti_res.meta['N']
    K = opti_res.meta['K']
    D = opti_res.meta['D']
    num = opti_res.meta['num']

    landscape, masks = load_landscape_and_masks(N, K, D, num)

    # Format : { ordre1 : [((tuple1), coef1), ..., ((tupleN), coefN)], ordreN: ... }
    # Interactions + poids
    surrogate_graph = get_surrogate_interaction(opti_res.coefs, masks)

    # Format : { ordre1 : [(tuple1), ..., (tupleN)], ordreN: ... }
    # Uniquement les intéractions
    real_graph = landscape.get_real_interaction()

    confusion_matrix_by_K = compare_interaction(surrogate_graph, real_graph, D)

    return opti_res.meta, confusion_matrix_by_K, opti_res.fidelity_history, opti_res.fitness_history, opti_res.final_scatter, opti_res.sparsity
