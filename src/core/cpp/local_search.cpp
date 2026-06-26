#include "local_search.hpp"

void WalshSurrogateLocalSearch::init_walsh_state(const py::array_t<int>& sol_, const py::array_t<double>& coefs_,
                                    std::vector<int>& current, std::vector<int>& best,
                                    std::vector<double>& phi, std::vector<double>& delta, double& cur_fit) {
    // On utilise unchecked() au lieu de std::vector pour lire la ram de Python directement sans faire de copie mémoire, plus rapide
    auto sol  = sol_.unchecked<1>();
    auto coefs = coefs_.unchecked<1>();

    int n_coefs = (int)_masks.size();

    for (int i = 0; i < _N; i++)
        current[i] = sol(i);

    best = current;

    // valeur de la fonction de Walsh pour le masque i sur current / phi[i] = (-1)^(somme des bits de current couverts par le masque i)
    for (int i = 0; i < n_coefs; i++) {
        int s = 0;
        for (int b : _masks[i])
            s += current[b];
        phi[i] = (s % 2 == 0) ? 1.0 : -1.0;
    }

    cur_fit = 0.0;
    for (int i = 0; i < n_coefs; i++)
        cur_fit += coefs(i) * phi[i];

    // delta[x] = gain prédit si on flippe le bit x
    for (int x = 0; x < _N; x++)
        for (int i : _bit_to_masks[x])
            delta[x] -= 2.0 * coefs(i) * phi[i];
}

void WalshSurrogateLocalSearch::apply_incremental_flip(int bit, const py::array_t<double>& coefs_,
                                          std::vector<int>& current, std::vector<double>& phi,
                                          std::vector<double>& delta, double& cur_fit) {
    auto coefs = coefs_.unchecked<1>();

    current[bit] = 1 - current[bit]; // flip
    cur_fit += delta[bit]; // évalution de la nouvelle fitness

    for (int i : _bit_to_masks[bit]) { // parcourt tous les masks affectés par le bit changé
        double old_phi = phi[i];
        phi[i] = -phi[i]; // tous les masques concernés changent de parité
        double upd = 4.0 * coefs(i) * old_phi;

        for (int y : _masks[i])
            delta[y] += upd;
    }
}
