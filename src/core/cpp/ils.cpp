#include "ils.hpp"

#include <random>
#include <algorithm>
#include <numeric>

std::vector<int> ILS::perturb(const std::vector<int>& solution, double alpha, unsigned int seed) {
    std::mt19937 rng(seed);
    std::vector<int> result = solution;

    // !! pour un N petit et/ou un alpha faible, (int)(alpha * _N) peut tronquer à 0 !!
    int n_flips = (int)(alpha * _N);

    std::vector<int> indices(_N);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), rng);

    // On inverse les n_flips premiers bits ordonnés alétoirement
    for (int i = 0; i < n_flips; i++)
        result[indices[i]] = 1 - result[indices[i]];

    return result;
}

std::vector<std::vector<int>> ILS::find_connected_components(const std::vector<bool>& is_differing, const py::array_t<double>& coefs_) {
    auto coefs = coefs_.unchecked<1>();
    std::vector<std::vector<int>> components;
    std::vector<bool> visited(_N, false);

    // DFS
    for (int start = 0; start < _N; start++) {
        if (!is_differing[start] || visited[start]) continue;

        std::vector<int> component;
        std::vector<int> stack = {start};
        visited[start] = true;

        while (!stack.empty()) {
            int current = stack.back();
            stack.pop_back();
            component.push_back(current);

            // voisins de current = toutes les variables qui partagent un masque actif avec current
            for (int mask_idx : _bit_to_masks[current]) {
                if (coefs(mask_idx) == 0.0) continue; // masque inactif

                for (int neighbor : _masks[mask_idx]) {
                    if (neighbor == current) continue;
                    if (!is_differing[neighbor]) continue; // bit commun -> ignoré
                    if (visited[neighbor]) continue;

                    visited[neighbor] = true;
                    stack.push_back(neighbor);
                }
            }
        }

        components.push_back(component);
    }

    // components = { C_1, C_2, ..., C_q }, partition de { j | is_differing[j] }
    // en composantes connexes via les masques actifs
    return components;
}

bool ILS::compare_component(const std::vector<int>& component, const std::vector<int>& p1, const std::vector<int>& p2, py::array_t<double> coefs_) {
    auto coefs = coefs_.unchecked<1>();

    std::vector<bool> mask_seen(_masks.size(), false);
    std::vector<int> relevant_masks;

    for (int var : component) {
        for (int mask_idx : _bit_to_masks[var]) {
            if (!mask_seen[mask_idx]) {
                mask_seen[mask_idx] = true; // si plusieurs var du composant sont liées au même masque on ne l'ajoute pas deux fois
                relevant_masks.push_back(mask_idx);
            }
        }
    }

    double g_p1 = 0.0, g_p2 = 0.0;

    for (int mask_idx : relevant_masks) {
        double c = coefs(mask_idx);
        if (c == 0.0) continue;

        int sum_p1 = 0, sum_p2 = 0;
        for (int b : _masks[mask_idx]) {
            sum_p1 += p1[b];
            sum_p2 += p2[b];
        }

        double phi_p1 = (sum_p1 % 2 == 0) ? 1.0 : -1.0;
        double phi_p2 = (sum_p2 % 2 == 0) ? 1.0 : -1.0;

        g_p1 += c * phi_p1;
        g_p2 += c * phi_p2;
    }

    return g_p1 > g_p2;
}

std::vector<int> ILS::partition_crossover(const std::vector<int>& p1, const std::vector<int>& p2, py::array_t<double> coefs_) {
    std::vector<bool> is_differing(_N);

    for (int i = 0; i < _N; i++)
        is_differing[i] = (p1[i] != p2[i]); // on prend uniquement les bits qui différent entre les deux optima locaux

    auto components = find_connected_components(is_differing, coefs_);

    std::vector<int> child = p1; // les bits communs entre p1 et p2 sont hérités

    for (const auto& component : components) {
        bool p1_greater = compare_component(component, p1, p2, coefs_);
        if (!p1_greater) {
            for (int var : component)
                child[var] = p2[var];
        }
    }

    return child;
}
