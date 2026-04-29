import ast
import os
import glob #
import numpy as np

def analyser_fichier(filepath):
    with open(filepath, 'r') as f:
        lignes = f.readlines()
        if not lignes:
            return None
        
        derniere_ligne = lignes[-1].strip()
        if derniere_ligne.startswith('['):
            coefs = np.array(ast.literal_eval(derniere_ligne))
            nb_zeros = np.sum(coefs == 0.0)
            nb_total = len(coefs)
            pourcentage = (nb_zeros / nb_total) * 100
            
            return pourcentage, nb_zeros, nb_total
    return None