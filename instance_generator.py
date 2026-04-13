import os
import random
import argparse

def generate_nk_instance(N, K, num_instance):
    path = os.path.join("instances", str(N), str(K))
    os.makedirs(path, exist_ok=True)
    
    filename = f"nk_{N}_{K}_{num_instance}.txt"
    filepath = os.path.join(path, filename)

    if os.path.exists(filepath):
        print("L'instance exite déjà ...")
        return
    
    with open(filepath, 'w') as f:
        f.write(f"{N} {K}\n")
        
        for i in range(N):
            for j in range(K + 1):
                neighbor = (i + j) % N
                f.write(f"{neighbor}\n")
                
        num_fitness_values = N * (2 ** (K + 1))
        for _ in range(num_fitness_values):
            f.write(f"{random.uniform(0.0, 1.0):.6f}\n")
            
    print(f"{filepath} généré !")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("N", type=int)
    parser.add_argument("K", type=int)
    parser.add_argument("NUM", type=int)
    
    args = parser.parse_args()
    
    generate_nk_instance(args.N, args.K, args.NUM)