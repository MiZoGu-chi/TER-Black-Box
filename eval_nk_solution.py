from nk import NK_Landscape
import numpy as np


pb = NK_Landscape("instances/64/2/nk_64_2_0.txt")

solution = np.random.randint(0,2,64)

print(solution)
print(pb.eval(solution))
