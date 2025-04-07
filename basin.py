from distribution import *


p = find_best_multinormal_center_hopping(10)
with open(of, 'a') as file:
    file.write(str(p))

