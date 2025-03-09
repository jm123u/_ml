import random
import math

citys = [
    (0,3),(0,0),
    (0,2),(0,1),
    (1,0),(1,3),
    (2,0),(2,3),
    (3,0),(3,3),
    (3,1),(3,2)
]

def distance(p1, p2):
    return math.dist(p1, p2)

def path_length(path):
    return sum(distance(citys[path[i]], citys[path[(i+1) % len(path)]]) for i in range(len(path)))

def two_opt_neighbor(path):
    a, b = sorted(random.sample(range(len(path)), 2))
    return path[:a] + path[a:b+1][::-1] + path[b+1:]

def hill_climbing(path, max_fail=10000):
    best_path, best_cost = path, path_length(path)
    fail_count = 0
    iterations = 0
    
    while fail_count < max_fail:
        new_path = two_opt_neighbor(best_path)
        new_cost = path_length(new_path)
        iterations += 1
        
        if new_cost < best_cost:
            best_path, best_cost = new_path, new_cost
            fail_count = 0 
        else:
            fail_count += 1
    
    return best_path

random.seed(42)
initial_path = random.sample(range(len(citys)), len(citys))
best_path = hill_climbing(initial_path)
print(f"optimal path : {best_path}\nshortest distance: {path_length(best_path):.4f}")