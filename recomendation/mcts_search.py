import time 
import random

def mcts_search(initial_node, time_limit=1.0, return_score=False):
    start = time.time()
    Q = {}
    N = {}
    children = {}
    
    def uct(n):
        if N.get(n, 0) == 0:
            return float('inf')
        return Q[n] / N[n] + 1.41 * (sum(N.values()) / N[n]) ** 0.5
    
    def select(n):
        while n in children and children[n]:
            n = max(children[n], key=uct)
        return n
    
    def expand(n):
        if n not in children:
            children[n] = n.find_children()
        return children[n]
    
    def simulate(n):
        current = n
        while not current.is_terminal():
            current = current.find_random_child()
            if current is None:
                break
        return current.reward() if current else 0
    
    def backpropagate(path, reward):
        for node in path:
            N[node] = N.get(node, 0) + 1
            Q[node] = Q.get(node, 0) + reward
    
    while time.time() - start < time_limit:
        node = initial_node
        path = [node]
        
        while node in children and children[node]:
            node = max(children[node], key=uct)
            path.append(node)
        
        children_nodes = expand(node)
        if children_nodes:
            child = random.choice(list(children_nodes))
            path.append(child)
            reward = simulate(child)
        else:
            reward = node.reward()
        
        backpropagate(path, reward)
    
    if not children.get(initial_node) or len(children[initial_node]) == 0:
        return [] if not return_score else ([], -float("inf"))
    
    best = max(children[initial_node], key=lambda n: Q.get(n, 0) / N.get(n, 1))
    best_score = Q.get(best, 0) / N.get(best, 1)
    
    if return_score:
        return best.action, best_score
    else:
        return best.action