from collections import defaultdict
import numpy as np

def argmax(xs):
    idxes = [i for i, x in enumerate(xs) if x == max(xs)]
    if len(idxes) == 1:
        return idxes[0]
    elif len(idxes) == 0:
        return np.random.choice(len(xs))

    selected = np.random.choice(idxes)
    return selected


def greedy_probs(Q, hour, state, epsilon=0, action_size=4):
    qs = [Q[(hour, state, action)] for action in range(action_size)]
    max_action = argmax(qs)  # OR np.argmax(qs)
    base_prob = epsilon / action_size
    action_probs = {action: base_prob for action in range(action_size)}  #{0: ε/4, 1: ε/4, 2: ε/4, 3: ε/4}
    action_probs[max_action] += (1 - epsilon)
    return action_probs

class TrackerAgent:
    def __init__(self):
        self.gamma = 0.9
        self.alpha = 0.2
        self.epsilon = 0.1 # ! e-greedy 계수 ( 0.8 )
        
        self.action_size = 5
        random_actions = {0:.20, 1:.20, 2:.20, 3:.20, 4:.20}
        self.b = defaultdict(lambda: random_actions)
        self.Q = defaultdict(lambda: 0)
        self.tilt_degree = 0
        
    # ! %(3,5), %(11,12): chosen_action
    def getAction(self, hour, state):
        # action_probs = self.b[state]
        # actions = list(action_probs.keys())
        # probs = list(action_probs.values()) 
        
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            qs = [self.Q[(hour, state, a)] for a in range(self.action_size)]
            return np.argmax(qs)
        
    def update(self, hour, state, action, reward, next_state, done):
        if done:
            next_q_max = 0
        else:
            next_qs = [self.Q[(hour,next_state, a)] for a in range(self.action_size)]
            next_q_max = max(next_qs)
            
        target = reward + self.gamma * next_q_max
        self.Q[(hour, state, action)] += self.alpha * (target - self.Q[(hour, state, action)])
        # self.b[state] = greedy_probs(self.Q, hour, state, self.epsilon)