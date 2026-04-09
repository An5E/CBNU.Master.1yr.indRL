import numpy as np
import matplotlib.pyplot as plt
from bandit import Bandit, Agent

class NonStatBandit:
    def __init__(self, arms=10):
        self.arms = arms
        self.rate = np.random.rand(arms)

    def play(self, arm):
        rate = self.rate[arm]
        self.rate += 0.1 * np.random.randn(self.arms)
        if rate > np.random.rand():
            return 1
        else:
            return 0
        
class AlphaAgent:
    def __init__(self, epsilon, alpha, actions=10):
        self.epsilon = epsilon
        self.Qs = np.zeros(actions)
        self.alpha = alpha

    def update(self, action, reward):
        self.Qs[action] += (reward - self.Qs[action]) * self.alpha

    def get_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, len(self.Qs))
        return np.argmax(self.Qs)
    
runs = 200
steps = 1000

# @Keonwoo : 범례에 추가할 epsilon 3종 배열 정의
epsilons = [0.1,0.3,0.01]

results = {}

# @Keonwoo : epsilons 배열 요소 순회하면서 epsilon 적용된 곡선 차트 값 추가 
for epsilon in epsilons:
    all_rates = np.zeros((runs, steps))

    for run in range(runs):
        agent = Agent(epsilon)


        bandit = NonStatBandit()
        total_reward = 0
        rates = []

        for step in range(steps):
            action = agent.get_action()
            reward = bandit.play(action)
            agent.update(action, reward)
            total_reward += reward
            rates.append(total_reward / (step + 1))

        all_rates[run] = rates

    avg_rates = np.average(all_rates, axis=0)
    results[epsilon] = avg_rates

plt.figure()
plt.ylabel('Rates')
plt.xlabel('Steps')
for key, avg_rates in results.items():
    plt.plot(avg_rates, label=key)
plt.legend()
plt.show()