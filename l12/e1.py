import numpy as np
import gym
from dezero import Model
from dezero import optimizers
import dezero.functions as F
import dezero.layers as L

class PolicyNet(Model): # 정책 신경망
    def __init__(self, action_size=2):
        super().__init__()
        self.l1 = L.Linear(128)
        self.l2 = L.Linear(action_size)

    def forward(self,x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        x = F.softmax(x)    # 확률 출력
        return x
    
class ValueNet(Model): # 가치 함수 신경망
    def __init__(self):
        super().__init__()
        self.l1 = L.Linear(256)
        self.l2 = L.Linear(1)

    def forward(self,x):
        x = F.relu(self.l1(x)) 
        x = self.l2(x)

        return x
    
class Agent:
    def __init__(self):
        # ! original Hp
        # self.gamma = 0.98
        # self.lr_pi = 0.0002
        # self.lr_v = 0.0005
        
        # ! modified -1 
        self.gamma = 0.99
        self.lr_pi = 0.00008
        self.lr_v = 0.0001
        
        
        self.action_size = 3

        self.pi = PolicyNet(action_size=self.action_size)
        self.v = ValueNet()
        self.optimizer_pi = optimizers.Adam(self.lr_pi).setup(self.pi)
        self.optimizer_v = optimizers.Adam(self.lr_v).setup(self.v)

    def get_action(self, state):
        state = state[np.newaxis, :]     # 배치 처리용 축 추가
        probs = self.pi(state)
        probs = probs[0]
        action = np.random.choice(len(probs), p=probs.data)
        return action, probs[action] # 선택된 행동과 해당 행동의 확률 반환
    
    def update(self, state, action_prob, reward, next_state, done):
        # 배치 처리용 축 추가
        state = state[np.newaxis, :]
        next_state = next_state[np.newaxis, :]

        # 가치 함수의 손실 계산
        print(f"{state}, {abs(state[:,1] * 10)}")
        target = reward + self.gamma * (self.v(next_state) + abs(state[:,1] * 100)) * (1- done) # TD 목표
        target.unchain()
        v = self.v(state) # 현재 상태의 가치 함수
        loss_v = F.mean_squared_error(v, target) # 두 값의 평균제곱오차

        # 정책의 손실 계산
        delta = target -v
        delta.unchain()
        loss_pi = -F.log(action_prob) * delta
        
        # print(f"{state} {action_prob} {v} {loss_v.data}")

        self.v.cleargrads()
        self.pi.cleargrads()
        loss_v.backward()
        loss_pi.backward()
        self.optimizer_v.update()
        self.optimizer_pi.update()




episodes = 300
env = gym.make('MountainCar-v0', render_mode='rgb_array')
# env = gym.make('MountainCar-v0', render_mode='human')
agent = Agent()
reward_history = []

for episode in range(episodes):
    state = env.reset()[0]
    done = False
    total_reward = 0

    while not done:
        action, prob = agent.get_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        # print(f"{next_state}, {reward}, {info}")
        
        done= terminated | truncated

        agent.update(state, prob, reward, next_state, done)

        state = next_state
        total_reward += reward

    reward_history.append(total_reward)
    if episode % 100 == 0:
        # env.render()
        print("episode: {}, total reward: {:.1f}".format(episode, total_reward))
    # print("episode: {}, total reward: {:.1f}".format(episode, total_reward))

# 그래프
from common.utils import plot_total_reward
plot_total_reward(reward_history)





# 학습 끝난 에이전트에 탐욕 행동을 선택하도록 하여 플레이
env2 = gym.make("MountainCar-v0", render_mode='human')

state = env2.reset()[0]
done = False
total_reward = 0

while not done:
    action, prob = agent.get_action(state)
    next_state, reward, terminated, truncated, info = env2.step(action)
    done = terminated | truncated

    agent.update(state, prob, reward, next_state, done)

    state = next_state
    total_reward += reward
    env2.render()

print("total reward: {}".format(total_reward))