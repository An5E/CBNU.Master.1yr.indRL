import copy
from collections import deque
import random
import matplotlib.pyplot as plt
import numpy as np
import gym
from dezero import Model
from dezero import optimizers
import dezero.functions as F
import dezero.layers as L


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)
        self.buffer.append(data)

    def __len__(self):
        return len(self.buffer)
    
    def get_batch(self):
        data = random.sample(self.buffer, self.batch_size)

        state = np.stack([x[0] for x in data])
        action = np.array([x[1] for x in data])
        reward = np.array([x[2] for x in data])
        next_state = np.stack([x[3] for x in data])
        done = np.array([x[4] for x in data]).astype(np.int32)
        return state, action, reward, next_state, done
    
class QNet(Model): # 신경망 클래스
    def __init__(self, action_size):
        super().__init__()
        self.l1 = L.Linear(128)
        self.l2 = L.Linear(128)
        self.l3 = L.Linear(action_size)

    def forward(self,x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x
    
class DQNAgent:  # 에이전트 클래스
    def __init__(self):
        self.gamma = 0.99
        self.lr = 0.0008
        self.epsilon = 0.29
        self.buffer_size = 10000       # 경험 재생 버퍼 크기
        self.batch_size = 32           # 미니 배치 크기
        self.action_size = 3

        self.replay_buffer = ReplayBuffer(self.buffer_size, self.batch_size)
        self.qnet = QNet(self.action_size)              # 원본 신경망
        self.qnet_target = QNet(self.action_size)       # 목표 신경망
        self.optimizer = optimizers.Adam(self.lr)
        self.optimizer.setup(self.qnet)                 # 옵티마이저에 qnet 등록 

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        
        else:
            state = state[np.newaxis, :]     # 배치 처리용 차원 추가
            qs = self.qnet(state)
            return qs.data.argmax()
        
    # ! ----
    def update(self, state, action, reward, next_state, done):
        # 경험 재생 버퍼에 경험 데이터 추가
        self.replay_buffer.add(state, action, reward, next_state, done)
        if len(self.replay_buffer) < self.batch_size:
            return # 데이터가 미니배치 크기만큼 쌓이지 않은 경우
        
        # 미니배치 크기만큼 데이터 쌓이면 미니배치 생성
        state, action, reward, next_state, done = self.replay_buffer.get_batch()
        qs = self.qnet(state)
        q = qs[np.arange(self.batch_size), action]

        next_qs = self.qnet_target(next_state)
        next_q = next_qs.max()
        next_q.unchain()
        target = reward + (1-done) * self.gamma * next_q

        loss = F.mean_squared_error(q, target)

        self.qnet.cleargrads()
        loss.backward()
        self.optimizer.update()
    # ! ----

    def sync_qnet(self): # 두 신경망 동기화
        self.qnet_target = copy.deepcopy(self.qnet) 




episodes = 200      # 에피소드 수
sync_interval = 20  # 신경망 동기화 주기 (20번째 에피소드마다 동기화)
# env = gym.make('MountainCar-v0', render_mode='rgb_array')
env = gym.make('MountainCar-v0', render_mode='human')
env.metadata['render_fps'] = 600

agent = DQNAgent()
reward_history = [] # 에피소드별 보상 기록

for episode in range(episodes):
    state = env.reset()[0]
    done = False
    total_reward = 0

    while not done:
        action = agent.get_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated | truncated

        agent.update(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

    if episode % sync_interval == 0:
        agent.sync_qnet()

    reward_history.append(total_reward)
    if episode % 10 == 0:
        print("episode: {}, total reward: {}".format(episode, total_reward))

# 카트 폴에서 에피소드별 보상 총합 추이
plt.xlabel('Episode')
plt.ylabel('Total reward')
plt.plot(range(len(reward_history)), reward_history)
plt.title(f"gamma:{agent.gamma}, lr:{agent.lr}, eps:{agent.epsilon}")
plt.show()




# 학습 끝난 에이전트에 탐욕 행동 선택하도록 하여 플레이
env2 = gym.make('MountainCar-v0', render_mode='human')
# env2.metadata['render_fps'] = 30

agent.epsilon = 0 # 탐욕 정책 (무작위 행동할 확률 입실론을 0으로 설정)
state = env2.reset()[0]
done = False
total_reward = 0

while not done:
    action = agent.get_action(state)
    next_state, reward, terminated, truncated, info = env2.step(action)
    done = terminated | truncated
    state = next_state
    total_reward += reward
    env2.render()

print('Total Reward:', total_reward)