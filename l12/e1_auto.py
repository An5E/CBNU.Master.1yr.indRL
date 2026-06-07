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

qnetSize_l1 = 128
qnetSize_l2 = 16
episodes = 500      # 에피소드 수
early_stop_limit = int(episodes/3)
sync_interval = 20  # 신경망 동기화 주기 (#번째 에피소드마다 동기화)

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
        self.l1 = L.Linear(qnetSize_l1)
        self.l2 = L.Linear(qnetSize_l2)
        self.l3 = L.Linear(action_size)

    def forward(self,x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x
    
class DQNAgent:  # 에이전트 클래스
    def __init__(self, gamma, lr, epsilon):
        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon
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


def unitRun(gamma, lr, epsilon):
    env = gym.make('MountainCar-v0', render_mode='rgb_array')
    # env = gym.make('MountainCar-v0', render_mode='human')
    # env.metadata['render_fps'] = 600

    agent = DQNAgent(gamma, lr, epsilon)
    reward_history = [] # 에피소드별 보상 기록

    hit_count = 0

    print(f"# qnet:{qnetSize_l1}*{qnetSize_l2}*3, gamma:{agent.gamma}, lr:{agent.lr}, eps:{agent.epsilon}")

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

        if terminated:
            hit_count += 1
        if episode % sync_interval == 0:
            agent.sync_qnet()

        reward_history.append(total_reward)
        if episode % 10 == 0:
            print("qnet:{}*{}, episode: {}, total reward: {}, hit: {}".format(qnetSize_l1, qnetSize_l2, episode, total_reward, hit_count))
        if episode > 0 and (episode % early_stop_limit == 0):
            # early stopping
            if hit_count <= 4:
                print("# Early stopped ---")
                return 

    if hit_count > 35:
        with open("e1_auto.result.txt", "a+") as f:
            f.write(f"qnet:{qnetSize_l1}*{qnetSize_l2}*3, episodes:{episodes}, gamma:{agent.gamma}, lr:{agent.lr}, eps:{agent.epsilon}, max. hit: {hit_count}, min. hit occur episode: {reward_history.index(min(reward_history))}\n")

# # 카트 폴에서 에피소드별 보상 총합 추이
# plt.xlabel('Episode')
# plt.ylabel('Total reward')
# plt.plot(range(len(reward_history)), reward_history)
# plt.title(f"qnet:{qnetSize_l1}*{qnetSize_l2}, gamma:{agent.gamma}, lr:{agent.lr}, eps:{agent.epsilon}")
# plt.show()


for i in range(999):
    gamma = round(random.uniform(0.98, 0.999), 3) 
    lr = round(random.uniform(0.002, 0.003), 3)
    epsilon = round(random.uniform(0.018, 0.09), 3)

    unitRun(gamma, lr, epsilon)
    
    





# 학습 끝난 에이전트에 탐욕 행동 선택하도록 하여 플레이
# env2 = gym.make('MountainCar-v0', render_mode='human')
# env2.metadata['render_fps'] = 120

# agent.epsilon = 0 # 탐욕 정책 (무작위 행동할 확률 입실론을 0으로 설정)
# state = env2.reset()[0]
# done = False
# total_reward = 0

# while not done:
#     action = agent.get_action(state)
#     next_state, reward, terminated, truncated, info = env2.step(action)
#     done = terminated | truncated
#     state = next_state
#     total_reward += reward
#     env2.render()

# print('Total Reward:', total_reward)