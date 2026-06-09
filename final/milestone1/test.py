import numpy as np
import random
import matplotlib.pyplot as plt

from fn import getRewardFromMPA,getHourlySolarPos, getMPAHourly

# 1. 하이퍼파라미터 및 환경 설정
eta = 0.1         # 학습률 (η)
gamma = 0.9       # 할인율 (γ)
epsilon = 0.8     # 탐색 확률 (ε-greedy)
threshold = 0.05   # 정지 상태 판단 임계값

# 시간 설정 (6시 ~ 18시, 총 720분 동안 1분 단위로 시뮬레이션)
start_hour = 6
end_hour = 18
total_minutes = (end_hour - start_hour) * 60  # 720 steps
total_hrs = (end_hour - start_hour)  # 12 steps

# 상태 및 행동 공간 설정
num_states = 3  # s0: Toward, s1: Leaving, s2: Rest
actions = [-2, -1, 0, 1, 2]
num_actions = len(actions)

# Q-테이블 초기화
q_table = np.zeros((num_states, num_actions))

solpos = getHourlySolarPos()
l_mpa = getMPAHourly(solpos[['azimuth','zenith']], 6, 18, 220)
angleT = [0,0,0,10,14,17,18,19,20,15,12,0]


# 2. 시간별 실제 MPA(최대 발전 각도) 생성 함수 (가상의 태양 궤적)
# 6시(0도) -> 12시(90도 남중) -> 18시(180도)로 부드럽게 이동한다고 가정
def get_true_mpa(current_minute):
    # 0분 ~ 720분 사이의 비율 (0.0 ~ 1.0)
    progress = current_minute / total_minutes
    # 0도에서 180도까지 사인 곡선 형태 또는 선형태 변형 (여기서는 직관적인 선형 태양 이동 모델 사용)
    return progress * 180.0

# 3. 데이터 저장용 리스트 (시각화 목적)
history_time = []
history_mpa = []
history_panel_angle = []
history_power = []

# 초기 상태 설정
current_angle = 0.0  # 6시 정각, 패널은 0도를 바라봄
p_old = 0.0
current_state = 2    # 초기 상태 Rest(2)로 가정

# 에피소드
# for ep in range(10):
# 4. 시뮬레이션 루프 진행 (6시 ~ 18시)
for t in range(total_minutes):
    # 현재 시간 계산 (시각화용 축 라벨링용)
    actual_hour = start_hour + (t / 60)
    
    # print(actual_hour, int(actual_hour))
    true_mpa = angleT[int(actual_hour)-6]
    
    # true_mpa = get_true_mpa(t)
    
    # ε-greedy 행동 선택
    if random.random() < (1 - epsilon):
        action_idx = np.argmax(q_table[current_state])
    else:
        action_idx = random.randint(0, num_actions - 1)
    chosen_action = actions[action_idx]
    
    # 행동 반영 (패널 각도 변경 및 0~180도 범위 제한)
    current_angle = max(0.0, min(180.0, current_angle + chosen_action))
    
    # 현재 전력 측정 (MPA와 패널 각도가 일치할 때 최대 100이 나오도록 설정)
    p_now = 100.0 - abs(true_mpa - current_angle)**2
    p_now = max(0.0, p_now) # 전력이 음수가 되지 않도록 방지
    
    # 보상 계산 (ΔP)
    reward = p_now - p_old
    
    # 다음 상태 정의
    if reward > threshold:
        next_state = 0   # s0: Toward
    elif reward < -threshold:
        next_state = 1   # s1: Leaving
    else:
        next_state = 2   # s2: Rest
        
    # Q-테이블 업데이트 (첫 번째 이미지 공식 적용)
    max_future_q = np.max(q_table[next_state])
    q_table[current_state, action_idx] = (1 - eta) * q_table[current_state, action_idx] + \
                                        eta * (reward + gamma * max_future_q)
    
    # 기록 저장
    history_time.append(actual_hour)
    history_mpa.append(true_mpa)
    history_panel_angle.append(current_angle)
    history_power.append(p_now)
    
    # 상태 및 전력 갱신
    current_state = next_state
    p_old = p_now

# 5. pyplot 시각화 영역
plt.figure(figsize=(12, 6))

# 첫 번째 그래프: 태양의 MPA vs 패널의 추종 각도
plt.subplot(2, 1, 1)
plt.plot(history_time, history_mpa, 'r--', label='True MPA (Sun)', linewidth=2)
plt.plot(history_time, history_panel_angle, 'b-', label='Panel Angle (RL Agent)', linewidth=1.5)
plt.title('Solar Tracking Simulation using Light-weight Q-Learning (6:00 - 18:00)', fontsize=12)
plt.xlabel('Time (Hour)')
plt.ylabel('Angle (Degree)')
plt.xticks(range(start_hour, end_hour + 1))
plt.grid(True, linestyle=':')
plt.legend()

# 두 번째 그래프: 시간에 따른 출력 전력 변화
plt.subplot(2, 1, 2)
plt.plot(history_time, history_power, 'g-', label='Output Power', linewidth=1.5)
plt.xlabel('Time (Hour)')
plt.ylabel('Power (W)')
plt.xticks(range(start_hour, end_hour + 1))
plt.grid(True, linestyle=':')
plt.legend()

plt.tight_layout()
plt.show()

print("최종 학습된 소형 Q-테이블 (3행 5열):")
print(q_table)
