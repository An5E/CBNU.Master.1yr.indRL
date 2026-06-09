import numpy as np
import random
import matplotlib.pyplot as plt

# 1. 하이퍼파라미터 및 환경 설정 (논문 알고리즘 1 기준)
eta = 0.2         # 학습률 (η)
gamma = 0.9       # 할인율 (γ)
threshold = 0.05   # 정지 상태(s2) 판단 임계값

# 시간 설정 (6시 ~ 18시, 총 720분 동안 1분 단위 시뮬레이션)
start_hour = 6
end_hour = 18
total_minutes = (end_hour - start_hour) * 60  # max_steps = 720

# 상태 및 행동 공간 설정
num_states = 3    # s0: Toward, s1: Leaving, s2: Rest
actions = [-2, -1, 0, 1, 2]
num_actions = len(actions)

# Q-테이블 초기화
q_table = np.zeros((num_states, num_actions))

# 시간별 실제 MPA(최대 발전 각도) 생성 함수 (가상의 태양 궤적)
def get_true_mpa(current_minute):
    progress = current_minute / total_minutes
    return progress * 180.0  # 6시(0도) -> 18시(180도) 선형 이동

# 데이터 저장용 리스트 (시각화 목적)
history_time = []
history_mpa = []
history_panel_angle = []
history_power = []

# 초기 상태 및 전력 세팅
current_angle = 0.0
p_old = 0.0
current_state = 2  # 초기 상태 주입

# --- 메인 루프 (Algorithm 1 구조 반영) ---
for i in range(total_minutes):
    actual_hour = start_hour + (i / 60)
    true_mpa = get_true_mpa(i)
    
    # 3: if rand() < 0.8
    if random.random() < 0.8:
        # [80% 확률] 무작위 탐색 구역 (Exploration)
        # 5: chosen_action = alist(randi(action_list_size))
        action_idx = random.randint(0, num_actions - 1)
        chosen_action = actions[action_idx]
        
        # 6~7: d_a = a + chosen_action 및 상태 업데이트 준비
        current_angle = max(0.0, min(180.0, current_angle + chosen_action))
        
        # 8: reward = p_now - p
        p_now = max(0.0, 100.0 - abs(true_mpa - current_angle)**2)
        reward = p_now - p_old
        
        # 상태 업데이트 로직 (논문 요약 반영)
        if reward > threshold: next_state = 0
        elif reward < -threshold: next_state = 1
        else: next_state = 2
        
        # 10: q_value 업데이트
        max_future_q = np.max(q_table[next_state])
        q_table[current_state, action_idx] = (1 - eta) * q_table[current_state, action_idx] + \
                                             eta * (reward + gamma * max_future_q)
    else:
        # [20% 확률] 최적 활용 구역 (Exploitation)
        # 12: chosen_action = alist(max(q_value))
        action_idx = np.argmax(q_table[current_state])
        chosen_action = actions[action_idx]
        
        # 13~14: d_a = a + chosen_action 및 상태 업데이트 준비
        current_angle = max(0.0, min(180.0, current_angle + chosen_action))
        
        # 15: reward = p_now - p
        p_now = max(0.0, 100.0 - abs(true_mpa - current_angle)**2)
        reward = p_now - p_old
        
        # 상태 업데이트 로직
        if reward > threshold: next_state = 0
        elif reward < -threshold: next_state = 1
        else: next_state = 2
        
        # 17: q_value 업데이트
        max_future_q = np.max(q_table[next_state])
        q_table[current_state, action_idx] = (1 - eta) * q_table[current_state, action_idx] + \
                                             eta * (reward + gamma * max_future_q)
    
    # 데이터 기록
    history_time.append(actual_hour)
    history_mpa.append(true_mpa)
    history_panel_angle.append(current_angle)
    history_power.append(p_now)
    
    # 변수 갱신 (s <- s', p <- p')
    current_state = next_state
    p_old = p_now

# --- 5. pyplot 시각화 영역 ---
plt.figure(figsize=(10, 6))

# 첫 번째 그래프: 태양의 MPA vs 패널의 추종 각도
plt.subplot(2, 1, 1)
plt.plot(history_time, history_mpa, 'r--', label='True MPA (Sun)', linewidth=2)
plt.plot(history_time, history_panel_angle, 'b-', label='Panel Angle (Algorithm 1)', linewidth=1.2)
plt.title('Algorithm 1 Solar Tracking Simulation (80% Exploration)', fontsize=12)
plt.ylabel('Angle (Degree)')
plt.xticks(range(start_hour, end_hour + 1))
plt.grid(True, linestyle=':')
plt.legend()

# 두 번째 그래프: 시간에 따른 출력 전력 변화
plt.subplot(2, 1, 2)
plt.plot(history_time, history_power, 'g-', label='Output Power', linewidth=1.2)
plt.xlabel('Time (Hour)')
plt.ylabel('Power (W)')
plt.xticks(range(start_hour, end_hour + 1))
plt.grid(True, linestyle=':')
plt.legend()

plt.tight_layout()
plt.show()

print("최종 학습된 논문 기준 Q-테이블:")
print(q_table)
