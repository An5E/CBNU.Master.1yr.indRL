import numpy as np
import matplotlib.pyplot as plt
import random

# 1. Figure 5: 시간 및 각도에 따른 출력 전력 모델링 (근사 함수)
def get_solar_power(hour, tilt_angle):
    # 논문 Figure 5의 특성: 정오(12-13시)에 최대 전력, 특정 각도에서 피크 발생
    # 6시~18시까지 최적 각도가 10~18도 사이에서 변하는 포물선 모델
    center_angle = 10 + 8 * np.sin(np.pi * (hour-6) / 12) # 이론적 최적 각도(MPA)
    max_p = 3800 * np.sin(np.pi * (hour-6) / 12) # 시간대별 최대 전력 (W)
    
    # 각도가 최적에서 멀어질수록 전력 감소 (가우시안 분포 근사)
    power = max_p * np.exp(-((tilt_angle - center_angle)**2) / 1060)
    return max(0, power)

# 2. Q-러닝 파라미터 설정 (논문 330-331p 참조)
hours = np.arange(0, 24)          # 6:00 ~ 18:00
possible_angles = np.linspace(10, 80, 141) # 0.5도 단위로 0~30도 상태 구성
actions = [-1.0, -0.5, 0, 0.5, 1.0]      # 각도 변경 단계 (Action list)

q_table = np.zeros((len(hours), len(possible_angles)))
alpha = 0.1
gamma = 0.9
epsilon = 0.8 # ! e-greedy 계수 ( 0.8 )

# 3. 학습 과정: Figure 5의 전력을 보상으로 사용
episodes = 2 # 2000
for ep in range(episodes):
    for h_idx, hour in enumerate(hours):
        # 초기 각도 설정
        curr_angle_idx = 20 # ! 초기 각도 10도로 고정 
        # random.randint(0, len(possible_angles)-1)
        print(possible_angles[curr_angle_idx])

        for step in range(10): # 각 시간대별 최적 각도 탐색 시도
            # 행동 선택 (e-greedy)
            if random.random() < epsilon:
                a_idx = random.randint(0, len(actions)-1)
            else:
                a_idx = np.argmax(q_table[h_idx, curr_angle_idx])
            
            # 다음 상태(각도) 계산
            prev_angle = possible_angles[curr_angle_idx]
            next_angle = np.clip(prev_angle + actions[a_idx], 0, 30)
            next_angle_idx = np.argmin(np.abs(possible_angles - next_angle))
            
            # 보상 계산: ΔP = P_now - P_prev (논문 식 4)
            p_prev = get_solar_power(hour, prev_angle)
            p_now = get_solar_power(hour, next_angle)
            reward = p_now - p_prev # 전력이 높아지는 방향으로 유도
            
            # Q-값 업데이트 (식 1)
            q_table[h_idx, curr_angle_idx] += alpha * (
                reward + gamma * np.max(q_table[h_idx, next_angle_idx]) - q_table[h_idx, curr_angle_idx]
            )
            curr_angle_idx = next_angle_idx

# 4. 결과 도출 (Figure 6 재현)
theoretical_mpa = [10 + 8 * np.sin(np.pi * (h - 6) / 12) for h in hours]
tracking_mpa = []

for i in range(len(hours)):
    # 각 시간대별 Q값이 가장 높은 각도를 최적 각도로 판단
    best_angle_idx = np.argmax(q_table[i])
    tracking_mpa.append(possible_angles[best_angle_idx])

# 5. 시각화
plt.figure(figsize=(12, 5))

# [왼쪽] Figure 5 컨셉 확인: 시간별 전력 곡선
plt.subplot(1, 2, 1)
for h in range(6,12):
    powers = [get_solar_power(h, a) for a in possible_angles]
    plt.plot(possible_angles, powers, label=f'{h}:00')
plt.title('Figure 5: Power vs Tilt Angle (Environment)')
plt.xlabel('Tilt Angle (deg)')
plt.ylabel('Power (W)')
plt.legend()
plt.grid(True, alpha=0.3)

# [오른쪽] Figure 6 재현: 최적 각도 추적 결과
plt.subplot(1, 2, 2)
plt.plot(hours, theoretical_mpa, 'r-', label='Theoretical MPA', marker='s', markersize=3, linewidth=2)
plt.plot(hours, tracking_mpa, 'b--', label='RL Tracking Angle', marker='s', markersize=4)
plt.title('Figure 6: Error between Tracking and Theoretical')
plt.xlabel('Hour of Day (h)')
plt.ylabel('Tilt Angle (deg)')
plt.xticks(hours)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()

# MAE 출력
mae = np.mean(np.abs(np.array(tracking_mpa) - np.array(theoretical_mpa)))
print(f"재현 결과 Mean Absolute Error (MAE): {mae:.4f} degrees")
