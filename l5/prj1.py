import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# 1. 환경 설정
# ---------------------------

TIME_STEPS = 24          # 하루를 24 step으로 단순화
# MAX_ANGLE = 90            # 패널 경사각 범위 (-90 ~ 90)
ANGLE_STEP = 24            # 이산화 단위

ACTIONS = [-2, -1, 0, 1, 2]  # 각도 변화 (step 단위)
N_ACTIONS = len(ACTIONS)

# 상태 이산화
ANGLE_BINS = np.arange(10, 20, ANGLE_STEP)
N_STATES = len(ANGLE_BINS)

def discretize_angle(angle):
    return int(np.digitize(angle, ANGLE_BINS) - 1)

# ---------------------------
# 2. 태양 위치 (간단 모델)
# ---------------------------

def sun_angle(t):
    return 130 - (110 * np.cos(t * (90 / TIME_STEPS)))

# ---------------------------
# 3. 발전량 모델
# ---------------------------

def power(panel_angle, sun_angle):
    theta = np.radians(panel_angle - sun_angle)
    return max(0, np.cos(theta))

# ---------------------------
# 4. Q-learning 설정
# ---------------------------

Q = np.full((N_STATES, N_ACTIONS), 1/N_ACTIONS) # ! 정책 확률 초기화 (.2)

alpha = 0.1
gamma = 0.9
epsilon = 0.8
episodes = 1 # 500

# ---------------------------
# 5. 학습
# ---------------------------

for ep in range(episodes): # ! 2:
    panel_angle = 10  # 초기 각도

    for t in range(TIME_STEPS - 1):
        s = discretize_angle(panel_angle)

        # epsilon-greedy
        if np.random.rand() < epsilon: # ! 3
            
            a_idx = np.random.randint(N_ACTIONS) # ! 5
        else:
            a_idx = np.argmax(Q[s]) # ! 12

        action = ACTIONS[a_idx]

        # 다음 상태
        next_angle = panel_angle + action * ANGLE_STEP
        next_angle = np.clip(next_angle, -90, 90)

        # 보상 (논문 핵심)
        p_now = power(panel_angle, sun_angle(t))
        p_next = power(next_angle, sun_angle(t + 1))
        reward = p_next - p_now
        
        # print(f"Episode {ep}, Time {t}, State {s}, Action {action}, Reward {reward:.4f} ( Power change: {p_next} - {p_now} )\n")

        s_next = discretize_angle(next_angle)

        # Q 업데이트
        Q[s, a_idx] += alpha * (
            reward + gamma * np.max(Q[s_next]) - Q[s, a_idx]
        )

        panel_angle = next_angle

    # epsilon 감소
    epsilon *= 0.995

# ---------------------------
# 6. 테스트 (학습된 정책)
# ---------------------------

panel_angles = []
sun_angles = []
powers = []

panel_angle = 0

for t in range(TIME_STEPS):
    s = discretize_angle(panel_angle)
    a_idx = np.argmax(Q[s])
    action = ACTIONS[a_idx]

    panel_angle += action * ANGLE_STEP
    panel_angle = np.clip(panel_angle, -90, 90)

    sa = sun_angle(t)
    p = power(panel_angle, sa)

    panel_angles.append(panel_angle)
    sun_angles.append(sa)
    powers.append(p)

# ---------------------------
# 7. 비교 (고정 패널)
# ---------------------------

fixed_powers = []
for t in range(TIME_STEPS):
    fixed_powers.append(power(0, sun_angle(t)))

# ---------------------------
# 8. 시각화
# ---------------------------

plt.figure(figsize=(12,5))

# 각도 비교
plt.subplot(1,2,1)
plt.plot(sun_angles, label="Sun angle")
plt.plot(panel_angles, label="Panel angle (RL)")
plt.title("Angle Tracking")
plt.legend()

# 발전량 비교
plt.subplot(1,2,2)
plt.plot(powers, label="RL Tracker")
plt.plot(fixed_powers, label="Fixed Panel")
plt.title("Power Output")
plt.legend()

plt.show()

print("RL total power:", sum(powers))
print("Fixed total power:", sum(fixed_powers))