import random
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pvlib import irradiance, solarposition

from common.utils import greedy_probs

class Environment:
    def __init__(self):
        self.action_space = [0,1,2,3,4]
        self.action_meaning = {
            0: "Decrease tilt by -- degree",
            1: "Decrease tilt by - degree",
            2: "No change",
            3: "Increase tilt by + degree",
            4: "Increase tilt by ++ degree"
        }
        
        self.goal_state = 0 # ! MPA 가 발생하는 경사각 
        self.start_state = 0 # ! 초기 경사각
        self.agent_state = self.start_state
        
    
    def reset(self):
        self.agent_state = self.start_state
        return self.agnet_state
    
    def next_state(self, state, action):
        
        action_move_map = [-2, -1, 0, 1, 2]
        move = action_move_map[action]
        next_state = state + move
        
        if next_state < 0 or next_state > 80:
            next_state = state        
        
        return next_state
        
    def getSolarPower(hour, tilt_angle):
        # 논문 Figure 5의 특성: 정오(12-13시)에 최대 전력, 특정 각도에서 피크 발생
        # 6시~18시까지 최적 각도가 10~18도 사이에서 변하는 포물선 모델
        center_angle = l_mpa[hour-6][1] # 이론적 최적 각도(MPA)
        max_p = l_mpa[hour-6][2] # 시간대별 최대 전력 (W)
        
        # 각도가 최적에서 멀어질수록 전력 감소 (가우시안 분포 근사)
        # print(l_mpa[hour-6][3:5])
        
        # ? {hour} 곡선에서 x={tilt_angle}인 y값 구하기. l_mpa에서 참조
        l_mpa[hour][tilt_angle]
        
        possible_angles = []
        
        # print(f"hour:{hour}, tilt_angle:{tilt_angle}, = {np.argmin(np.abs(possible_angles - tilt_angle))}")
        
        power = max_p  * l_mpa[hour-6][4][np.argmin(np.abs(possible_angles - tilt_angle))] # np.exp(-((tilt_angle - center_angle)**2) / )
        return max(0, power)

    def reward(self, state, action, next_state):
        # ? getSolarPower? 
        # ! 발전량 최대치가 나오는 경사각으로 이동
        # ! MPA 곡선은 비교 데이터일 뿐, 추종하는 것이 아님
        return self.getSolarPower(hour, next_state) - self.getSolarPower(hour, state)
    
    def step(self, action):
        state = self.agent_state
        next_state = self.next_state(state, action)
        reward = self.reward(state, action, next_state)
        done = (reward < 0) # ! 보상(발전량 변화율)이 낮아지면 종료
        
        self.agent_state = next_state
        return next_state, reward, done

class TrackerAgent:
    def __init__(self):
        self.gamma = 0.9
        self.alpha = 0.1
        self.epsilon = 0.8 # ! e-greedy 계수 ( 0.8 )
        
        self.action_size = 5
        random_actions = {0:.20, 1:.20, 2:.20, 3:.20, 4:.20}
        self.b = defaultdict(lambda: random_actions)
        self.Q = defaultdict(lambda: 0)
        self.tilt_degree = 0
        
    def getAction(self, state):
        action_probs = self.b[state]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions, p=probs)
        
    def update(self, state, action, reward, next_state, done):
        if done:
            next_q_max = 0
        else:
            next_qs = [self.Q[next_state, a] for a in range(self.action_size)]
            next_q_max = max(next_qs)
            
        target = reward + self.gamma * next_q_max
        self.Q[state, action] += (target - self.Q[state, action]) * self.alpha

        self.b[state] = greedy_probs(self.Q, state, self.epsilon)


# ! 태양 방위, 고도각 출력 함수 ( Fig4. )
def getHourlySolarPos():
    lat, lon = 35.08, -106.65 # ! 앨버커키 위도, 경도
    tz = 'MST' # Mountain Standard Time
    times = pd.date_range('2012-08-02 00:00:00', '2012-08-02 23:00:00', freq='1h', tz=tz)

    # 태양 위치 계산
    solpos = solarposition.get_solarposition(times, lat, lon)
    
    solpos['hr'] = solpos.index.strftime("%H")
    solpos.reset_index(drop=True, inplace=True)
    solpos.index = solpos['hr'].astype(int)
    
    surface_tilt = 30
    surface_azimuth = 180 # ! 남향
    aoi = irradiance.aoi(surface_tilt, surface_azimuth, solpos['apparent_zenith'], solpos['azimuth'])

    power_curve = np.cos(np.radians(aoi))
    power_curve[solpos['elevation'] < 0] = 0
    
    return solpos[['azimuth', 'zenith']]

def getMPAHourly(src: pd.DataFrame, max_power):
    ret = []
    src.reset_index(drop=True, inplace=True)
    
    data = {
        'hour': np.arange(6, 19), # 6시 ~ 18시 낮 시간대
        'azimuth': src.azimuth[6:19].values,
        'zenith': src.zenith[6:19].values
    }
    
    df = pd.DataFrame(data)

    # 2. 설정 값
    tilt_range = np.linspace(0, 80, 161)  # X축: 패널 기울기 (0~90도)
    panel_azimuth = 180  # 패널 설치 방향 (정남향)

    # 3. 시각화 설정
    # plt.figure(figsize=(12, 8))
    # colors = cm.plasma(np.linspace(0, 1, len(df))) # 시간 흐름에 따른 색상

    # 4. 시간별(Row별) 루프 수행
    for i, row in df.iterrows():
        h = row['hour']
        s_azi = row['azimuth']
        s_zen = row['zenith']
        s_alt = 90 - s_zen  # 천정각을 고도각으로 변환
        
        powers = []
        for tilt in tilt_range:
            # 라디안 변환
            s_alt_r, s_azi_r = np.radians(s_alt), np.radians(s_azi)
            t_r, p_azi_r = np.radians(tilt), np.radians(panel_azimuth)
            
            # 입사각(AOI) 코사인 계산
            # cos(AOI) = sin(alt)*cos(tilt) + cos(alt)*sin(tilt)*cos(s_azi - p_azi)
            cos_theta = (np.sin(s_alt_r) * np.cos(t_r) + 
                        np.cos(s_alt_r) * np.sin(t_r) * np.cos(s_azi_r - p_azi_r))
            
            power = max_power * np.clip(cos_theta, 0, 1)
            powers.append(power)
        
        # 최대 발전 각도(MPA) 추출
        max_idx = np.argmax(powers)
        mpa_tilt = tilt_range[max_idx]
        max_p = powers[max_idx]
        
        ret.append((h, mpa_tilt, max_p, tilt_range, powers))
        
        # 그래프 그리기 (선형 차트)
        # plt.plot(tilt_range, powers, label=f'{int(h):02d}h (MPA:{mpa_tilt:.1f}°)', 
        #         color=colors[i], linewidth=1.8, alpha=0.8)
        
        # MPA 지점에 포인트 표시
        # plt.scatter(mpa_tilt, max_p, color=colors[i], s=40, edgecolor='black', zorder=5)

    return ret



# 1. Figure 6: 시간 및 각도에 따른 출력 전력 모델링 (근사 함수)

# 2. Q-러닝 파라미터 설정 (논문 330-331p 참조)

# possible_angles = np.linspace(0, 80, 161) # 0.5도 단위로 0~80도 상태 구성
# actions = [-1.0, -0.5, 0, 0.5, 1.0]      # 각도 변경 단계 (Action list)

# q_table = np.zeros((len(hours), len(possible_angles)))

# 3. 학습 과정: Figure 5의 전력을 보상으로 사용

def test():
    solpos = getHourlySolarPos()
    l_mpa = getMPAHourly(solpos[['azimuth','zenith']], 3500)
    
    print(l_mpa)

def main():
    solpos = getHourlySolarPos()
    l_mpa = getMPAHourly(solpos[['azimuth','zenith']], 3500)
    hours = np.arange(6, 19)          # 6:00 ~ 18:00
    
    env = Environment()
    agent = TrackerAgent()

    episodes = 2 # 2000
    for ep in range(episodes):
        for h_idx, hour in enumerate(hours):
            # 초기 각도 설정
            curr_angle_idx = 0 # ! 초기 각도 0도 고정 (MPA 곡선에 따라) 

            while True:
                action = agent.getAction(state)
                next_state, reward, done = env.step(action)
                
                agent.update(state, action, reward, next_state, done)
                if done:
                    break
                
                state = next_state
                

            # for step in range(10): # 각 시간대별 최적 각도 탐색 시도
                # 행동 선택 (e-greedy)
                # if random.random() < epsilon:
                #     a_idx = random.randint(0, len(actions)-1)
                # else:
                #     a_idx = np.argmax(q_table[h_idx, curr_angle_idx])
                
                # 다음 상태(각도) 계산
                # prev_angle = possible_angles[curr_angle_idx]
                # next_angle = np.clip(prev_angle + actions[a_idx], 0, 30)
                # next_angle_idx = np.argmin(np.abs(possible_angles - next_angle))
                
                # 보상 계산: ΔP = P_now - P_prev (논문 식 4)
                # p_prev = getSolarPower(hour, prev_angle)
                # p_now = getSolarPower(hour, next_angle)
                # reward = p_now - p_prev # 전력이 높아지는 방향으로 유도
                
                # Q-값 업데이트 (식 1)
                # q_table[h_idx, curr_angle_idx] += alpha * (
                #     reward + gamma * np.max(q_table[h_idx, next_angle_idx]) - q_table[h_idx, curr_angle_idx]
                # )
                # curr_angle_idx = next_angle_idx

    # 4. 결과 도출 (Figure 6 재현)
    tracking_mpa = []

    # for i in range(len(hours)):
        # 각 시간대별 Q값이 가장 높은 각도를 최적 각도로 판단
        # best_angle_idx = np.argmax(q_table[i])
        # tracking_mpa.append(possible_angles[best_angle_idx])

    # 5. 시각화
    plt.figure(figsize=(16, 5))
    plt.subplot(1, 3, 1)
    plt.plot(solpos['azimuth'], marker='s', markersize=3, color='r', label="azimuth angle")
    plt.plot(solpos['zenith'], marker='s', markersize=3, color='b', label="zenith angle")
    plt.legend()
    plt.ylabel('Angle (deg)')
    plt.xlabel('Hour of Day (h)')
    plt.title('Fig 4`. 2012-08-02 Solar position in Albuquerque')

    plt.subplot(1, 3, 2)
    for h, mpa_tilt, max_p, tilt_range, powers in l_mpa:
        plt.plot(tilt_range, powers, label=f'{int(h):02d}:00 (MPA:{mpa_tilt:.1f}°)')
        plt.scatter(mpa_tilt, max_p, color='black', s=40, edgecolor='black', zorder=5)
    plt.legend()
    plt.ylabel('Power (W)')
    plt.xlabel('Hour of Day (h)')
    plt.title("Fig 5`. MPA Curves for Each Hour")

    # [왼쪽] Figure 5 컨셉 확인: 시간별 전력 곡선
    # for h in range(6,12):
    #     powers = [get_solar_power(h, a) for a in possible_angles]
    #     plt.plot(possible_angles, powers, label=f'{h}:00')
    # plt.title('Figure 5: Power vs Tilt Angle (Environment)')
    # plt.xlabel('Tilt Angle (deg)')
    # plt.ylabel('Power (W)')
    # plt.legend()
    # plt.grid(True, alpha=0.3)

    # [오른쪽] Figure 6 재현: 최적 각도 추적 결과

    theoretical_mpa = [(h, mpa_tilt) for h, mpa_tilt, _, _, _ in l_mpa]

    # print([mpa[0] for mpa in theoretical_mpa])
    # print([float(mpa[1]) for mpa in theoretical_mpa])
    # print(tracking_mpa)

    plt.subplot(1, 3, 3)
    plt.plot([float(mpa[1]) for mpa in theoretical_mpa], 'r-', label='Theoretical MPA', marker='s', markersize=3, linewidth=2)
    plt.plot(tracking_mpa, 'b--', label='RL Tracking Angle', marker='s', markersize=4)
    plt.title('Figure 6`: Q-learned Tracking vs Theoretical MPA')
    plt.xlabel('Hour of Day (h)')
    plt.ylabel('Tilt Angle (deg)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()

    # MAE 출력
    mae = np.mean(np.abs(np.array(tracking_mpa) - np.array([float(mpa[1]) for mpa in theoretical_mpa])))
    print(f"재현 결과 Mean Absolute Error (MAE): {mae:.4f} degrees")

if __name__ == "__main__":
    # main()
    test()