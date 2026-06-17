import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from agent import TrackerAgent
from environment import Environment
from fn import debug_print, getHourlySolarPos, getMPAHourly, getSolarPower
from glb import endHr, startHr


def main():
    panelCapacity = 220 # Watt
    
    solpos = getHourlySolarPos() # 0 ~ 23h : (azimuth, zenith)
    l_mpa = getMPAHourly(solpos[['azimuth','zenith']], startHr, endHr, panelCapacity)

    hours = np.arange(startHr, endHr+1)          # 6:00 ~ 18:00
    
    env = Environment()
    agent = TrackerAgent()

    # ! 모델 훈련 부분
    episodes = 1000
    for ep in range(episodes):
        # 초기 각도 설정
        state = env.reset() # ! 초기 State

        for hour in enumerate(hours):
            for i in range(60):
                action = agent.getAction(hour, state)
                
                next_state, reward, done = env.step(action, hour)                
                agent.update(hour, state, action, reward, next_state, done)
                
                if done: 
                    break
            
                state = next_state


    # ! Figure 6 재현 부분
    tracking_mpa = []
    
    current_state = 2
    init_angle = 0
    
    for hour in hours:
        ias = []
        
        for i in range(60):
            # * 현재 angle의 state 기준
            # theoretical_mpa[i]  getSolarPower(i, init_angle)
            qs = [agent.Q[(hour, current_state, a)] for a in range(5)]
            
            best_action_idx = np.argmax(qs)

            move = [x * env.angle_factor for x in [-2, -1, 0, 1, 2]][best_action_idx]

            next_angle = init_angle + move
            
            p_curr = env.getSolarPower(hour, init_angle)
            p_next = env.getSolarPower(hour, next_angle) if 0 <= next_angle and next_angle <= 30 else p_curr
            
            if 0 <= next_angle and next_angle <= 30:
                init_angle = next_angle

            current_state = env.next_state(p_next-p_curr)
            
            ias.append(init_angle)
        mn = np.mean(ias)
        tracking_mpa.append(mn)

    # ! 차트 출력
    plt.figure(figsize=(14, 4))
    plt.subplot(131)
    plt.plot(solpos['azimuth'], marker='s', markersize=3, color='r', label="azimuth angle")
    plt.plot(solpos['zenith'], marker='s', markersize=3, color='b', label="zenith angle")
    plt.legend()
    plt.ylabel('Tilt Angle (deg)')
    plt.xlabel('Hour of Day (h)')
    plt.title('Fig 4`. 2012-08-02 Solar position in Albuquerque')

    plt.subplot(132)
    for h, mpa_tilt, max_p, tilt_range, powers in l_mpa:
        plt.plot(tilt_range, powers, label=f'{int(h):02d}:00 (MPA:{mpa_tilt:.1f}°)')
        plt.scatter(mpa_tilt, max_p, color='black', s=40, edgecolor='black', zorder=5)
    plt.legend()
    plt.ylabel('Power (W)')
    plt.xlabel('Tilt Angle (deg)')
    plt.title("Fig 5`. MPA Curves for Each Hour")
    
    theoretical_mpa = [(h, mpa_tilt) for h, mpa_tilt, _, _, _ in l_mpa]

    xaxis = [mpa[0] for mpa in theoretical_mpa]

    plt.subplot(133)
    
    plt.plot(xaxis,[float(mpa[1]) for mpa in theoretical_mpa], 'r-', label='Theoretical MPA', marker='s', markersize=3, linewidth=2)
    plt.plot(xaxis,tracking_mpa, 'b--', label='mean(RL Tracking Angle) per hour', marker='s', markersize=4)
    
    plt.title('Fig 6`: Q-learned Tracking vs Theoretical MPA')
    plt.xlabel('Hour of Day (h)')
    plt.ylabel('Tilt Angle (deg)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    main()