import numpy as np
import matplotlib.pyplot as plt

from environment import Environment
from agent import TrackerAgent

from fn import getHourlySolarPos, getMPAHourly, getSolarPower, debug_print
from glb import startHr, endHr 
import pandas as pd

import random


def main():
    panelCapacity = 220 # Watt
    
    solpos = getHourlySolarPos() # 0 ~ 23h : (azimuth, zenith)
    l_mpa = getMPAHourly(solpos[['azimuth','zenith']], startHr, endHr, panelCapacity)

    hours = np.arange(startHr, endHr+1)          # 6:00 ~ 18:00
    
    env = Environment()
    agent = TrackerAgent()

    episodes = 1000 # 2000
    for ep in range(episodes):
        # 초기 각도 설정
        state = env.reset() # ! 초기 State
        # hour = int(random.uniform(6, 18))

        # ! 시간 값은 상태에 포함되지 않음
        for h_idx, hour in enumerate(hours):
            # while True:
            for i in range(60):
                action = agent.getAction(hour, state)
                
                # * l_mpa에서 MPA 참조할 것
                next_state, reward, done = env.step(action, hour)
                
                # print(f"state: {state}, action: {action} => next_state: {next_state}, reward: {reward}")
                agent.update(hour, state, action, reward, next_state, done)
                
                # print(f"episode: {ep}, hour: {hour}, state: {state}, action: {action}, next_state: {next_state}, reward: {reward}, agent.Q[{state},{action}]: {agent.Q[state, action]}")
                
                
                if done: 
                    debug_print(f" DONE episode: {ep}, hour: {hour}, state: {state}, action: {action}, next_state: {next_state}, reward: {reward}, done: {done} ... agent.Q[{state},{action}]: {agent.Q[state, action]}\n")        
                    break
            
                state = next_state

    # ! Figure 6 재현
    tracking_mpa = []
    possible_angles = np.linspace(0, 30, 31)
    
    pd.DataFrame(agent.Q.items()).to_csv("./result.csv")
    
    
    current_state = 2
    # print([x * 2 for x in [-2, -1, 0, 1, 2]])
    init_angle = 0
    
    for hour in hours:
        ias = []
        qs = [agent.Q[(hour, current_state, a)] for a in range(5)]
        
        for i in range(60):
            # * 현재 angle의 state 기준
            # theoretical_mpa[i]  getSolarPower(i, init_angle)
            
            
            best_action_idx = np.argmax(qs)
            debug_print(f"{hour} , {best_action_idx}: {max(qs)}")
            # tracking_mpa.append(st)

            move = [x * env.angle_factor for x in [-2, -1, 0, 1, 2]][best_action_idx]

            # print(f"{hour}h:{best_action_idx}({move}),{agent.Q[(hour, current_state,best_action_idx)]},ia:{init_angle},mv:{move}")

            next_angle = init_angle + move
            if 0 <= next_angle and next_angle <= 30:
                init_angle = next_angle

            ias.append(init_angle)

            p_curr = env.getSolarPower(hour, init_angle)
            p_next = env.getSolarPower(hour, next_angle) if next_angle < 30 else p_curr
            current_state = env.next_state(p_next-p_curr)
            
        mn = np.mean(ias)
        print(f"hour:{hour}, mean: {mn}")
        tracking_mpa.append(mn)

    # ! 차트 출력
    plt.figure(figsize=(14, 4))
    plt.subplot(131)
    plt.plot(solpos['azimuth'], marker='s', markersize=3, color='r', label="azimuth angle")
    plt.plot(solpos['zenith'], marker='s', markersize=3, color='b', label="zenith angle")
    plt.legend()
    plt.ylabel('Angle (deg)')
    plt.xlabel('Hour of Day (h)')
    plt.title('Fig 4`. 2012-08-02 Solar position in Albuquerque')

    plt.subplot(132)
    for h, mpa_tilt, max_p, tilt_range, powers in l_mpa:
        plt.plot(tilt_range, powers, label=f'{int(h):02d}:00 (MPA:{mpa_tilt:.1f}°)')
        plt.scatter(mpa_tilt, max_p, color='black', s=40, edgecolor='black', zorder=5)
    plt.legend()
    plt.ylabel('Power (W)')
    plt.xlabel('Hour of Day (h)')
    plt.title("Fig 5`. MPA Curves for Each Hour")
    
    theoretical_mpa = [(h, mpa_tilt) for h, mpa_tilt, _, _, _ in l_mpa]

    xaxis = [mpa[0] for mpa in theoretical_mpa]

    plt.subplot(133)
    
    print(xaxis)
    print(tracking_mpa)
    
    plt.plot(xaxis,[float(mpa[1]) for mpa in theoretical_mpa], 'r-', label='Theoretical MPA', marker='s', markersize=3, linewidth=2)
    plt.plot(xaxis,tracking_mpa, 'b--', label='RL Tracking Angle', marker='s', markersize=4)
    
    plt.title('Fig 6`: Q-learned Tracking vs Theoretical MPA')
    plt.xlabel('Hour of Day (h)')
    plt.ylabel('Tilt Angle (deg)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()

    # MAE 출력
    # mae = np.mean(np.abs(np.array(tracking_mpa) - np.array([float(mpa[1]) for mpa in theoretical_mpa])))
    # print(f"재현 결과 Mean Absolute Error (MAE): {mae:.4f} degrees")

if __name__ == "__main__":
    main()