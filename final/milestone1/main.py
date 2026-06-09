import numpy as np
import matplotlib.pyplot as plt

from environment import Environment
from agent import TrackerAgent

from fn import getHourlySolarPos, getMPAHourly, debug_print
from glb import startHr, endHr 
import pandas as pd


def main():
    panelCapacity = 220 # Watt
    
    solpos = getHourlySolarPos() # 0 ~ 23h : (azimuth, zenith)
    l_mpa = getMPAHourly(solpos[['azimuth','zenith']], startHr, endHr, panelCapacity)
        
    hours = np.arange(startHr, endHr)          # 6:00 ~ 18:00
    
    env = Environment()
    agent = TrackerAgent()

    episodes = 5 # 2000
    for ep in range(episodes):
        # 초기 각도 설정
        state = env.reset() # ! 초기 State

        # ! 시간 값은 상태에 포함되지 않음
        for h_idx, hour in enumerate(hours):
            # while True:
            for i in range(60):
                action = agent.getAction(state)
                
                next_state, reward, done = env.step(action, hour)
                # print(f"state: {state}, action: {action} => next_state: {next_state}, reward: {reward}")
                agent.update(state, action, reward, next_state, done)
                
                
                if done: 
                #     if hour == 6:
                    print(f"episode: {ep}, hour: {hour}, state: {state}, action: {action}, next_state: {next_state}, reward: {reward}, done: {done} ... agent.Q[{state},{action}]: {agent.Q[state, action]}")
                #         debug_print(f"episode: {ep}, hour: {hour}, state: {state}, action: {action}, next_state: {next_state}, reward: {reward}, done: {done} ... agent.Q[{state},{action}]: {agent.Q[state, action]}")
                
                    break
                
                state = next_state

    # ! Figure 6 재현
    tracking_mpa = []
    possible_angles = np.linspace(0, 30, 31)

    st = 0

    print(agent.Q.keys())
    print(np.array(list(agent.Q.values())))

    init_angle = 0
    
    # print([x * 2 for x in [-2, -1, 0, 1, 2]])
    
    for i in range(len(hours)+1):
        best_action_idx = np.argmax([agent.Q[i, 0],agent.Q[i, 1], agent.Q[i, 2], agent.Q[i, 3], agent.Q[i, 4]])
        debug_print(f"{i} , {best_action_idx}: {max(agent.Q[i, 0],agent.Q[i, 1], agent.Q[i, 2], agent.Q[i, 3], agent.Q[i, 4])}")

        st += possible_angles[best_action_idx]
        # tracking_mpa.append(st)

        print(f"{i+6}h:{best_action_idx},{agent.Q[i,best_action_idx]}")

        next_state = init_angle + [x * 2 for x in [-2, -1, 0, 1, 2]][best_action_idx]
        if next_state > 0 and next_state < 30:
            init_angle = next_state

        tracking_mpa.append(init_angle)

    debug_print("## tracking_mpa")
    debug_print(tracking_mpa)

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