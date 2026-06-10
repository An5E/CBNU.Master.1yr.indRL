from fn import getRewardFromMPA, debug_print
from glb import startHr, endHr

class Environment:
    def __init__(self):
        self.action_space = [0,1,2,3,4]
        self.action_meaning = {
            0: "Decrease tilt by -2x degree",
            1: "Decrease tilt by -1x degree",
            2: "No change",
            3: "Increase tilt by +1x degree",
            4: "Increase tilt by +2x degree"
        }
        
        self.angle_factor = 2
        self.action_move_map = [x * self.angle_factor for x in [-2, -1, 0, 1, 2]] 
                
        # self.goal_state = (12, 0) # ! MPA 경사각 , (time, tilt_angle)
        # self.start_state = (0, self.init_tilt) # ! 초기 경사각
        
        self.start_state = 2 # ! 초기 state
        self.start_angle = 0
        self.agent_state = self.start_state
        self.agent_angle = self.start_angle
    
    def reset(self):
        self.agent_state = self.start_state
        return self.agent_state
    
    # ! %6~7, %13~14: update state
    def next_state(self, reward):
        
        # * (reward == 0) != MPA 각도 도달 (Rest)
        # * (시간대 MPA - reward) 결과를 next_state로 전달해야 함
        if reward > 0:
            next_state = 0
        elif reward < 0:
            next_state = 1
        else:
            next_state = 2

        return next_state
        
    def getSolarPower(self, hour, tilt_angle):        
        # ? {hour} 곡선에서 x={tilt_angle}인 y값 구하기. l_mpa에서 참조
        print(f"""  hr:{hour}, ta:{tilt_angle}""")
        return max(0, getRewardFromMPA(hour, tilt_angle, startHour=startHr, endHour=endHr))

    # ! %7, %15: reward = p_now-p
    def reward(self, current_angle, hour, action):
        # ! 발전량 최대치가 나오는 경사각으로 이동
        # ! MPA 곡선은 비교 예시 데이터일 뿐, 추종할 값이 아님                
        
        # ! delta pwr
        move = self.action_move_map[action]
        
        # ! tilt 각도 제한
        next_angle = current_angle + move
        if next_angle < 0 or next_angle > 30:
            next_angle = current_angle

        p_now = self.getSolarPower(hour, next_angle)
        p = self.getSolarPower(hour, current_angle)
        print(f"  env=> hour:{hour}, current_angle: {current_angle}, action: {action}, next_angle: {next_angle} | p_now: {p_now}, p: {p}, delta: {p_now-p}")

        return p_now-p, next_angle
    
    def step(self, action, hour):
        angle = self.agent_angle
        state = self.agent_state
        
        # * MDP를 만족하는지? (다음 상태가 현재 상태에서 수행된 동작과 상관 관계가 있는지)
        reward, next_angle = self.reward(angle, hour, action)
        next_state = self.next_state(reward)
        
        done = next_state == 2 # False # (state_ < 0 or state_ > 30)  # ! 보상(발전량 변화율)이 변화하면 종료
        
        print(f"reward:: state: {state}->{next_state}, next_angle:{next_angle}, action: {action}, hour: {hour}")
        # if done:
            # print(f"reward:: state: {state}->{next_state}, next_angle:{next_angle}, action: {action}, hour: {hour}")

        self.agent_state = next_state
        self.agent_angle = next_angle

        return next_state, reward, done