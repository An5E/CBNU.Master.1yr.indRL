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
                
        # self.goal_state = (12, 0) # ! MPA 경사각 , (time, tilt_angle)
        # self.start_state = (0, self.init_tilt) # ! 초기 경사각
        
        self.start_state = 0 # ! 초기 tilt angle
        self.agent_state = self.start_state
    
    def reset(self):
        self.agent_state = self.start_state
        return self.agent_state
    
    # ! %6~7, %13~14: update state
    def next_state(self, state, action):
        angle_factor = 2
        action_move_map = [x * angle_factor for x in [-2, -1, 0, 1, 2]] 
        
        move = action_move_map[action]
        
        # next_state = (state[0], state[1] + move)
        next_state = state + move
        
        if next_state < 0 or next_state > 20:
            next_state = state
        
        return next_state
        
    def getSolarPower(self, hour, tilt_angle):        
        # ? {hour} 곡선에서 x={tilt_angle}인 y값 구하기. l_mpa에서 참조
        return max(0, getRewardFromMPA(hour, tilt_angle, startHour=startHr, endHour=endHr))

    # ! %7, %15: reward = p_now-p
    def reward(self, state, action, hour, next_state):
        # ! 발전량 최대치가 나오는 경사각으로 이동
        # ! MPA 곡선은 비교 예시 데이터일 뿐, 추종할 값이 아님        
        # return self.getSolarPower(hour, next_state[1]) - self.getSolarPower(hour, state[1])
        
        # ! delta pwr
        return self.getSolarPower(hour, next_state) - self.getSolarPower(hour, state)
    
    def step(self, action, hour):
        state = self.agent_state
        next_state = self.next_state(state, action)
        reward = self.reward(state, action, hour, next_state)
        done = (reward < 0) # ! 보상(발전량 변화율)이 낮아지면 종료
        
        if done:
            debug_print(f"reward:: state: {state}->{next_state}, action: {action}, hour: {hour}")

        self.agent_state = next_state
        return next_state, reward, done