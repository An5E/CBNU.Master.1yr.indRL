# 주제  
 - Q-learning 학술 논문 읽고 요약, 발표  

 - 논문 내용 직접 구현, 테스트는 선택 사항    

 - 선정 주제: 단축 태양광 추종 장치(Single-Axis Solar Tracker)의 MPPT(최적 발전 각도 추종) 제어에 대한 Q-learning 강화 학습 적용  

## 서론  
 - 강화학습 기법 중 Q-learning 적용한 논문 탐색  

 - 연구 분야와 직접적 연관 없음  

 - 다음 두 가지 방법이 일반적으로 사용됨  

   1. 발전 설비 변위에서 GPS 모듈 통해 태양 방위, 천정각을 구해 하는 방식    

   2. CDS 광다이오드를 태양광 패널 전면부에 부착. 광원 감지하여 그 방향으로 제어하는 방식  

## 본론  
### 🅿️ 문제 접근  
  
![alt text](image.png)  

 - 주제 구상 후보군 중 다음 조건 부합하는 주제 선택  
  
   1. **Agent(동작 주체)** 가 **State(현재 상태)** 를 갖고 **Action(행동)** 을 수행할 수 있는 **Environment(환경)** 존재할 것   
  
   2. Agent 가 State 에서 Action 수행하여 다음 State 에서 기대할 수 있는 **Reward(보상)** 의 개선이 가능할 것    
  

 - 주제 구상 후보군  

   1. 태양광 발전량 예측 (LSTM 회귀 모델 기반. Q-learning 적용과 거리가 멂)  

   2. **|> 양축 태양광 추종 장치 (State, Action, Reward 개선 가능성 명확)**   


### 📜 논문 선정  
  
![alt text](image-1.png)  

- 22년 9월, 산업 IoT 및 빅데이터, 공급망 사슬 분야 국제 컨퍼런스 (IIoTBDSC, 베이징) 발표 논문  

- Q-learning 강화학습을 단축 태양광 추종 장치 최적 발전 각도. MPPT 제어에 적용  
  
### 📄 요약  
  
![alt text](image-2.png)  

- 

### 🖇️ 연구 분야 및 현업 연관성  
  
![alt text](image-3.png)  

### 🏁 RL 모델 설정  
  
![alt text](image-4.png)  
  
### 🎛️ 정책 제어 (Policy Control)  
  
![alt text](image-5.png)  
  
### 🛠️ 정책 평가 (Policy Evaluation)  
  
![alt text](image-6.png)  
  
### ▶️ ~~실증~~ 재현 순서  
  
![alt text](image-7.png)  
  
### ⏹️ ~~실증~~ 재현 결과  
  
![alt text](image-8.png)  
  
![alt text](image-9.png)  

### 💡 개선 아이디어 및 고찰  
  
![alt text](image-10.png)  
  

## 결론  
### 💡 고찰  
 - 
### ❓강의담당 교수님 피드백  
 - 
