import numpy as np
import matplotlib.pyplot as plt
from dezero import Model
from dezero import optimizers
import dezero.layers as L
import dezero.functions as F

# 데이터셋 생성
np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100,1)

lr = 0.3
iters = 10000

class TwoLayerNet(Model):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.l1 = L.Linear(hidden_size)
        self.l2 = L.Linear(out_size)

    def forward(self,x):
        y = F.sigmoid(self.l1(x))
        y = self.l2(y)
        return y
    
model = TwoLayerNet(10, 1)

# ! 반복문으로 열거할 Optimizer를 (Label:Object) tuple로 정의
optimizerSet = [("MomentumSGD",optimizers.MomentumSGD(lr)), ("AdaGrad",optimizers.AdaGrad(lr)), ("Adam",optimizers.Adam())]

# ! Optimizer별 iter 경과에 따른 손실 계수 차트를 subplot으로 그리기 위한 (1*3) 레이아웃 정의 
plt.figure(figsize=(18,6))

# ! OptimizerSet에 정의된 tuple을 순서대로 호출
for k,item in enumerate(optimizerSet):
    item[1].setup(model) # 최적화할 모델을 옵티마이저에 등록

    print(f"# {item[0]}")
    # ! iter 경과별 loss 값을 담을 배열 선언
    losses = []

    # ! iters 횟수만큼 반복
    for i in range(iters):
        y_pred = model(x)
        loss = F.mean_squared_error(y, y_pred)

        model.cleargrads()
        loss.backward()

        item[1].update() # 옵티마이저로 매개변수 갱신
        if i % 1000 == 0:
            print(loss.data)
            losses.append(loss.data)
    # ! 차트 그릴 subplot 지정
    plt.subplot(1,3,k+1)

    # ! optimizerSet Label 호출
    plt.title(f"{item[0]} loss (lr={lr})")

    # ! 차트 X,Y Axis label 설정
    # plt.xlabel('iters/1000')
    # plt.ylabel('loss')

    t = np.arange(0, 1, .01)[:, np.newaxis]
    y_pred = model(t)

    print(y_pred)
    plt.plot(t, y_pred.data-loss.data, color='r')
    # ! subplot 간 가로 간격을 이격
    plt.subplots_adjust(wspace=0.3)

    # ! subplot y축 최소, 최대값을 각각 0,1로 설정 (3개 subplot 차트에 같은 기준 값 설정)
    # plt.ylim(0,1)

plt.show()