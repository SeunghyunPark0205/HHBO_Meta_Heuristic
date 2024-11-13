import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Iris 데이터셋 로드 및 전처리
data = load_iris()
X = data.data
y = data.target.reshape(-1, 1)
encoder = OneHotEncoder(sparse_output=False)
y = encoder.fit_transform(y)

# 학습용, 테스트용 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 신경망 구조 설정 (입력층 4, 은닉층 9, 출력층 3)
input_size = X.shape[1]
hidden_size = 9
output_size = y.shape[1]

# 신경망의 가중치와 편향에 대한 총 파라미터 수
dim = input_size * hidden_size + hidden_size + hidden_size * output_size + output_size

# 소프트맥스 함수 정의
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# 목적 함수 (MSE)
def neural_network_mse(weights):
    # 가중치와 편향 분할
    w1 = weights[:input_size * hidden_size].reshape((input_size, hidden_size))
    b1 = weights[input_size * hidden_size:input_size * hidden_size + hidden_size]
    w2 = weights[input_size * hidden_size + hidden_size:input_size * hidden_size + hidden_size + hidden_size * output_size].reshape((hidden_size, output_size))
    b2 = weights[-output_size:]
    
    # 순전파 계산
    hidden_layer = np.maximum(0, np.dot(X_train, w1) + b1)  # ReLU 활성화 함수
    output_layer = np.dot(hidden_layer, w2) + b2
    predictions = softmax(output_layer)
    
    # 평균 제곱 오차 계산
    mse = np.mean((predictions - y_train) ** 2)
    return mse

# 알고리즘 클래스 정의 (HHBO, PSO, SSA)
class HHBO:
    def __init__(self, obj_func, dim, lb, ub, num_agents=30, max_iter=100):
        self.obj_func = obj_func
        self.dim = dim
        self.lb = lb
        self.ub = ub
        self.num_agents = num_agents
        self.max_iter = max_iter
        self.agents = np.random.uniform(lb, ub, (num_agents, dim))
        self.best_position = None
        self.best_score = float("inf")
        self.scores = []

    def optimize(self):
        for t in range(self.max_iter):
            E1 = 2 * np.exp(-((2 * t) / self.max_iter)**2)
            for i in range(self.num_agents):
                score = self.obj_func(self.agents[i])
                if score < self.best_score:
                    self.best_score = score
                    self.best_position = self.agents[i].copy()
            
            for i in range(self.num_agents):
                E2 = np.random.rand()
                E3 = np.random.uniform(-1, 1)
                theta = np.random.uniform(0, 2 * np.pi, self.dim)
                radius = E1 * (self.ub - self.lb) * E2
                new_position = self.best_position + radius * np.cos(theta)
                new_position = np.clip(new_position, self.lb, self.ub)
                self.agents[i] = new_position

            self.scores.append(self.best_score)
        return self.best_position, self.best_score

class PSO:
    def __init__(self, obj_func, dim, lb, ub, num_agents=30, max_iter=100, w=0.04, c1=2, c2=2):
        self.obj_func = obj_func
        self.dim = dim
        self.lb = lb
        self.ub = ub
        self.num_agents = num_agents
        self.max_iter = max_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.positions = np.random.uniform(lb, ub, (num_agents, dim))
        self.velocities = np.zeros((num_agents, dim))
        self.best_personal_positions = self.positions.copy()
        self.best_personal_scores = np.full(num_agents, float("inf"))
        self.global_best_position = None
        self.global_best_score = float("inf")
        self.scores = []

    def optimize(self):
        for t in range(self.max_iter):
            for i in range(self.num_agents):
                score = self.obj_func(self.positions[i])
                if score < self.best_personal_scores[i]:
                    self.best_personal_scores[i] = score
                    self.best_personal_positions[i] = self.positions[i].copy()
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = self.positions[i].copy()
            
            for i in range(self.num_agents):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                self.velocities[i] = (self.w * self.velocities[i] +
                                      self.c1 * r1 * (self.best_personal_positions[i] - self.positions[i]) +
                                      self.c2 * r2 * (self.global_best_position - self.positions[i]))
                self.positions[i] = self.positions[i] + self.velocities[i]
                self.positions[i] = np.clip(self.positions[i], self.lb, self.ub)

            self.scores.append(self.global_best_score)
        return self.global_best_position, self.global_best_score

class SSA:
    def __init__(self, obj_func, dim, lb, ub, num_agents=30, max_iter=100):
        self.obj_func = obj_func
        self.dim = dim
        self.lb = lb
        self.ub = ub
        self.num_agents = num_agents
        self.max_iter = max_iter
        self.positions = np.random.uniform(lb, ub, (num_agents, dim))
        self.best_position = None
        self.best_score = float("inf")
        self.scores = []

    def optimize(self):
        for t in range(self.max_iter):
            c1 = 2 * np.exp(-(4 * t / self.max_iter)**2)
            for i in range(self.num_agents):
                score = self.obj_func(self.positions[i])
                if score < self.best_score:
                    self.best_score = score
                    self.best_position = self.positions[i].copy()
            
            for i in range(1, self.num_agents):
                if i < self.num_agents / 2:
                    self.positions[i] = self.best_position + c1 * (np.random.rand(self.dim) * (self.ub - self.lb) + self.lb)
                else:
                    self.positions[i] = self.positions[i] + np.random.uniform(-1, 1, self.dim) * np.abs(self.positions[i] - self.best_position)
                self.positions[i] = np.clip(self.positions[i], self.lb, self.ub)

            self.scores.append(self.best_score)
        return self.best_position, self.best_score

# 설정값
lb, ub = -1, 1
num_agents, max_iter = 30, 100

# 알고리즘 실행
hhbo = HHBO(neural_network_mse, dim, lb, ub, num_agents=num_agents, max_iter=max_iter)
pso = PSO(neural_network_mse, dim, lb, ub, num_agents=num_agents, max_iter=max_iter)
ssa = SSA(neural_network_mse, dim, lb, ub, num_agents=num_agents, max_iter=max_iter)

hhbo.optimize()
pso.optimize()
ssa.optimize()

# 최적화 과정 시각화
plt.figure(figsize=(12, 8))
plt.plot(hhbo.scores, label="HHBO", marker='o')
plt.plot(pso.scores, label="PSO", marker='s')
plt.plot(ssa.scores, label="SSA", marker='^')
plt.xlabel("Iteration")
plt.ylabel("MSE (Log Scale)")
plt.yscale("log")
plt.title("Optimization of Neural Network Weights using HHBO, PSO, and SSA")
plt.legend()
plt.grid()
plt.show()

