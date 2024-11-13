import numpy as np
import matplotlib.pyplot as plt

# 목적 함수 (Sphere 함수)
def objective_function(x):
    return np.sum(x**2)

# HHBO 알고리즘
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
        self.history = []

    def optimize(self):
        for t in range(self.max_iter):
            E1 = 2 * np.exp(-((2 * t) / self.max_iter)**2)
            for i in range(self.num_agents):
                score = self.obj_func(self.agents[i])
                if score < self.best_score:
                    self.best_score = score
                    self.best_position = self.agents[i].copy()
            
            current_positions = []
            for i in range(self.num_agents):
                E2 = np.random.rand()
                E3 = np.random.uniform(-1, 1)
                theta = np.random.uniform(0, 2 * np.pi)
                radius = E1 * (self.ub - self.lb) * E2
                new_position = self.best_position + radius * np.array([np.cos(theta), np.sin(theta)])
                new_position = np.clip(new_position, self.lb, self.ub)
                self.agents[i] = new_position
                current_positions.append(new_position)
            
            self.history.append(np.array(current_positions))
            self.scores.append(self.best_score)
        return self.best_position, self.best_score

# PSO 알고리즘
class PSO:
    def __init__(self, obj_func, dim, lb, ub, num_agents=30, max_iter=100):
        self.obj_func = obj_func
        self.dim = dim
        self.lb = lb
        self.ub = ub
        self.num_agents = num_agents
        self.max_iter = max_iter
        self.positions = np.random.uniform(lb, ub, (num_agents, dim))
        self.velocities = np.zeros((num_agents, dim))
        self.best_personal_positions = self.positions.copy()
        self.best_personal_scores = np.full(num_agents, float("inf"))
        self.global_best_position = None
        self.global_best_score = float("inf")
        self.scores = []
        self.history = []

    def optimize(self):
        w, c1, c2 = 0.5, 1.5, 1.5
        for t in range(self.max_iter):
            current_positions = []
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
                self.velocities[i] = (w * self.velocities[i] +
                                      c1 * r1 * (self.best_personal_positions[i] - self.positions[i]) +
                                      c2 * r2 * (self.global_best_position - self.positions[i]))
                self.positions[i] = self.positions[i] + self.velocities[i]
                self.positions[i] = np.clip(self.positions[i], self.lb, self.ub)
                current_positions.append(self.positions[i])
            
            self.history.append(np.array(current_positions))
            self.scores.append(self.global_best_score)
        return self.global_best_position, self.global_best_score

# SSA 알고리즘
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
        self.history = []

    def optimize(self):
        for t in range(self.max_iter):
            c1 = 2 * np.exp(-(4 * t / self.max_iter)**2)
            current_positions = []
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
                current_positions.append(self.positions[i])
            
            self.history.append(np.array(current_positions))
            self.scores.append(self.best_score)
        return self.best_position, self.best_score

# 알고리즘 실행
dim = 2
lb = -10
ub = 10
max_iter = 30

hhbo = HHBO(objective_function, dim, lb, ub, max_iter=max_iter)
pso = PSO(objective_function, dim, lb, ub, max_iter=max_iter)
ssa = SSA(objective_function, dim, lb, ub, max_iter=max_iter)

hhbo.optimize()
pso.optimize()
ssa.optimize()

# 로그 스케일 성능 비교 시각화
plt.figure(figsize=(12, 8))
plt.plot(hhbo.scores, label="HHBO", marker='o')
plt.plot(pso.scores, label="PSO", marker='s')
plt.plot(ssa.scores, label="SSA", marker='^')
plt.xlabel("Iteration")
plt.ylabel("Best Score (Log Scale)")
plt.yscale("log")
plt.title("Comparison of HHBO, PSO, and SSA Optimization (Log Scale)")
plt.legend()
plt.grid()
plt.show()

# 각 단계별 에이전트 위치 시각화 함수
def plot_positions(history, title):
    plt.figure(figsize=(10, 6))
    for t, positions in enumerate(history):
        plt.clf()
        plt.scatter(positions[:, 0], positions[:, 1], color='blue', label="Agents")
        plt.scatter(hhbo.best_position[0], hhbo.best_position[1], color='red', marker='*', s=200, label="Best Position")
        plt.xlim(lb, ub)
        plt.ylim(lb, ub)
        plt.title(f"{title} Optimization Step {t+1}")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.legend()
        plt.pause(0.3)
    plt.show()

# 단계별 위치 시각화
print("HHBO Step-by-Step Optimization:")
plot_positions(hhbo.history, "HHBO")

print("PSO Step-by-Step Optimization:")
plot_positions(pso.history, "PSO")

print("SSA Step-by-Step Optimization:")
plot_positions(ssa.history, "SSA")

