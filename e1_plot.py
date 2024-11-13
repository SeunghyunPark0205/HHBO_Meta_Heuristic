import numpy as np
import matplotlib.pyplot as plt

# 변수 L 설정
L = 200  # 예시로 최대 반복 횟수를 200으로 설정

# l 값을 0부터 L까지 생성
l_values = np.linspace(0, L, 500)

# E_1 수식 계산
E_1_values = 2 * np.exp(-((2 * l_values) / L)**2)

# 그래프 그리기
plt.plot(l_values, E_1_values, label=r'$E_1 = 2e^{-(2l/L)^2}$')
plt.xlabel('l')
plt.ylabel('E_1')
plt.title(r'Plot of $E_1 = 2e^{-(2l/L)^2}$')
plt.legend()
plt.grid(True)
plt.show()

