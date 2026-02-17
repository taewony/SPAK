import torch
import matplotlib.pyplot as plt
import numpy as np

# 1. 초기 변수 설정
x = torch.tensor([0.0], requires_grad=True)
optimizer = torch.optim.SGD([x], lr=0.1)

# 2. 변화 과정을 기록할 리스트 (초기값 포함)
xs = [x.item()]
ys = [((x - 5)**2 + 10).item()]

# 3. 30 스텝 경사 하강법 수행
for step in range(30):
    optimizer.zero_grad()
    y = (x - 5)**2 + 10      # 현재 x에 대한 손실
    y.backward()              # 기울기 계산
    optimizer.step()          # x 업데이트

    # 업데이트된 x와 그 때의 y 저장
    xs.append(x.item())
    ys.append(((x - 5)**2 + 10).item())

    # (선택) 중간 결과 출력
    if (step + 1) % 5 == 0:
        print(f"Step {step+1}: x = {x.item():.4f}, y = {ys[-1]:.4f}")

print(f"\n최종 결과: x = {x.item():.2f} (목표: 5.00)")

# 4. 그래프 그리기
# 4.1 함수 곡선 데이터 준비
x_curve = np.linspace(-2, 12, 200)
y_curve = (x_curve - 5)**2 + 10

plt.figure(figsize=(10, 6))
plt.plot(x_curve, y_curve, 'b-', label='f(x) = (x-5)² + 10', linewidth=2)

# 4.2 경사 하강법 궤적 그리기
plt.scatter(xs, ys, c=range(len(xs)), cmap='viridis', s=50, zorder=5)
plt.plot(xs, ys, 'r--', alpha=0.5, label='Gradient descent path')

# 4.3 시작점과 끝점 강조
plt.scatter(xs[0], ys[0], c='red', s=100, marker='o', label='Start (x=0)', edgecolors='black')
plt.scatter(xs[-1], ys[-1], c='green', s=100, marker='*', label='Final (x≈5)', edgecolors='black')

# 4.4 최적점 표시
plt.scatter(5, 10, c='black', s=150, marker='X', label='Global minimum (5,10)')

plt.title('Gradient Descent on a Simple Quadratic Function')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True, alpha=0.3)
plt.colorbar(label='Step number')
plt.show()