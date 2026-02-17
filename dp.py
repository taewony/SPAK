import torch

x = torch.tensor([6.0], requires_grad=True)
optimizer = torch.optim.SGD([x], lr=0.1)

# 옵티마이저가 현재 저장된 기울기(x.grad)를 사용하여 파라미터를 업데이트합니다.
# SGD의 업데이트 규칙: x = x - (lr * x.grad)
# 예를 들어 첫 스텝에서 x=0, grad=-10, lr=0.1이면 x는 
# x = 0 - (0.1 * -10) = 1로 업데이트됩니다.
# 이 과정을 반복하면서 점차 최솟점인 x=5에 가까워집니다.

for step in range(30):
    # 함수 정의: y = (x - 5)^2 + 10
    y = (x - 1)**2 + 10
    
    # 역전파: 현재 x 위치에서 y의 기울기(dy/dx)를 자동으로 계산
    optimizer.zero_grad() # 이전 기울기 초기화
    y.backward()          # 자동 미분 실행
    
    # 파라미터 업데이트: x = x - (lr * 기울기)
    optimizer.step()
    
    if step % 3 == 0:
        print(f"Step {step+1}: x = {x.item():.4f}, y = {y.item():.4f}")

print(f"\n최종 결과: x는 {x.item():.2f}에 도달했습니다.")