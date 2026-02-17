import torch

x = torch.tensor([0.0], requires_grad=True)
optimizer = torch.optim.SGD([x], lr=0.1)

for step in range(30):
    # 함수 정의: y = (x - 5)^2 + 10
    y = (x - 5)**2 + 10
    
    # 역전파: 현재 x 위치에서 y의 기울기(dy/dx)를 자동으로 계산
    optimizer.zero_grad() # 이전 기울기 초기화
    y.backward()          # 자동 미분 실행
    
    # 파라미터 업데이트: x = x - (lr * 기울기)
    optimizer.step()
    
    if (step + 1) % 5 == 0:
        print(f"Step {step+1}: x = {x.item():.4f}, y = {y.item():.4f}")

print(f"\n최종 결과: x는 {x.item():.2f}에 도달했습니다 (목표값: 5.00)")