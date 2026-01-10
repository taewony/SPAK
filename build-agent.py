import os
import sys
import subprocess
import glob
import json
import google.generativeai as genai
from io import StringIO
from contextlib import redirect_stdout

# 1. 설정
API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=API_KEY)

# 2. 시스템 프롬프트 (RLM 철학 적용)
SYSTEM_PROMPT = """
당신은 'Recursive Build Agent'입니다.
당신의 목표는 사용자의 요청이나 spec 문서를 기반으로 프로젝트를 구축하는 것입니다.

**핵심 원칙:**
1. 당신은 직접 텍스트를 생성하여 파일을 만들지 않습니다. 대신 **Python 코드를 생성하여 실행**함으로써 파일을 조작합니다.
2. 복잡한 논리(예: 트리 구조 계산, 파일 간 의존성 확인)는 반드시 Python 코드로 계산하여 확인합니다.
3. `context`라는 전역 변수에 현재 작업 상태를 저장할 수 있습니다.
4. 작업은 항상 [상태 확인] -> [코드 생성] -> [실행 결과 확인] -> [다음 작업] 순서로 진행합니다.

**사용 가능한 환경:**
- 현재 디렉토리: 프로젝트 루트
- Python 라이브러리: os, sys, json, glob, subprocess 등 표준 라이브러리
"""

model = genai.GenerativeModel(
    model_name="gemini-2.0-flash-exp", # 또는 1.5-pro
    system_instruction=SYSTEM_PROMPT
)

# 3. Python REPL (샌드박스)
context = {} # RLM의 핵심: 상태를 저장하는 메모리

def execute_python_code(code):
    """LLM이 생성한 코드를 실행하고 stdout과 context 변화를 캡처"""
    buffer = StringIO()
    global context
    
    try:
        # 안전한 실행을 위해 일부 제한을 둘 수 있음 (프로토타입에서는 생략)
        with redirect_stdout(buffer):
            exec(code, globals(), context)
        result = buffer.getvalue()
        return f"[SUCCESS]\nOutput:\n{result}"
    except Exception as e:
        return f"[ERROR]\n{str(e)}"

# 4. 재귀적 실행 루프 (Recursive Loop)
def run_agent(goal):
    chat = model.start_chat(history=[])
    
    # 초기 상태 주입 (현재 파일 구조)
    # Windows 환경 호환성을 위해 subprocess 호출 수정 또는 예외 처리
    try:
        if os.name == 'nt': # Windows
             # Windows에서는 find 명령어가 다르므로 dir로 대체하거나 파이썬으로 구현
             # 여기서는 간단히 os.walk를 이용한 파이썬 로직으로 대체 가능하지만,
             # 에이전트가 스스로 파악하도록 빈 상태로 시작해도 무방함.
             # 일단 간단한 dir 명령어로 대체
             file_tree = subprocess.getoutput("dir /B")
        else:
            file_tree = subprocess.getoutput("find . -maxdepth 2 -not -path '*/.*'")
    except Exception:
        file_tree = "파일 구조를 읽을 수 없음 (권한 문제 등)"

    current_message = f"목표: {goal}\n\n현재 파일 구조:\n{file_tree}\n\n첫 번째 단계를 위한 Python 코드를 작성하거나, 질문을 하세요."

    print(f"🎯 Goal: {goal}")

    while True:
        # 1. LLM에게 생각 요청
        # [수정됨] sendMessage -> send_message
        try:
            response = chat.send_message(current_message)
            content = response.text
        except Exception as e:
            print(f"❌ API Error: {e}")
            break
        
        # 2. 코드 블록 파싱 (```python ... ```)
        if "```python" in content:
            code_start = content.find("```python") + 9
            code_end = content.find("```", code_start)
            code = content[code_start:code_end].strip()
            
            print(f"\n🤖 [Thought & Code]:\n{content}")
            
            # 3. REPL 실행 (Deterministic Process 위임)
            print(f"\n⚙️ [Executing Code]...")
            exec_result = execute_python_code(code)
            print(f"✅ [Result]:\n{exec_result}")
            
            # 4. 결과를 다음 턴의 입력으로 사용 (Recursion)
            current_message = f"코드 실행 결과:\n{exec_result}\n\n다음 단계는 무엇입니까?"
            
            # 'DONE' 시그널이 있으면 종료 로직 추가 가능
        else:
            # 코드가 없으면 대화/질문으로 간주
            print(f"\n🤖 [Agent]: {content}")
            user_input = input("\n👤 [User] (Type 'exit' to quit): ")
            if user_input.lower() == 'exit':
                break
            current_message = user_input

if __name__ == "__main__":
    if len(sys.argv) < 2:
        # 테스트를 위해 기본값 설정
        default_goal = "GPU programming을 설명하는 학습용 index.html 파일을 만들어줘."
        print(f"Usage: python agent.py 'Your Build Goal'")
        print(f"No goal provided. Using default: {default_goal}")
        run_agent(default_goal)
    else:
        run_agent(sys.argv[1])