import os
import sys
import time
import subprocess
from io import StringIO
from contextlib import redirect_stdout

try:
    import google.generativeai as genai
except ImportError:
    !pip install -q -U google-generativeai
    import google.generativeai as genai

from google.colab import userdata

# ==============================================================================
# 1. Environment & Tool Abstraction (환경과 도구)
# ==============================================================================
class Environment:
    """Colab 실행 환경을 캡슐화"""
    def __init__(self):
        self.context = {} # 실행 컨텍스트 (변수 저장소)

    def get_file_structure(self):
        try:
            return subprocess.getoutput("find . -maxdepth 2 -not -path '*/.*'")
        except:
            return "Unknown"

    def execute_python(self, code):
        """Python 코드를 실행하고 결과를 반환"""
        buffer = StringIO()
        try:
            with redirect_stdout(buffer):
                exec_globals = globals().copy()
                exec_globals.update(self.context)
                exec(code, exec_globals, self.context)
            result = buffer.getvalue()
            return f"[SUCCESS]\n{result if result.strip() else '(No Output)'}"
        except Exception as e:
            return f"[ERROR]\n{e}"

# ==============================================================================
# 2. Agent Class (독립적인 작업자)
# ==============================================================================
class Agent:
    def __init__(self, name, model_name="gemini-1.5-flash"):
        self.name = name
        self.env = Environment()
        self.model = self._setup_model(model_name)
        self.chat = self.model.start_chat(history=[])
        
    def _setup_model(self, model_name):
        # (기존의 모델 자동 탐색 로직을 여기에 포함)
        # 간소화를 위해 직접 지정, 실제론 위에서 짠 자동 탐색 로직 사용 권장
        return genai.GenerativeModel(
            model_name=model_name,
            system_instruction=f"""
            당신은 '{self.name}'입니다. 
            주어진 목표를 달성하기 위해 Python 코드를 작성하고 실행하세요.
            """
        )

    def run(self, goal, max_turns=5):
        print(f"\n🤖 **Agent [{self.name}] Started Goal:** {goal}")
        
        current_msg = f"목표: {goal}\n현재 파일 구조:\n{self.env.get_file_structure()}"
        
        for turn in range(max_turns):
            print(f"   ↳ Turn {turn+1} thinking...", end="")
            
            # API 호출 (재시도 로직 포함 필요)
            try:
                resp = self.chat.send_message(current_msg)
                content = resp.text
                print(" Done.")
            except Exception as e:
                print(f" Error: {e}")
                break

            if "```python" in content:
                code = content.split("```python")[1].split("```")[0].strip()
                result = self.env.execute_python(code)
                print(f"     [Exec] Result length: {len(result)}")
                current_msg = f"실행 결과:\n{result}\n다음 단계는?"
                
                if "DONE" in content:
                    print(f"✅ **Agent [{self.name}] Finished!**")
                    return "DONE"
            else:
                current_msg = "Python 코드로 행동하세요."
                if "DONE" in content:
                    return "DONE"

# ==============================================================================
# 3. Main Execution
# ==============================================================================
# API 키 설정 (이전과 동일하게 처리)
try:
    api_key = userdata.get('GEMINI_API_KEY')
except:
    api_key = input("API Key: ")
genai.configure(api_key=api_key)

# 메인 실행
root_agent = Agent("RootBuilder")
root_agent.run("현재 폴더에 'hello_world.py'를 만들고 'print(hello)'를 작성해.")