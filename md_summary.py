import ollama
import os
import sys

# 설정: 사용자 환경에 맞춘 모델명 (설치된 모델명과 정확히 일치해야 함)
MODEL_NAME = "gemma3:4b"  
INPUT_FILE = "input.md"

def summarize_markdown_file(file_path):
    # 1. 파일 존재 여부 확인
    if not os.path.exists(file_path):
        print(f"오류: '{file_path}' 파일을 찾을 수 없습니다.")
        return

    # 2. Markdown 파일 읽기
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            markdown_content = f.read()
            
        print(f"--- '{file_path}' 내용을 읽어왔습니다 ({len(markdown_content)} 글자) ---")
        print(f"--- '{MODEL_NAME}' 모델로 요약을 시작합니다 (GPU 메모리 모니터링 권장) ---\n")

    except Exception as e:
        print(f"파일 읽기 중 오류 발생: {e}")
        return

    # 3. 프롬프트 구성 (AI Engineering: Context 주입)
    # 시스템 프롬프트에 '역할'을 부여하고, 유저 메시지에 '데이터'를 넣습니다.
    messages = [
        {
            'role': 'system',
            'content': '너는 유능한 AI 어시스턴트야. 사용자가 제공하는 Markdown 문서의 내용을 파악하고, 핵심 내용을 한국어로 간결하게 요약해줘.'
        },
        {
            'role': 'user',
            'content': f"다음은 Markdown 문서의 내용이야:\n\n{markdown_content}\n\n이 내용을 요약해줘."
        }
    ]

    # 4. Ollama API 호출 (스트리밍 방식 적용)
    # 1060 3GB는 VRAM이 꽉 차면 느려질 수 있으므로, stream=True로 
    # 생성되는 토큰을 실시간으로 화면에 뿌려주는 것이 UX상 좋습니다.
    try:
        stream = ollama.chat(model=MODEL_NAME, messages=messages, stream=True)
        
        print(">>> AI 응답:")
        for chunk in stream:
            # 실시간으로 글자 출력 (줄바꿈 없이 이어서 출력)
            print(chunk['message']['content'], end='', flush=True)
            
        print("\n\n--- 완료 ---")

    except Exception as e:
        print(f"\n[오류 발생] 모델 실행 중 문제가 생겼습니다: {e}")
        print("팁: VRAM 부족일 수 있습니다. 실행 중인 다른 프로그램을 끄거나 문서를 줄여보세요.")

if __name__ == "__main__":
    summarize_markdown_file(INPUT_FILE)