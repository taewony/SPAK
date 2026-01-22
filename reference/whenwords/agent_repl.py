import ollama
import os
import glob
import time
import sys
import convert_md_to_html

# 설정
MODEL_NAME = "gemma3:4b"  
TARGET_FOLDER = "./docs"

# 3) 전역 변수로 상태(State) 관리
answer = "" 

def call_llm(prompt, system_role="너는 유능한 분석가야."):
    """Ollama API 호출"""
    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {'role': 'system', 'content': system_role},
                {'role': 'user', 'content': prompt}
            ],
            stream=False
        )
        return response['message']['content']
    except Exception as e:
        return f"Error: {str(e)}"

def summarize_single_file(file_path):
    """개별 파일 요약 (Map)"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # RTX 1060 3GB VRAM 보호를 위한 길이 제한
    if len(content) > 3000:
        content = content[:3000] + "\n...(내용 잘림)..."
        
    prompt = f"다음 문서를 읽고 핵심 내용을 3줄로 요약해줘:\n\n{content}"
    return call_llm(prompt)

def process_summary_docs():
    """재귀적 요약 로직 실행 (Map-Reduce)"""
    md_files = glob.glob(os.path.join(TARGET_FOLDER, "*.md"))
    if not md_files:
        print("Error: 처리할 Markdown 파일이 없습니다.")
        return ""

    print(f"\n[System] 총 {len(md_files)}개의 파일을 분석합니다...")
    
    # Map 단계
    intermediate_summaries = []
    for i, file_path in enumerate(md_files):
        print(f"  [{i+1}/{len(md_files)}] Reading: {os.path.basename(file_path)}", end=" ", flush=True)
        try:
            summary = summarize_single_file(file_path)
            intermediate_summaries.append(f"File: {os.path.basename(file_path)}\nSummary: {summary}")
            print("-> Done.")
            time.sleep(1) # GPU 쿨다운
        except Exception as e:
            print(f"-> Fail: {e}")

    # Reduce 단계
    print("\n[System] 전체 문맥 통합(Reduce) 중... (잠시만 기다려주세요)")
    combined_context = "\n\n".join(intermediate_summaries)
    final_prompt = (
        f"다음은 여러 문서의 요약본이다. 전체 핵심 주제와 결론을 포함한 '종합 보고서'를 작성해:\n\n"
        f"{combined_context}"
    )
    
    result = call_llm(final_prompt, system_role="너는 프로젝트 매니저야.")
    return result

def main_repl():
    global answer # 전역 변수 사용 선언
    
    def print_help():
        print("명령어:")
        print("  summary docs   : ./docs 폴더 내 파일 요약 수행")
        print("  save answer    : 메모리에 있는 결과를 answer.md로 저장")
        print("  md2html <path> : 특정 md 파일을 html로 변환")
        print("                   (예: md2html playbook/example.md -> example.html)")
        print("  help           : 명령어 목록 보기")
        print("  exit           : 종료")

    print(f"--- AI Agent REPL ({MODEL_NAME}) ---")
    print_help()
    print("---------------------------------------")

    while True:
        try:
            # 1) Python REPL 입력 대기
            user_input = input("\n(agent) >>> ").strip()

            if user_input == "exit":
                print("Good bye.")
                break
            
            elif user_input == "help":
                print_help()

            # 2) 요약 명령 실행
            elif user_input == "summary docs":
                result = process_summary_docs()
                if result:
                    # 3) 결과 출력 및 변수에 저장
                    answer = result 
                    print("\n=== [LLM Answer] ===")
                    print(answer)
                    print("====================")
                    print("[System] 결과가 'answer' 변수에 저장되었습니다.")
                else:
                    print("[System] 결과 생성 실패.")

            # 4) 저장 명령 실행
            elif user_input == "save answer":
                if not answer:
                    print("[Warning] 저장할 내용이 없습니다. 먼저 'summary docs'를 실행하세요.")
                else:
                    try:
                        with open("answer.md", "w", encoding="utf-8") as f:
                            f.write(answer)
                        print(f"[System] 현재 폴더에 'answer.md' 파일로 저장했습니다.")
                    except Exception as e:
                        print(f"[Error] 파일 저장 실패: {e}")
            
            # 5) MD to HTML 변환
            elif user_input.startswith("md2html "):
                md_path = user_input.replace("md2html ", "").strip()
                try:
                    html_path = convert_md_to_html.convert_single_file(md_path)
                    print(f"[System] Converted '{md_path}' -> '{html_path}'")
                except Exception as e:
                    print(f"[Error] 변환 실패: {e}")

            # 그 외 입력
            else:
                if user_input:
                    print(f"알 수 없는 명령어입니다: {user_input}")

        except KeyboardInterrupt:
            print("\n(Ctrl+C detected. Type 'exit' to quit)")
        except Exception as e:
            print(f"[Critical Error] {e}")

if __name__ == "__main__":
    # docs 폴더가 없으면 생성
    if not os.path.exists(TARGET_FOLDER):
        os.makedirs(TARGET_FOLDER)
    
    main_repl()