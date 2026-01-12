import ollama
import json
import sys
import os
from rich.console import Console
from rich.prompt import Prompt
from dom_env import DocumentEnv

console = Console()

# 1. 사용할 도구(Tools) 정의
dom_tools = [
    {
        'type': 'function',
        'function': {
            'name': 'get_structure',
            'description': '문서의 전체 목차 구조(ID, 태그)를 확인합니다. 탐색 전에 반드시 먼저 호출하여 지도를 확보해야 합니다.',
            'parameters': {
                'type': 'object', 
                'properties': {
                    'root_selector': {
                        'type': 'string', 
                        'description': '특정 부분만 보고 싶을 때 사용 (선택 사항). 생략하면 전체를 봅니다.'
                    }
                }
            }
        }
    },
    {
        'type': 'function',
        'function': {
            'name': 'read_node',
            'description': 'CSS Selector를 사용하여 특정 섹션의 구체적인 텍스트 내용을 읽습니다.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'selector': {
                        'type': 'string', 
                        'description': '읽을 대상의 CSS Selector (예: #intro, section > title)'
                    }
                },
                'required': ['selector']
            }
        }
    },
    {
        'type': 'function',
        'function': {
            'name': 'map_reduce',
            'description': '여러 섹션이나 항목(List of items)을 한 번에 처리해야 할 때 사용합니다. (Large Query 처리용) Selector로 여러 요소를 선택하면, 각 요소마다 하위 에이전트가 실행되어 결과를 모아줍니다.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'selector': {
                        'type': 'string',
                        'description': '반복할 대상들의 CSS Selector (예: section, step, li)'
                    },
                    'sub_query': {
                        'type': 'string',
                        'description': '각 대상에게 수행할 개별 지시사항 (예: 이 항목을 요약해줘, 여기서 날짜를 추출해줘)'
                    }
                },
                'required': ['selector', 'sub_query']
            }
        }
    }
]

def run_agent(model_name="gemma3:4b", file_path=None):
    # 컨텍스트 로드
    if file_path and os.path.exists(file_path):
        console.print(f"[bold green]📂 Loading context from: {file_path}[/bold green]")
        env = DocumentEnv.from_file(file_path)
    else:
        console.print("[yellow]⚠️ No valid file provided. Using sample context.[/yellow]")
        sample_xml = """
        <doc id="manual">
            <header id="top"><title>Recursive DOM Agent Manual</title></header>
            <section id="chap1"><title>Chapter 1: Concept</title><p>Treat context as a database.</p></section>
            <section id="chap2"><title>Chapter 2: Implementation</title><step>Install Ollama</step></section>
        </doc>
        """
        env = DocumentEnv(sample_xml)
    
    messages = [{'role': 'system', 'content': '당신은 문서를 탐색하여 사용자의 질문에 답하는 에이전트입니다. 문서를 보려면 반드시 도구를 사용해야 합니다.'}]

    console.print(f"[bold green]🤖 Recursive DOM Agent Started ({model_name})[/bold green]")
    console.print("[dim]Type 'exit' to quit.[/dim]\n")

    while True:
        user_input = Prompt.ask("[bold cyan]User[/bold cyan]")
        if user_input.lower() in ['exit', 'quit']:
            break

        messages.append({'role': 'user', 'content': user_input})

        # LLM 호출 (도구 포함)
        response = ollama.chat(
            model=model_name,
            messages=messages,
            tools=dom_tools
        )
        
        msg = response['message']
        messages.append(msg) # 대화 내역 저장

        # 도구 호출이 발생했는지 확인
        if msg.get('tool_calls'):
            console.print(f"[yellow]⚡ Model decided to use tools: {len(msg['tool_calls'])} calls[/yellow]")
            
            for tool in msg['tool_calls']:
                fn_name = tool['function']['name']
                args = tool['function']['arguments']
                
                console.print(f"  [dim]Executing {fn_name}({args})...[/dim]")
                
                # 도구 실행
                result_content = ""
                if fn_name == 'get_structure':
                    result_content = env.get_structure(args.get('root_selector'))
                elif fn_name == 'read_node':
                    result_content = env.read_node(args['selector'])
                elif fn_name == 'map_reduce':
                    result_content = env.map_reduce(args['selector'], args['sub_query'])
                
                # 결과 출력
                console.print(f"  [dim]Result length: {len(result_content)} chars[/dim]")
                
                # 결과를 LLM에게 반환 (Role: tool)
                messages.append({
                    'role': 'tool',
                    'content': str(result_content),
                })
            
            # 도구 결과 포함하여 다시 LLM 호출 (최종 답변 생성)
            final_response = ollama.chat(model=model_name, messages=messages)
            console.print(f"\n[bold green]Agent:[/bold green] {final_response['message']['content']}\n")
            messages.append(final_response['message'])
            
        else:
            # 도구 없이 바로 답변한 경우
            console.print(f"\n[bold green]Agent:[/bold green] {msg['content']}\n")

if __name__ == "__main__":
    target_file = sys.argv[1] if len(sys.argv) > 1 else None
    run_agent(file_path=target_file)