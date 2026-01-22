import ollama
from bs4 import BeautifulSoup
import json
import markdown
import os
from rich.console import Console
from rich.tree import Tree

console = Console()

class DocumentEnv:
    def __init__(self, content, is_html=True):
        """
        초기화: 콘텐츠를 파싱하여 DOM 트리 생성
        """
        if not is_html:
            # Markdown인 경우 HTML로 변환
            content = markdown.markdown(content)
        
        # lxml 파서 사용
        self.soup = BeautifulSoup(content, 'lxml')
        console.print("[dim]DOM Environment initialized.[/dim]")

    @staticmethod
    def from_file(file_path):
        """
        파일 경로에서 문서를 로드 (md, html 지원)
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        _, ext = os.path.splitext(file_path)
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        is_html = ext.lower() in ['.html', '.htm', '.xml']
        return DocumentEnv(content, is_html=is_html)

    def map_reduce(self, selector, sub_query, model_name="gemma3:4b"):
        """
        [Recursive] Selector에 해당하는 모든 노드에 대해 각각 하위 에이전트를 실행(Map)하고,
        결과를 리스트로 반환(Reduce 전 단계)합니다.
        """
        targets = self.soup.select(selector)
        if not targets:
            return f"No elements found for selector: {selector}"
        
        results = []
        console.print(f"[bold magenta]🔄 Map-Reduce Triggered:[/bold magenta] Spawning {len(targets)} sub-agents for '{selector}'")
        
        for i, target in enumerate(targets):
            content = target.get_text(strip=True)
            # 내용이 너무 짧으면 건너뛰거나 포함 (정책 결정)
            if not content: 
                continue

            console.print(f"  [magenta]Sub-agent #{i+1}[/magenta] processing...")
            
            # Sub-Agent 호출 (독립된 LLM 세션)
            response = ollama.chat(
                model=model_name,
                messages=[
                    {'role': 'system', 'content': f"당신은 전체 문서의 일부분만 보고 있습니다. 주어진 텍스트를 바탕으로 다음 요청을 수행하세요: {sub_query}"},
                    {'role': 'user', 'content': f"Text Fragment:\n{content}"}
                ]
            )
            results.append(f"Node #{i+1} Result: {response['message']['content']}")
            
        return "\n---\n".join(results)

    def get_structure(self, root_selector=None):
        """
        문서의 뼈대(ID, 태그명)만 트리 구조로 반환 (토큰 절약용)
        root_selector가 있으면 해당 부분부터, 없으면 전체 문서
        """
        root = self.soup.select_one(root_selector) if root_selector else self.soup
        if not root:
            return "No element found."

        structure_lines = []
        
        # 재귀적으로 구조를 텍스트로 표현
        def traverse(element, depth=0):
            if element.name:
                indent = "  " * depth
                elem_id = f"#{element.get('id')}" if element.get('id') else ""
                # 내용이 너무 길면 생략, 짧으면 일부 표시
                text_preview = element.get_text(strip=True)[:30]
                if text_preview:
                    text_preview = f": {text_preview}..."
                
                line = f"{indent}<{element.name}{elem_id}>{text_preview}"
                structure_lines.append(line)
                
                for child in element.children:
                    if child.name: # 태그인 경우만 탐색
                        traverse(child, depth + 1)

        traverse(root)
        return "\n".join(structure_lines)

    def read_node(self, selector):
        """
        특정 CSS Selector에 해당하는 노드의 텍스트 내용을 반환
        """
        selected = self.soup.select(selector)
        if not selected:
            return f"No content found for selector: {selector}"
        
        # 여러 요소가 잡히면 구분해서 반환
        results = []
        for i, tag in enumerate(selected):
            content = tag.get_text(strip=True)
            tag_info = f"<{tag.name} id='{tag.get('id', 'N/A')}'>"
            results.append(f"--- Match {i+1} {tag_info} ---\n{content}\n")
            
        return "\n".join(results)

    def get_dom_tree_visual(self):
        """
        Rich 라이브러리를 사용한 시각화용 트리 객체 반환 (디버깅용)
        """
        root_tag = self.soup.find() # 최상위 태그
        if not root_tag:
            return Tree("Empty Document")

        tree = Tree(f"[bold blue]<{root_tag.name}>[/bold blue]")
        
        def add_children(node, soup_element):
            for child in soup_element.children:
                if child.name:
                    label = f"[green]<{child.name}>[/green]"
                    if child.get('id'):
                        label += f" [yellow]#{child.get('id')}[/yellow]"
                    branch = node.add(label)
                    add_children(branch, child)
        
        add_children(tree, root_tag)
        return tree

# 테스트용 코드
if __name__ == "__main__":
    # 더미 데이터 생성
    sample_xml = """
    <doc id="root">
        <section id="intro">
            <title>Introduction</title>
            <p>Welcome to the Recursive DOM Agent project.</p>
        </section>
        <section id="method">
            <title>Methodology</title>
            <div id="step1">Step 1: Setup</div>
            <div id="step2">Step 2: Coding</div>
        </section>
    </doc>
    """
    
    env = DocumentEnv(sample_xml)
    
    console.print("\n[bold]1. Structure View:[/bold]")
    print(env.get_structure())
    
    console.print("\n[bold]2. Read Node (#method):[/bold]")
    print(env.read_node("#method"))
    
    console.print("\n[bold]3. Visual Tree:[/bold]")
    console.print(env.get_dom_tree_visual())
