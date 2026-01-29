from lark import Lark, Transformer
from pathlib import Path

# ==========================================
# 1. 문법 정의 (Grammar)
# ==========================================

grammar = r"""
?start: page

page: "page" NAME "{" page_body "}"

page_body: (component | layout | style | behavior)*

component: "component" NAME "{" layout "}"

layout: "layout" "{" node+ "}"

?node: loop
     | element
     | textnode

# Loop
loop: NAME "(" [props] ")" "foreach" NAME

props: prop ("," prop)*
prop: NAME ":" expr

# Element
# 태그명과 선택자에 하이픈 허용 (CSS_IDENT 사용)
element: tag selector? ["{" node* "}"]

tag: CSS_IDENT
selector: ("#" CSS_IDENT)? ("." CSS_IDENT)*

# TextNode
textnode: "text" value

value: ESCAPED_STRING | expr
# 표현식(변수명)은 여전히 일반 NAME(하이픈 불가) 사용
expr: NAME ("." NAME)*

# Style
style: "style" "{" style_rule+ "}"
style_rule: selector "{" css_decl+ "}"

# [수정] 속성명에 하이픈 허용 (grid-template-columns 등)
css_decl: CSS_IDENT ":" /[^;]+/ ";"

# Behavior
behavior: "behavior" "{" rule+ "}"
rule: "when" NAME "on" selector "then" action ";"
action: /[^;]+/

# --- Terminals ---
# 하이픈(-)을 포함하는 식별자 정의 (CSS 속성, 태그, 클래스용)
CSS_IDENT: /[a-zA-Z0-9-_]+/

%import common.CNAME -> NAME
%import common.ESCAPED_STRING
%import common.WS
%import common.C_COMMENT
%import common.CPP_COMMENT

%ignore WS
%ignore C_COMMENT
%ignore CPP_COMMENT
"""

# ==========================================
# 2. IR (Intermediate Representation)
# ==========================================

class IR:
    def __init__(self):
        self.components = {}
        self.layout = []
        self.styles = []
        self.behaviors = []

# ==========================================
# 3. Transformer
# ==========================================

class DSLTransformer(Transformer):
    def __init__(self):
        super().__init__()
        self.ir = IR()

    def page(self, items):
        return self.ir

    def page_body(self, items):
        for it in items:
            if isinstance(it, list):
                self.ir.layout.extend(it)

    def component(self, items):
        name = items[0]
        layout_nodes = items[1]
        self.ir.components[name] = layout_nodes

    def layout(self, items):
        return items 

    def element(self, items):
        tag = items[0]
        selector = ""
        children = []
        
        current_idx = 1
        # selector가 있는지 확인 (문자열 체크)
        if len(items) > current_idx and isinstance(items[current_idx], str):
            selector = items[current_idx]
            current_idx += 1
            
        if len(items) > current_idx:
            children = items[current_idx:]
            
        return ("el", tag, selector, children)

    def selector(self, items):
        return "".join(items)

    def tag(self, items):
        return str(items[0])
    
    # CSS_IDENT 토큰을 문자열로 변환
    def CSS_IDENT(self, tok):
        return str(tok)

    def textnode(self, items):
        return ("text", items[0])

    def value(self, items):
        val = items[0]
        if hasattr(val, 'type') and val.type == "ESCAPED_STRING":
            return val[1:-1]
        return val

    def expr(self, items):
        return "{{" + ".".join(items) + "}}"

    def loop(self, items):
        comp_name = items[0]
        list_source = items[-1]
        return ("loop", comp_name, list_source)

    def style(self, items):
        for r in items:
            self.ir.styles.append(r)

    def style_rule(self, items):
        sel = items[0]
        decls = items[1:]
        return (sel, decls)

    def css_decl(self, items):
        return f"{items[0]}:{items[1]};"

    def behavior(self, items):
        for r in items:
            self.ir.behaviors.append(r)

    def rule(self, items):
        ev = items[0]
        sel = items[1]
        act = items[2]
        return (ev, sel, act)

    def action(self, items):
        return str(items[0]).strip()

    def NAME(self, tok):
        return str(tok)

# ==========================================
# 4. Code Generation
# ==========================================

VOID_ELEMENTS = {"input", "img", "br", "hr", "meta", "link"}

def render_nodes(nodes, ir):
    html = ""
    for n in nodes:
        if n[0] == "el":
            _, tag, sel, children = n
            attrs = ""
            
            if sel:
                idv = ""
                classes = []
                parts = sel.replace("#", " #").replace(".", " .").split()
                for p in parts:
                    if p.startswith("#"):
                        idv = p[1:]
                    elif p.startswith("."):
                        classes.append(p[1:])
                
                if idv:
                    attrs += f' id="{idv}"'
                if classes:
                    attrs += f' class="{" ".join(classes)}"'

            if tag in VOID_ELEMENTS:
                html += f"<{tag}{attrs}>\n"
            else:
                inner_html = render_nodes(children, ir)
                html += f"<{tag}{attrs}>{inner_html}</{tag}>\n"

        elif n[0] == "text":
            html += n[1]

        elif n[0] == "loop":
            _, comp, src = n
            tmpl = render_nodes(ir.components.get(comp, []), ir)
            html += f"\n\n{tmpl}\n"

    return html

def build_html(ir):
    body = render_nodes(ir.layout, ir)

    css = ""
    for sel, decls in ir.styles:
        css += f"{sel} {{ {' '.join(decls)} }}\n"

    js = ""
    for ev, sel, act in ir.behaviors:
        js += f"""
    document.querySelectorAll("{sel}").forEach(el => {{
        el.addEventListener("{ev}", () => {{ {act} }});
    }});
"""

    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Generated Page</title>
    <style>
{css}
    </style>
</head>
<body>
{body}
<script>
{js}
</script>
</body>
</html>
"""

# ==========================================
# 5. Main Execution
# ==========================================

if __name__ == "__main__":
    try:
        dsl_text = Path("input.dsl").read_text(encoding="utf-8")
        
        # Earley 파서 사용 (유연한 문법 처리)
        parser = Lark(grammar, parser="earley")
        tree = parser.parse(dsl_text)

        tr = DSLTransformer()
        ir = tr.transform(tree)

        html = build_html(ir)

        Path("index.html").write_text(html, encoding="utf-8")
        print("✅ Success! 'index.html' has been generated.")
        
    except Exception as e:
        print("❌ Error:", e)