import json
from lark import Lark, Transformer
from pathlib import Path

# ==========================================
# 1. Flexible Grammar (JSON-like + Components)
# ==========================================

grammar = r"""
start: page

page: "page" NAME "{" body "}"

body: (pair | component)*

component: "component" NAME "{" body "}"

pair: key ":" value

key: NAME

# value 정의
value: STRING 
     | NUMBER 
     | boolean
     | array 
     | object 
     | function_call
     | loop_def
     | inline_component
     | VARIABLE

object: "{" body "}"
array: "[" [value ("," value)*] "]"

# 인라인 컴포넌트 (ex: WishlistButton { ... })
inline_component: NAME "{" body "}"

# 함수 호출 (ex: flexbox(...))
function_call: NAME "(" [args] ")"
args: arg ("," arg)*
arg: pair | value

# 반복문
loop_def: NAME "foreach" VARIABLE "{" body "}"

# [수정됨] boolean을 명시적 터미널로 정의
boolean: TRUE | FALSE
TRUE: "true"
FALSE: "false"

VARIABLE: "$" NAME ("." NAME)*

# Common Terminals
%import common.C_COMMENT
%import common.CNAME -> NAME
%import common.ESCAPED_STRING -> STRING
%import common.SIGNED_NUMBER -> NUMBER
%import common.WS

%ignore WS
%ignore C_COMMENT
"""

# ==========================================
# 2. Transformer (DSL -> Dictionary IR)
# ==========================================
class DSLToDict(Transformer):
    def start(self, items):
        return items[0]

    def page(self, items):
        return {
            "type": "page",
            "name": str(items[0]),
            "children": items[1]
        }

    def body(self, items):
        return items

    def component(self, items):
        return {
            "type": "component",
            "name": str(items[0]),
            "props": items[1]
        }

    def pair(self, items):
        return {items[0]: items[1]}

    def key(self, items):
        return str(items[0])

    def value(self, items):
        return items[0]

    def string(self, items):
        return items[0][1:-1] # 따옴표 제거

    def array(self, items):
        return items

    def object(self, items):
        result = {}
        for item in items:
            if isinstance(item, dict):
                result.update(item)
        return result
    
    def inline_component(self, items):
        comp_name = str(items[0])
        props = {}
        body_items = items[1]
        if isinstance(body_items, list):
            for item in body_items:
                if isinstance(item, dict):
                    props.update(item)
                    
        return {
            "type": "component_usage",
            "name": comp_name,
            "properties": props
        }

    def function_call(self, items):
        func_name = items[0]
        arguments = []
        if len(items) > 1:
            arguments = items[1]
        
        return {
            "type": "function", 
            "callee": str(func_name), 
            "args": arguments
        }
    
    def args(self, items):
        return items

    def arg(self, items):
        return items[0]

    def loop_def(self, items):
        comp_name = str(items[0])
        source = str(items[1])
        props = {}
        body_items = items[2]
        if isinstance(body_items, list):
            for item in body_items:
                if isinstance(item, dict):
                    props.update(item)
        
        return {
            "type": "loop",
            "component": comp_name,
            "source": source,
            "properties": props
        }

    def VARIABLE(self, tok):
        return str(tok)
    
    def STRING(self, tok):
        return str(tok)[1:-1]
    
    def NUMBER(self, tok):
        return float(tok)

    # [수정됨] 토큰 값을 확인하여 Boolean 반환
    def boolean(self, items):
        return items[0].type == "TRUE"

    def NAME(self, tok):
        return str(tok)
    
    # [추가] TRUE/FALSE 터미널 처리 (필요시)
    def TRUE(self, tok):
        return tok
    
    def FALSE(self, tok):
        return tok

# ==========================================
# 3. Main Parsing Logic
# ==========================================
if __name__ == "__main__":
    try:
        dsl_text = Path("input.dsl").read_text(encoding="utf-8")
        
        parser = Lark(grammar, parser="lalr")
        tree = parser.parse(dsl_text)
        
        tr = DSLToDict()
        ir_data = tr.transform(tree)
        
        Path("page_ir.json").write_text(json.dumps(ir_data, indent=2), encoding="utf-8")
        print("✅ Parsing Complete! 'page_ir.json' created.")
        
    except Exception as e:
        print("❌ Parsing Error:", e)