from lark import Lark, Transformer, v_args
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

# --- Grammar Definition ---
AISPEC_GRAMMAR = r"""
start: system_def

system_def: "system" NAME "{" component_def* "}"

component_def: 
    | "component" NAME "{" member_def* "}" -> component_block
    | "import" NAME                        -> import_stmt

member_def:
    | "description" ":" STRING ";"         -> description
    | "state" NAME "{" field_list "}"      -> state_def
    | "function" NAME "(" param_list ")" "->" type_ref ";" -> function_def
    | "invariant" ":" logic_expr ";"       -> invariant
    | "constraint" ":" logic_expr ";"      -> constraint

field_list: (NAME ":" type_ref)*
param_list: (NAME ":" type_ref ("," NAME ":" type_ref)*)?

type_ref: NAME 
        | "List" "[" type_ref "]" 
        | "Map" "[" type_ref "," type_ref "]"

logic_expr: /[^\;]+/

NAME: /[a-zA-Z_]\w*/
STRING: /"[^"]*"/

%import common.WS
%ignore WS
"""

# --- AST Nodes ---

@dataclass
class TypeRef:
    name: str
    args: List['TypeRef'] = field(default_factory=list)

@dataclass
class Field:
    name: str
    type: TypeRef

@dataclass
class FunctionSpec:
    name: str
    params: List[Field]
    return_type: TypeRef

@dataclass
class StateSpec:
    name: str
    fields: List[Field]

@dataclass
class ComponentSpec:
    name: str
    description: str = ""
    states: List[StateSpec] = field(default_factory=list)
    functions: List[FunctionSpec] = field(default_factory=list)
    invariants: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)

@dataclass
class SystemSpec:
    name: str
    components: List[ComponentSpec] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)

# --- Transformer ---

class AISpecTransformer(Transformer):
    def start(self, items):
        return items[0]

    def system_def(self, items):
        name = items[0].value
        components = [i for i in items[1:] if isinstance(i, ComponentSpec)]
        imports = [i for i in items[1:] if isinstance(i, str)]
        return SystemSpec(name=name, components=components, imports=imports)

    def component_block(self, items):
        name = items[0].value
        spec = ComponentSpec(name=name)
        for item in items[1:]:
            if isinstance(item, tuple) and item[0] == 'desc':
                spec.description = item[1]
            elif isinstance(item, StateSpec):
                spec.states.append(item)
            elif isinstance(item, FunctionSpec):
                spec.functions.append(item)
            elif isinstance(item, tuple) and item[0] == 'invariant':
                spec.invariants.append(item[1])
            elif isinstance(item, tuple) and item[0] == 'constraint':
                spec.constraints.append(item[1])
        return spec

    def import_stmt(self, items):
        return items[0].value

    def description(self, items):
        return ('desc', items[0].value.strip('"'))

    def state_def(self, items):
        name = items[0].value
        fields = items[1]
        return StateSpec(name=name, fields=fields)

    def function_def(self, items):
        name = items[0].value
        params = items[1] or []
        ret_type = items[2]
        return FunctionSpec(name=name, params=params, return_type=ret_type)

    def invariant(self, items):
        return ('invariant', items[0].value.strip())

    def constraint(self, items):
        return ('constraint', items[0].value.strip())

    def field_list(self, items):
        fields = []
        for i in range(0, len(items), 2):
            fields.append(Field(name=items[i].value, type=items[i+1]))
        return fields

    def param_list(self, items):
        params = []
        if not items: return []
        # param_list returns a list of tokens/TypeRefs due to tree structure
        # Simplified handling:
        for i in range(0, len(items), 2):
             params.append(Field(name=items[i].value, type=items[i+1]))
        return params

    def type_ref(self, items):
        if len(items) == 1:
            return TypeRef(name=items[0].value)
        # Handle List/Map generic types
        base = items[0].value if hasattr(items[0], 'value') else items[0]
        args = items[1:] # items[1] would be the generic arg
        return TypeRef(name=base, args=args)
    
    def logic_expr(self, items):
        return items[0]

# --- Compiler API ---

class AISpecCompiler:
    def __init__(self):
        self.parser = Lark(AISPEC_GRAMMAR, start='start', parser='lalr')
        self.transformer = AISpecTransformer()

    def compile(self, source_code: str) -> SystemSpec:
        tree = self.parser.parse(source_code)
        return self.transformer.transform(tree)

    def compile_file(self, file_path: str) -> SystemSpec:
        with open(file_path, 'r', encoding='utf-8') as f:
            return self.compile(f.read())
