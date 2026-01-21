from lark import Lark, Transformer, v_args, Token
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

# --- Grammar Definition ---
AISPEC_GRAMMAR = r"""
start: meta_def? contract_def? system_def

meta_def: "meta" "{" meta_field* "}"

meta_field: NAME "=" STRING

system_def: "system" NAME "{" component_def* "}"
          | "kernel" NAME "{" component_def* "}"

contract_def: "contract" NAME "{" contract_member* "}"

contract_member:
    | NAME "=" list_literal                -> contract_list_field
    | "autonomy" "{" config_field* "}"     -> autonomy_block

autonomy_block: "autonomy" "{" config_field* "}"

config_field: NAME "=" value

list_literal: "[" (STRING ("," STRING)*)? "]"

value: STRING | NUMBER | "true" | "false" | list_literal

component_def: 
    | "component" NAME "{" member_def* "}" -> component_block
    | "component" NAME                     -> component_decl
    | "effect" NAME "{" effect_op* "}"     -> effect_block
    | "workflow" NAME "(" param_list ")" "{" workflow_step* "}" -> workflow_def
    | "struct" NAME "{" field_list "}"     -> struct_def
    | "import" NAME                        -> import_stmt

member_def:
    | "description" ":" STRING ";"?         -> description
    | "state" NAME "{" field_list "}"      -> state_def
    | "function" NAME "(" param_list ")" "->" type_ref ";"? -> function_def
    | "function" NAME "(" param_list ")" "->" type_ref func_body -> function_def_body
    | "invariant" ":" STRING ";"?           -> invariant_string
    | "constraint" ":" STRING ";"?          -> constraint_string
    | "workflow" NAME "(" param_list ")" "{" workflow_step* "}" -> workflow_def

effect_op: "operation" NAME "(" param_list ")" "->" type_ref ";"

workflow_step: "step" NAME "{" logic_expr "}"

func_body: "{" logic_expr "}"

field_list: (field_decl)*
param_list: (field_decl ("," field_decl)*)?

field_decl: NAME ":" type_ref

type_ref: NAME                          -> type_simple
        | "List" generic_args           -> type_list
        | "Map" generic_args            -> type_map
        | "Result" generic_args         -> type_result

generic_args: "[" type_ref ("," type_ref)* "]"
            | "<" type_ref ("," type_ref)* ">"

logic_expr: /[^\}]+/

NAME: /[a-zA-Z_]\w*/
STRING: /"[^"]*"/
NUMBER: /\d+(\.\d+)?/

%import common.WS
%import common.CPP_COMMENT 
%import common.C_COMMENT
%import common.SH_COMMENT

%ignore WS
%ignore CPP_COMMENT
%ignore C_COMMENT
%ignore SH_COMMENT
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
    body: Optional[str] = None

@dataclass
class StateSpec:
    name: str
    fields: List[Field]

@dataclass
class EffectSpec:
    name: str
    operations: List[FunctionSpec] = field(default_factory=list)

@dataclass
class WorkflowSpec:
    name: str
    params: List[Field]
    steps: List[str] = field(default_factory=list)

@dataclass
class StructSpec:
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
    workflows: List[WorkflowSpec] = field(default_factory=list)

@dataclass
class AutonomySpec:
    config: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ContractSpec:
    name: str
    lists: Dict[str, List[str]] = field(default_factory=dict)
    autonomy: Optional[AutonomySpec] = None

@dataclass
class SystemSpec:
    name: str
    metadata: Dict[str, str] = field(default_factory=dict)
    components: List[ComponentSpec] = field(default_factory=list)
    structs: List[StructSpec] = field(default_factory=list)
    effects: List[EffectSpec] = field(default_factory=list)
    workflows: List[WorkflowSpec] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    contract: Optional[ContractSpec] = None

# --- Transformer ---

class AISpecTransformer(Transformer):
    def start(self, items):
        # items can be [meta, system, contract] or [system, contract] or [meta, system] etc.
        system = next((i for i in items if isinstance(i, SystemSpec)), None)
        meta = next((i for i in items if isinstance(i, dict)), None)
        contract = next((i for i in items if isinstance(i, ContractSpec)), None)
        
        if system:
            if meta:
                system.metadata = meta
            if contract:
                system.contract = contract
        return system

    def meta_def(self, items):
        return {k: v for k, v in items}

    def meta_field(self, items):
        return (items[0].value, items[1].value.strip('"'))

    def system_def(self, items):
        name = items[0].value
        components = [i for i in items[1:] if isinstance(i, ComponentSpec)]
        structs = [i for i in items[1:] if isinstance(i, StructSpec)]
        effects = [i for i in items[1:] if isinstance(i, EffectSpec)]
        workflows = [i for i in items[1:] if isinstance(i, WorkflowSpec)]
        imports = [i for i in items[1:] if isinstance(i, str)]
        return SystemSpec(name=name, components=components, structs=structs, effects=effects, workflows=workflows, imports=imports)

    def contract_def(self, items):
        name = items[0].value
        spec = ContractSpec(name=name)
        for item in items[1:]:
            if isinstance(item, tuple) and item[0] == 'list':
                spec.lists[item[1]] = item[2]
            elif isinstance(item, AutonomySpec):
                spec.autonomy = item
        return spec

    def contract_list_field(self, items):
        name = items[0].value
        # items[1] is the list_literal result
        return ('list', name, items[1])

    def autonomy_block(self, items):
        config = {k: v for k, v in items}
        return AutonomySpec(config=config)

    def config_field(self, items):
        return (items[0].value, items[1])

    def list_literal(self, items):
        # items contains STRING tokens
        return [t.value.strip('"') for t in items]

    def value(self, items):
        val = items[0]
        if hasattr(val, 'type') and val.type == 'STRING':
            return val.value.strip('"')
        if hasattr(val, 'type') and val.type == 'NUMBER':
            return float(val.value)
        if isinstance(val, list):
            return val
        return val.value # true/false

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
            elif isinstance(item, WorkflowSpec):
                spec.workflows.append(item)
            elif isinstance(item, tuple) and item[0] == 'invariant':
                spec.invariants.append(item[1])
            elif isinstance(item, tuple) and item[0] == 'constraint':
                spec.constraints.append(item[1])
        return spec
    
    def component_decl(self, items):
        return ComponentSpec(name=items[0].value)
    
    def struct_def(self, items):
        name = items[0].value
        fields = items[1]
        return StructSpec(name=name, fields=fields)

    def effect_block(self, items):
        name = items[0].value
        ops = [i for i in items[1:] if isinstance(i, FunctionSpec)]
        return EffectSpec(name=name, operations=ops)

    def workflow_def(self, items):
        name = items[0].value
        params = items[1]
        steps = [i for i in items[2:] if isinstance(i, str)] # Simplified steps
        return WorkflowSpec(name=name, params=params, steps=steps)

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

    def function_def_body(self, items):
        name = items[0].value
        params = items[1] or []
        ret_type = items[2]
        body = items[3]
        return FunctionSpec(name=name, params=params, return_type=ret_type, body=body)

    def func_body(self, items):
        # logic_expr returns token
        return items[0].value.strip()

    def effect_op(self, items):
        # Maps operation to FunctionSpec for simplicity
        name = items[0].value
        params = items[1] or []
        ret_type = items[2]
        return FunctionSpec(name=name, params=params, return_type=ret_type)

    def workflow_step(self, items):
        return f"Step {items[0].value}: {items[1].value.strip()}"

    def invariant_string(self, items):
        return ('invariant', items[0].value.strip('"'))

    def constraint_string(self, items):
        return ('constraint', items[0].value.strip('"'))

    def field_list(self, items):
        return items

    def param_list(self, items):
        return items

    def field_decl(self, items):
        return Field(name=items[0].value, type=items[1])

    def type_simple(self, items):
        return TypeRef(name=items[0].value)

    def type_list(self, items):
        return TypeRef(name="List", args=items[0])

    def type_map(self, items):
        return TypeRef(name="Map", args=items[0])

    def type_result(self, items):
        return TypeRef(name="Result", args=items[0])

    def generic_args(self, items):
        return [i for i in items if isinstance(i, TypeRef)]
    
    def logic_expr(self, items):
        return items[0]

# --- Compiler API ---

class Compiler:
    def __init__(self):
        self.parser = Lark(AISPEC_GRAMMAR, start='start', parser='lalr')
        self.transformer = AISpecTransformer()

    def compile(self, source_code: str) -> SystemSpec:
        tree = self.parser.parse(source_code)
        return self.transformer.transform(tree)

    def compile_file(self, file_path: str) -> SystemSpec:
        with open(file_path, 'r', encoding='utf-8') as f:
            return self.compile(f.read())
    
    def validate_syntax(self, code: str) -> bool:
        try:
            self.parser.parse(code)
            return True
        except Exception:
            return False
