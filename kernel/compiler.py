from lark import Lark, Transformer, v_args, Token
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

# --- Grammar Definition ---
AISPEC_GRAMMAR = r"""
start: meta_def? contract_def? system_model? system_def

meta_def: "meta" "{" meta_field* "}"

meta_field: NAME "=" STRING

system_def: "system" NAME "{" (component_def | page_def)* "}"
          | "kernel" NAME "{" component_def* "}"

system_model: "system_model" NAME "{" model_stmt* "}"

model_stmt: "axiom" ":" STRING ";"?        -> axiom_stmt
          | "heuristic" ":" STRING ";"?    -> heuristic_stmt
          | "prediction" ":" STRING ";"?   -> prediction_stmt

contract_def: "contract" NAME "{" contract_member* "}"

contract_member:
    | NAME "=" list_literal                -> contract_list_field
    | "autonomy" "{" config_field* "}"     -> autonomy_block

autonomy_block: "autonomy" "{" config_field* "}"

config_field: NAME "=" value

list_literal: "[" (STRING ("," STRING)*)? "]"

value: STRING | NUMBER | "true" | "false" | list_literal

# --- Web / App DSL Extensions ---

page_def: "page" NAME "{" page_body "}"

page_body: (member_def | style_block | component_inst)*

style_block: "style" css_selector "{" css_decl* "}"
css_selector: /[^{]+/
css_decl: /[^{}]+;/ 

component_inst: NAME ( "bind" NAME )?

template_block: "template" ":" html_dsl
html_dsl: /[^{}]+(\{[^{}]*\})*/   // Simple nested brace matching (limit 1 level for now)

transition_def: "transition" NAME "(" param_list ")" ":" func_body 
              | "transition" NAME ":" func_body

on_update_def: "on" "update" "(" param_list ")" ":" func_body

# --- GPU DSL Extensions ---

gpu_kernel_def: "kernel" NAME "{" gpu_member* "}"

gpu_member: "threads" ":" tuple_literal     -> gpu_threads
          | "blocks" ":" tuple_literal      -> gpu_blocks
          | "shared_mem" NAME type_ref      -> gpu_shared_mem
          | "fusion" ":" list_literal       -> gpu_fusion
          | function_def                    -> gpu_function

tuple_literal: "(" NUMBER ("," NUMBER)* ")"

# --- End DSL Extensions ---

component_def: 
    | "component" NAME "{" member_def* "}" -> component_block
    | "component" NAME                     -> component_decl
    | "effect" NAME "{" effect_op* "}"     -> effect_block
    | "workflow" NAME "(" param_list ")" "{" workflow_step* "}" -> workflow_def
    | "struct" NAME "{" field_list "}"     -> struct_def
    | "import" NAME                        -> import_stmt
    | gpu_kernel_def                       -> gpu_kernel_def

member_def:
    | "description" ":" STRING ";"?         -> description
    | "state" NAME "{" field_list "}"      -> state_def
    | "state" NAME ":" type_ref            -> state_simple_def
    | "function" NAME "(" param_list ")" "->" type_ref ";"? -> function_def
    | "function" NAME "(" param_list ")" "->" type_ref func_body -> function_def_body
    | "invariant" ":" STRING ";"?           -> invariant_string
    | "constraint" ":" STRING ";"?          -> constraint_string
    | "workflow" NAME "(" param_list ")" "{" workflow_step* "}" -> workflow_def
    | "utils" "{" util_func* "}"           -> utils_block
    # Web DSL members
    | "head" ":" style_block*              -> head_block
    | "body" ":" component_inst*           -> body_block
    | template_block
    | transition_def
    | on_update_def

util_func: "function" NAME "(" param_list ")" func_body

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
        | type_array                    -> type_array

type_array: NAME array_dims
array_dims: ("[" NUMBER "]")+

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
    dims: List[int] = field(default_factory=list)

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
    type: Optional[TypeRef] = None

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
class ModelStatement:
    type: str # axiom, heuristic, prediction
    content: str

@dataclass
class SystemModelSpec:
    name: str
    statements: List[ModelStatement] = field(default_factory=list)

@dataclass
class StructSpec:
    name: str
    fields: List[Field]

# --- Web DSL AST ---
@dataclass
class StyleSpec:
    selector: str
    declarations: List[str]

@dataclass
class TemplateSpec:
    content: str

@dataclass
class TransitionSpec:
    name: str
    params: List[Field]
    body: str

@dataclass
class PageSpec:
    name: str
    styles: List[StyleSpec] = field(default_factory=list)
    body_components: List[str] = field(default_factory=list)

# --- GPU DSL AST ---
@dataclass
class GPUKernelSpec:
    name: str
    threads: tuple = (1, 1, 1)
    blocks: tuple = (1, 1, 1)
    shared_mem: Dict[str, TypeRef] = field(default_factory=dict)
    fusion_ops: List[str] = field(default_factory=list)
    functions: List[FunctionSpec] = field(default_factory=list)

@dataclass
class ComponentSpec:
    name: str
    description: str = ""
    states: List[StateSpec] = field(default_factory=list)
    functions: List[FunctionSpec] = field(default_factory=list)
    invariants: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    workflows: List[WorkflowSpec] = field(default_factory=list)
    # Web DSL
    template: Optional[TemplateSpec] = None
    transitions: List[TransitionSpec] = field(default_factory=list)
    styles: List[StyleSpec] = field(default_factory=list)

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
    system_model: Optional[SystemModelSpec] = None
    components: List[ComponentSpec] = field(default_factory=list)
    pages: List[PageSpec] = field(default_factory=list)
    structs: List[StructSpec] = field(default_factory=list)
    effects: List[EffectSpec] = field(default_factory=list)
    workflows: List[WorkflowSpec] = field(default_factory=list)
    gpu_kernels: List[GPUKernelSpec] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    contract: Optional[ContractSpec] = None

# --- Transformer ---

class AISpecTransformer(Transformer):
    def start(self, items):
        system = next((i for i in items if isinstance(i, SystemSpec)), None)
        meta = next((i for i in items if isinstance(i, dict)), None)
        contract = next((i for i in items if isinstance(i, ContractSpec)), None)
        model = next((i for i in items if isinstance(i, SystemModelSpec)), None)
        
        if system:
            if meta:
                system.metadata = meta
            if contract:
                system.contract = contract
            if model:
                system.system_model = model
        return system

    def meta_def(self, items):
        return {k: v for k, v in items}

    def meta_field(self, items):
        return (items[0].value, items[1].value.strip('"'))

    def system_def(self, items):
        name = items[0].value
        components = [i for i in items[1:] if isinstance(i, ComponentSpec)]
        pages = [i for i in items[1:] if isinstance(i, PageSpec)]
        structs = [i for i in items[1:] if isinstance(i, StructSpec)]
        effects = [i for i in items[1:] if isinstance(i, EffectSpec)]
        workflows = [i for i in items[1:] if isinstance(i, WorkflowSpec)]
        gpu_kernels = [i for i in items[1:] if isinstance(i, GPUKernelSpec)]
        imports = [i for i in items[1:] if isinstance(i, str)]
        return SystemSpec(name=name, components=components, pages=pages, structs=structs, effects=effects, workflows=workflows, gpu_kernels=gpu_kernels, imports=imports)

    def system_model(self, items):
        name = items[0].value
        statements = [i for i in items[1:] if isinstance(i, ModelStatement)]
        return SystemModelSpec(name=name, statements=statements)

    def axiom_stmt(self, items):
        return ModelStatement(type="axiom", content=items[0].value.strip('"'))

    def heuristic_stmt(self, items):
        return ModelStatement(type="heuristic", content=items[0].value.strip('"'))

    def prediction_stmt(self, items):
        return ModelStatement(type="prediction", content=items[0].value.strip('"'))

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
        return ('list', name, items[1])

    def autonomy_block(self, items):
        config = {k: v for k, v in items}
        return AutonomySpec(config=config)

    def config_field(self, items):
        return (items[0].value, items[1])

    def list_literal(self, items):
        return [t.value.strip('"') for t in items]

    def value(self, items):
        val = items[0]
        if hasattr(val, 'type') and val.type == 'STRING':
            return val.value.strip('"')
        if hasattr(val, 'type') and val.type == 'NUMBER':
            return float(val.value)
        if isinstance(val, list):
            return val
        return val.value 

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
            elif isinstance(item, TemplateSpec):
                spec.template = item
            elif isinstance(item, TransitionSpec):
                spec.transitions.append(item)
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
        steps = [i for i in items[2:] if isinstance(i, str)] 
        return WorkflowSpec(name=name, params=params, steps=steps)

    def import_stmt(self, items):
        return items[0].value

    def description(self, items):
        return ('desc', items[0].value.strip('"'))

    def state_def(self, items):
        name = items[0].value
        fields = items[1]
        return StateSpec(name=name, fields=fields)

    def state_simple_def(self, items):
        name = items[0].value
        type_ref = items[1]
        return StateSpec(name=name, fields=[], type=type_ref)

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
        return items[0].value.strip()

    def effect_op(self, items):
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

    # --- Web DSL Methods ---
    def page_def(self, items):
        name = items[0].value
        body_items = items[1]
        styles = [i for i in body_items if isinstance(i, StyleSpec)]
        comps = [i for i in body_items if isinstance(i, str)]
        return PageSpec(name=name, styles=styles, body_components=comps)

    def page_body(self, items):
        return items

    def head_block(self, items):
        return items 

    def body_block(self, items):
        return items

    def style_block(self, items):
        selector = items[0].value.strip()
        decls = [d.value.strip() for d in items[1:]]
        return StyleSpec(selector=selector, declarations=decls)

    def component_inst(self, items):
        return items[0].value

    def template_block(self, items):
        return TemplateSpec(content=items[0].value)

    def transition_def(self, items):
        name = items[0].value
        if len(items) == 2:
            return TransitionSpec(name=name, params=[], body=items[1].value)
        else:
            return TransitionSpec(name=name, params=items[1], body=items[2].value)

    def on_update_def(self, items):
        # Maps on update to a special transition for now
        return TransitionSpec(name="on_update", params=items[0], body=items[1].value)

    # --- GPU DSL Methods ---
    def gpu_kernel_def(self, items):
        name = items[0].value
        spec = GPUKernelSpec(name=name)
        for item in items[1:]:
            if isinstance(item, tuple) and item[0] == 'threads':
                spec.threads = item[1]
            elif isinstance(item, tuple) and item[0] == 'blocks':
                spec.blocks = item[1]
            elif isinstance(item, tuple) and item[0] == 'shared':
                spec.shared_mem[item[1]] = item[2]
            elif isinstance(item, tuple) and item[0] == 'fusion':
                spec.fusion_ops = item[1]
            elif isinstance(item, FunctionSpec):
                spec.functions.append(item)
        return spec

    def gpu_threads(self, items):
        return ('threads', items[0])

    def gpu_blocks(self, items):
        return ('blocks', items[0])
    
    def gpu_shared_mem(self, items):
        return ('shared', items[0].value, items[1])
    
    def gpu_fusion(self, items):
        return ('fusion', items[0])

    def tuple_literal(self, items):
        return tuple(float(t.value) for t in items)

    def type_array(self, items):
        name = items[0].value
        dims = items[1]
        return TypeRef(name=name, dims=dims)
    
    def array_dims(self, items):
        return [int(t.value) for t in items]

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