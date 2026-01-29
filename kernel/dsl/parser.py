from lark import Lark, Transformer, v_args
import os
from .ast import Task, ToolStep, LLMStep, SuccessCondition, SystemModel, AgentSpec, Metric

# Load grammar relative to this file
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
GRAMMAR_PATH = os.path.join(CURRENT_DIR, "grammar.lark")

with open(GRAMMAR_PATH, "r") as f:
    DSL_GRAMMAR = f.read()

class DSLTransformer(Transformer):
    def start(self, items):
        # Items can be [system_model, task] or [task]
        model = next((i for i in items if isinstance(i, SystemModel)), None)
        task = next((i for i in items if isinstance(i, Task)), None)
        if task and model:
            task.system_model = model
        return AgentSpec(task=task, system_model=model)

    def system_model(self, items):
        name = items[0].value
        stmts = items[1:] 
        
        axioms = [s['content'] for s in stmts if s['type'] == 'axiom']
        heuristics = [s['content'] for s in stmts if s['type'] == 'heuristic']
        predictions = [s['content'] for s in stmts if s['type'] == 'prediction']
        
        return SystemModel(name=name, axioms=axioms, heuristics=heuristics, predictions=predictions)

    def model_stmt(self, items):
        key = items[0].value
        content = items[-1].value.strip('"')
        return {'type': key, 'content': content}

    def task_def(self, items):
        name = items[0].value
        # Steps are direct children of type ToolStep or LLMStep (from step_def)
        steps = [item for item in items if isinstance(item, (ToolStep, LLMStep))]
        
        # Evaluation is a list of Metrics (from evaluation_block)
        evaluation = next((item for item in items if isinstance(item, list) and len(item)>0 and isinstance(item[0], Metric)), [])
        
        # Success criteria is a list of conditions (from success_criteria)
        # Note: We need to be careful not to confuse evaluation list with success list.
        # Success list contains SuccessCondition objects.
        success_criteria = next((item for item in items if isinstance(item, list) and len(item)>0 and isinstance(item[0], SuccessCondition)), [])
        
        return Task(name=name, steps=steps, evaluation=evaluation, success_criteria=success_criteria)

    def step_def(self, items):
        step_id = items[0].value
        step_obj = items[1]
        step_obj.id = step_id
        return step_obj

    def evaluation_block(self, items):
        return items

    def metric_def(self, items):
        metric_id = items[0].value
        logic_step = items[1] # ToolStep or LLMStep
        return Metric(id=metric_id, logic=logic_step)

    def metric_body(self, items):
        return items[0]

    def step_body(self, items):
        return items[0]

    def tool_call(self, items):
        tool_name = items[0].value
        params = items[1]
        output_var = params.pop('output_var', None)
        return ToolStep(id="temp", type="tool", tool_name=tool_name, params=params, output_var=output_var)

    def llm_call(self, items):
        operation = items[0].value 
        params = items[1]
        
        role = params.get('role', 'system')
        prompt_template = params.get('prompt_template', '')
        output_var = params.get('output_var', None)
        
        if 'role' in params: del params['role']
        if 'prompt_template' in params: del params['prompt_template']
        if 'output_var' in params: del params['output_var']
        
        return LLMStep(id="temp", type="llm", role=role, prompt_template=prompt_template, output_var=output_var, config=params)

    def param_list(self, items):
        return {k: v for k, v in items}

    def param_pair(self, items):
        return (items[0].value, items[1])

    def llm_params(self, items):
        return {k: v for k, v in items}

    def llm_param(self, items):
        return (items[0].value, items[1])

    def value(self, items):
        val = items[0]
        if hasattr(val, 'type'):
            if val.type == 'STRING' or val.type == 'MULTILINE_STRING':
                return val.value.strip('"').strip('"""')
            if val.type == 'NAME':
                return val.value
        return val

    def success_criteria(self, items):
        return items[0]

    def condition_block(self, items):
        return items

    def condition(self, items):
        check_type = items[0].value
        args = items[1] if len(items) > 1 else []
        return SuccessCondition(check_type=check_type, args=args)

    def arg_list(self, items):
        return items

class DSLParser:
    def __init__(self):
        self.parser = Lark(DSL_GRAMMAR, start='start', parser='lalr', transformer=DSLTransformer())

    def parse(self, text: str) -> AgentSpec:
        return self.parser.parse(text)

    def parse_file(self, path: str) -> AgentSpec:
        with open(path, "r") as f:
            return self.parse(f.read())