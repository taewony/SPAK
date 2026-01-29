system_model MetaOptimizer {
  axiom: "To improve an agent, align its Prompt with the System Axioms."
  heuristic: "If a safety metric fails, the Planner step likely lacks explicit constraints."
}

task EvolveAgent {
  
  # 1. Observe: Read the past execution
  step read_trace: tool.read {
    path: "trace.json"
    output_var: past_trace
  }
  
  step read_metrics: tool.read {
    path: "metrics.json"
    output_var: past_metrics
  }

  step read_dsl: tool.read {
    path: "sysadmin_v1.dsl"
    output_var: current_dsl_code
  }

  # 2. Abductive Reasoning: Why did it fail?
  step diagnose_root_cause: llm.query {
    role: "SystemArchitect"
    prompt_template: """
    Analyze the Agent Performance.
    
    Goal: High Safety Score (checking logs before acting).
    Result (Metrics): {{past_metrics}}
    Execution Trace: {{past_trace}}
    Current DSL:
    {{current_dsl_code}}
    
    Perform Abductive Reasoning:
    1. Observe the gap between Goal and Result.
    2. Hypothesize the root cause in the DSL (e.g., Is the prompt too vague?).
    
    Output your reasoning.
    """
    output_var: diagnosis
  }

  # 3. Refine: Generate Improved DSL
  step generate_patch: llm.query {
    role: "SystemArchitect"
    prompt_template: """
    Based on the diagnosis: {{diagnosis}}
    
    Rewrite the 'sysadmin_v1.dsl' code to fix the issue.
    
    Rules:
    1. Keep the same structure.
    2. MODIFY the 'prompt_template' in the 'planner' step to explicitly enforce safety (Checking Logs).
    3. Do not change tool definitions.
    
    Output the FULL valid DSL content.
    """
    output_var: improved_dsl_code
  }

  # 4. Deploy: Save v2
  step deploy_v2: tool.write {
    path: "sysadmin_v2.dsl"
    content: "{{improved_dsl_code}}"
  }
}
