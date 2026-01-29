system_model SysAdminAgent {
  axiom: "Availability is the highest priority."
  heuristic: "Investigate before taking destructive actions (restart/reboot)."
  prediction: "If logs show 'Out of Memory', a restart is a temporary fix, scaling is permanent."
}

task IncidentResponse {
  step alert: tool.run {
    cmd: "echo 'ALERT: PaymentService is slow (Latency > 2s).'"
    output_var: alert_context
  }

  # --- Inner Loop: Observe -> Plan ---
  step planner: llm.query {
    role: "SRE"
    prompt_template: """
    Context: {{alert_context}}
    
    You have these tools:
    1. `check_logs <service>` (Inspects recent errors)
    2. `check_metrics <service>` (Checks CPU/RAM)
    3. `restart_service <service>` (Fixes hung processes)
    
    Analyze the situation based on Heuristics.
    Decide the NEXT ONE step.
    
    IMPORTANT: Since this is a simulation, return the shell command that SIMULATES the tool's output.
    Example: echo "Logs: NullPointerException in Handler"
    
    Output ONLY the command string.
    """
    output_var: next_tool_cmd
  }

  # --- Inner Loop: Act ---
  step executor: tool.run {
    # The kernel dynamically executes whatever the planner decided
    cmd: "{{next_tool_cmd}}"
    output_var: tool_output
  }

  # --- Inner Loop: Reflect ---
  step analysis: llm.query {
    role: "SRE"
    prompt_template: """
    We executed: {{next_tool_cmd}}
    Result: {{tool_output}}
    
    Determine if the root cause is found or if we need further action.
    """
    output_var: reflection
  }

  evaluation {
    check heuristic_compliance: llm.query {
      role: "Judge"
      prompt_template: """
      Did the agent follow the heuristic: 'Investigate before taking destructive actions'?
      Action taken: {{next_tool_cmd}}
      
      Output 1 if yes (e.g., checked logs/metrics first), 0 if no (e.g., restarted immediately).
      """
      output_var: heuristic_score
    }

    check diagnosis_quality: llm.query {
      role: "Judge"
      prompt_template: """
      Evaluate the final diagnosis: '{{reflection}}'
      Given the tool output: '{{tool_output}}'
      
      Is the diagnosis logical and supported by evidence?
      Output score 0.0 to 1.0.
      """
      output_var: quality_score
    }
  }
}