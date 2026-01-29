system_model CoachingAgent {
  axiom: "Every roadmap item must map to an identified gap."
  axiom: "Roadmap must respect user constraints if provided."
  heuristic: "Gap Analysis must be categorized into: Knowledge, Skills, and Resources."
  prediction: "If the roadmap lacks timelines, the user will fail to execute it."
}

task CoachingWorkflow {
  # --- Phase 1: Context Gathering ---
  step init: tool.run {
    cmd: "echo 'Goal: Learn to build AI Agents. Constraints: 2 hours/week.'"
    output_var: user_goal_and_constraints
  }

  step gather_user_skills: tool.run {
    # Simulating the user's input for this session
    cmd: "echo 'I know Python intermediate. I have never used LLM APIs. I know Git.'"
    output_var: user_skills
  }

  # --- Phase 2: Analysis (The Brain) ---
  step diagnosis: llm.query {
    role: "Coach"
    prompt_template: """
    Analyze the user context to identify gaps.
    User Goal & Constraints: {{user_goal_and_constraints}}
    User Current Skills: {{user_skills}}
    
    Output a structured Gap Analysis (Knowledge, Skills, Resources).
    """
    output_var: gap_analysis
  }

  # --- Phase 3: Planning (The Architect) ---
  step planning: llm.query {
    role: "Coach"
    prompt_template: """
    Based on this Gap Analysis:
    {{gap_analysis}}
    
    Create a detailed Execution Roadmap.
    1. Break down into Phases (Weeks).
    2. Each action must address a specific gap.
    3. Respect the constraint: {{user_goal_and_constraints}}
    """
    output_var: detailed_roadmap
  }

  # --- Phase 4: Closing (The Delivery) ---
  step closing: llm.query {
    role: "Coach"
    prompt_template: """
    You are closing the session. Present the final plan to the user.
    
    Gap Analysis Summary:
    {{gap_analysis}}
    
    Your Roadmap:
    {{detailed_roadmap}}
    
    End with a motivating closing statement and one immediate 'next step' action.
    """
    output_var: final_message
  }

  step save_output: tool.run {
    cmd: "echo 'Session Closed. Final Output: {{final_message}}'"
  }
}
