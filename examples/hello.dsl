task HelloSPAK {
  step s1: tool.run {
    cmd: "echo 'Hello from Shell'"
    output_var: shell_out
  }

  step s2: llm.query {
    role: "greeter"
    prompt_template: "Say hello to: {{shell_out}}"
    output_var: greeting
  }

  step s3: tool.run {
    cmd: "echo 'LLM said: {{greeting}}'"
  }

  success:
    file_exists("trace.json")
}
