system_model ResearchChef {
  axiom: "No hallucination"
  heuristic: "Check source"
}

task Research {
  step s1: tool.run {
    cmd: "echo test"
  }
}
