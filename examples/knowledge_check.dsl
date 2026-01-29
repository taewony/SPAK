system_model KnowledgeChef {
  axiom: "Every section must have at least one factual citation."
  axiom: "No unsupported conclusions."
  heuristic: "Synthesize narrative by Topic first, then by Validated Claims."
  prediction: "Without semantic labeling, context might be lost."
}

task KnowledgePipeline {
  
  step load_corpus: tool.run {
    # Simulating reading MD files from a directory
    cmd: "echo 'Doc1: AI agents are autonomous systems that perceive and act. Doc2: To execute safely, agents require a deterministic kernel to validate actions.'"
    output_var: raw_corpus
  }

  step semantic_analysis: llm.query {
    role: "ForensicAnalyst"
    prompt_template: """
    Analyze the corpus: {{raw_corpus}}
    1. Segment into paragraphs.
    2. Assign keywords and topic labels to each segment.
    """
    output_var: labeled_segments
  }

  step claim_extraction: llm.query {
    role: "ForensicAnalyst"
    prompt_template: """
    From segments: {{labeled_segments}}
    Extract Claims (Factual, Causal, Conclusion).
    """
    output_var: candidate_claims
  }

  step evidence_validation: llm.query {
    role: "ForensicAnalyst"
    prompt_template: """
    Validate claims: {{candidate_claims}}
    Against corpus: {{raw_corpus}}
    
    Rule: Discard claim if Evidence < threshold (High confidence only).
    Output list of Validated Claims with Citations.
    """
    output_var: validated_claims
  }

  step synthesis: llm.query {
    role: "ChiefEditor"
    prompt_template: """
    Write a narrative synthesis based on: {{validated_claims}}
    Structure: Claim -> Explanation -> Evidence -> Conclusion.
    """
    output_var: final_document
  }

  step save_result: tool.run {
    cmd: "echo 'Synthesis Complete. Doc: {{final_document}}'"
  }
}