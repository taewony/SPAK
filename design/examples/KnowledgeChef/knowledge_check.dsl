// knowledge engineering pipeline: 
//  문서 ETL > semantic labeling > claim extraction > evidence alignment > narrative synthesis

system KnowledgeChef {

  agents {
    primary ChiefEditor persona=ForensicAnalyst
  }

  input {
    source_dir : Path
    synthesis_intent optional
  }

  data_models {
    Paragraph { id, text, source_file }
    Claim { id, text, type }
    Evidence { paragraph_id, relevance_score }
  }

  workflow {

    step corpus_building:
      read all "*.md" from source_dir
      segment into Paragraph units

    step semantic_tagging:
      for each Paragraph:
        assign keywords
        assign topic labels

    step claim_detection:
      for each Paragraph:
        extract Claims of type:
          - factual
          - causal
          - conclusion

    step evidence_retrieval:
      for each Claim:
        rank Paragraphs by factual support
        attach Evidence links

    step validation:
      discard Claim if Evidence < threshold

    step document_planning:
      build outline:
        by topic
        then by validated claims

    step composition:
      write sections:
        claim → explanation → cited evidence → conclusion

    step consolidation:
      merge into single narrative document

    step export:
      emit synthesized_document
      emit traceability_index
  }

  artifacts {
    synthesized_document
    traceability_index
  }

  checks {
    every section has >=1 factual citation
    no unsupported conclusions
  }

}
