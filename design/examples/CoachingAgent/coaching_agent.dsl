system CoachingAgent {

  agents {
    primary Coach persona=CoachPersona
  }

  context {
    user_goal
    conversation_history
    user_constraints optional
  }

  goals {
    clarify goal
    diagnose gaps
    propose executable plan
  }

  workflow {

    step session_start:
      Coach.activate persona
      Coach.confirm understanding of user_goal

    step dialogue:
      Coach.ask about:
        - current skills
        - past attempts
        - constraints
      loop until user signals session_end

    step diagnosis:
      Coach.derive current_state
      Coach.compute gaps:
        skill_gaps
        knowledge_gaps
        resource_gaps

    step planning:
      Coach.create roadmap with:
        milestones
        timelines
        recommended actions

    step finalize:
      emit gap_analysis
      emit roadmap
  }

  artifacts {
    gap_analysis {
      categories: [skill, knowledge, resource]
    }

    roadmap {
      must_include: [milestones, timeline, actions]
    }
  }

  checks {
    every roadmap_item maps_to a gap
    roadmap respects user_constraints if provided
  }
}
