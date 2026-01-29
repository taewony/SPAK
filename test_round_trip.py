from kernel.mas import MinimalAgentSpec, Intent, MASTask, MASInvariant, SuccessCriteria, MASItem
from kernel.metric_engine import MetricEngine

def test_metrics():
    # 1. Define Original Spec (Gold Standard for DateApp)
    original = MinimalAgentSpec(
        name="DateApp",
        intent=Intent(
            description="Generate a web page that displays a date picker and animates the date display.",
            domain="Web"
        ),
        tasks=[
            MASTask("RenderLayout", "Create HTML structure"),
            MASTask("HandleEvents", "Bind click/change events to state"),
            MASTask("Animate", "Apply CSS animation on update")
        ],
        items=[
            MASItem("DateState", {"value": "Date object"}),
            MASItem("DOMElement", {"tag": "HTML Tag"})
        ],
        invariants=[
            MASInvariant("State updates must trigger re-render"),
            MASInvariant("Animation must complete within 1s")
        ]
    )

    # 2. Simulate Recovered Spec (Slightly noisy, as if from an LLM)
    recovered = MinimalAgentSpec(
        name="DateApp_Recovered",
        intent=Intent(
            description="Create a web interface with date selection and visual feedback animations.", # Semantically similar
            domain="Web"
        ),
        tasks=[
            MASTask("RenderLayout", "Build the DOM tree"), # Match
            MASTask("EventBinding", "Attach listeners"),   # Mismatch name, similar intent
            MASTask("Animate", "Run CSS keyframes")        # Match
        ],
        items=[
            MASItem("Date", {"val": "Date"}),
        ],
        invariants=[
            MASInvariant("UI reflects state changes"), # Semantic match
            MASInvariant("Animation duration < 1000ms")
        ]
    )

    # 3. Calculate Metrics
    engine = MetricEngine()
    metrics = engine.calculate_consistency_index(original, recovered)

    print("🔍 Round-Trip Consistency Test Results")
    print("========================================")
    print(f"Original Intent:  {original.intent.description}")
    print(f"Recovered Intent: {recovered.intent.description}")
    print(f"--> Intent Similarity: {metrics['intent_similarity']:.4f}")
    print("----------------------------------------")
    print(f"Original Tasks: {[t.name for t in original.tasks]}")
    print(f"Recovered Tasks: {[t.name for t in recovered.tasks]}")
    print(f"--> Task Alignment F1: {metrics['task_alignment_f1']:.4f}")
    print("----------------------------------------")
    print(f"--> Invariant Preservation: {metrics['invariant_preservation']:.4f}")
    print("========================================")
    print(f"✅ Final Consistency Index: {metrics['consistency_index']:.4f}")

if __name__ == "__main__":
    test_metrics()
