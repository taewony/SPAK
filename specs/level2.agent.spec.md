meta {
    name = "CalculatorAgent"
    version = "1.0"
    domain = "Computational Tool Use"
    purpose = "To perform precise mathematical calculations using external tools."
    description = "Level 2: Agent capable of using Math Tools via Effects"
}

// --- Operational Contract ---
contract AgentScope {
    supported_intents = [
        "Math Operations: performing arithmetic (add, sub, mul, div)."
    ]

    success_criteria = [
        "Accuracy: Results must be mathematically correct.",
        "Tool Usage: Must correctly invoke the Math effect."
    ]
}

system CalculatorAgent {
    effect Math {
        operation add(a: Float, b: Float) -> Float;
        operation sub(a: Float, b: Float) -> Float;
        operation mul(a: Float, b: Float) -> Float;
        operation div(a: Float, b: Float) -> Float;
    }

    component Solver {
        description: "Solves simple math problems using the Math effect.";

        function calculate(a: Float, b: Float, op: String) -> Float {
            perform Math.add(a, b)
        }
    }
}
