# Recursive-LLM Reverse Analysis

## 1. Core Architecture

The `recursive-llm` system is an implementation of **Recursive Language Models (RLM)** designed to handle unbounded context lengths by treating context as data stored in a variable rather than embedding it entirely in the prompt.

### 1.1 The "Context as Variable" Pattern
Instead of:
```python
prompt = f"Summarize this: {huge_document}"
llm.complete(prompt)
```
RLM uses:
```python
context = huge_document # Stored in Python memory
prompt = "Use the `context` variable to answer the query."
# LLM writes code to inspect `context`
```

### 1.2 Key Components

1.  **`RLM` Class (`core.py`)**
    - **State**:
        - `context`: The large string (document).
        - `messages`: Conversation history.
        - `depth`: Recursion depth tracker.
    - **Control Flow**:
        - Loop `max_iterations` times.
        - Call LLM -> Get Code or `FINAL(answer)`.
        - If Code -> Execute in REPL -> Add result to history -> Repeat.
        - If `FINAL` -> Return answer.

2.  **`REPLExecutor` (`repl.py`)**
    - **Sandbox**: Uses `RestrictedPython` to execute generated code safely.
    - **Output Management**: Captures `stdout` and truncates it to avoid prompt bloat.
    - **Environment**: Injects variables (`context`, `query`) and tools (`re`, `recursive_llm`) into the scope.

3.  **Recursion Mechanism**
    - The REPL environment contains a `recursive_llm(sub_query, sub_context)` function.
    - When the LLM calls this, it creates a *new* `RLM` instance with `depth + 1`.
    - This enables a "Divide and Conquer" strategy for large texts.

## 2. Execution Flow (Trace)

1.  **Initialization**: `rlm = RLM(model="gpt-4", max_depth=5)`
2.  **User Query**: `rlm.completion("Find all dates", context=huge_log_file)`
3.  **Iteration 1**:
    - **System Prompt**: "You are RLM... Context size: 10MB... Write code."
    - **LLM**: `print(context[:200])`
    - **REPL**: (Executes and returns first 200 chars)
    - **Observation**: "2023-01-01 Error..."
4.  **Iteration 2**:
    - **LLM**: (Realizes file is too big) `mid = len(context)//2; recursive_llm("Find dates", context[:mid])`
    - **Sub-Agent**: (Starts new RLM process on first half)
    - ... (Sub-agent completes and returns list of dates) ...
5.  **Iteration N**:
    - **LLM**: `FINAL("Found 50 dates...")`

## 3. Design Elements for Generic Build-Agent

From this analysis, we extract the following reusable patterns for our generic system:

### 3.1 The "Thought-Act-Observe" Loop
The core loop is generic:
- **Thought**: LLM decides what to do based on history.
- **Act**: Execution of a capability (Tool/Effect).
- **Observe**: Feedback from the environment.

### 3.2 Dynamic Tool Injection
`recursive-llm` dynamically injects the `recursive_llm` function into the REPL. Our system should generalize this to inject *any* defined capability (FileSystem, Shell, etc.) into the execution scope.

### 3.3 State Encapsulation
The `RLM` object holds the specific state for one task. This maps perfectly to our **Value Semantics** approach where an Agent is an object wrapping a State.

### 3.4 Depth Control
Recursion must be guarded. `max_depth` and `max_iterations` are critical "Safety Invariants" that we will formalize in our `AISpec`.

## 4. Limitation Analysis
- **Sequential Execution**: The current RLM is sequential. Our system should support async/parallel execution (e.g., `await asyncio.gather(sub_agent1, sub_agent2)`).
- **Hardcoded Tools**: RLM hardcodes `re` and `recursive_llm`. We need a `ToolRegistry`.
- **Implicit Protocol**: The "FINAL()" protocol is fragile. We should use structured outputs (JSON) or explicit Effect objects.
