from typing import List, Dict
from kernel.semantic_kernel import perform
from kernel.effects import Generate, LLMRequest, Reply, Listen, UserOutput, UserInput, ReasoningTrace, TraceLog

class Coach:
    def __init__(self):
        self.state = {
            "goal": "General Improvement",
            "history": [],
            "user_feeling": ""
        }

    def configure_session(self, goal: str) -> str:
        """Sets the predefined coaching session goal."""
        self.state["goal"] = goal
        return f"Session configured with goal: '{goal}'"

    def start_session(self, feeling: str = None) -> str:
        """Starts the interactive multi-turn coaching session."""
        
        # 1. Generate Dynamic Greeting based on Goal
        perform(ReasoningTrace(TraceLog(
            thought=f"I need to greet the user and establish context for goal: {self.state['goal']}",
            plan={"action": "start_session", "sub_task": "greeting"}
        )))
        
        prompt = f"""
        You are an expert coach. The user's goal is: "{self.state['goal']}".
        Generate a warm, motivating greeting and ask an opening question to gauge their current state.
        """
        greeting = perform(Generate(LLMRequest(messages=[{"role": "user", "content": prompt}])))
        
        print(f"\n[Coach]: {greeting}")
        self.state["history"].append(f"Coach: {greeting}")
        
        # Initial input handling
        if feeling:
            user_input = feeling
            print(f"[User]: {user_input}")
        else:
            user_input = perform(Listen(UserInput(prompt="[User]: ")))

        # 2. Continuous Loop
        while True:
            if user_input.lower() in ["quit", "exit", "quit()", "exit()"]:
                print("\n[Coach]: Goodbye! Keep working on your goal.")
                break

            self.state["history"].append(f"User: {user_input}")
            
            perform(ReasoningTrace(TraceLog(
                thought=f"User said '{user_input}'. I need to guide them towards '{self.state['goal']}'.",
                plan={"action": "start_session", "sub_task": "loop_response"}
            )))
            
            # Construct Prompt with History
            history_text = "\n".join(self.state["history"][-10:]) # Keep context
            prompt = f"""
            You are helping the user achieve: "{self.state['goal']}".
            
            Conversation History:
            {history_text}
            
            Respond to the user naturally. Guide them towards the goal.
            """
            
            response = perform(Generate(LLMRequest(messages=[{"role": "user", "content": prompt}])))
            
            print(f"\n[Coach]: {response}")
            self.state["history"].append(f"Coach: {response}")
            
            user_input = perform(Listen(UserInput(prompt="\n[User]: ")))
            
        return "Session ended."

    def provide_feedback(self, performance_score: int = None) -> str:
        # Legacy/Simple mode
        if performance_score is not None:
            if performance_score >= 85:
                return "Great job! Keep up the good work!"
            elif performance_score < 40:
                return "It looks like you need some extra practice."
            else:
                return "Good effort, but there is room for improvement."
        
        # Gap Analysis Mode (No argument)
        if not self.state["history"]:
            return "No session history available for analysis."
            
        perform(ReasoningTrace(TraceLog(
            thought="Analyzing session history to identify gaps between current state and goal.",
            plan={"action": "provide_feedback"}
        )))
            
        history_text = "\n".join(self.state["history"])
        prompt = f"""
        Analyze the coaching session history below regarding the goal: "{self.state['goal']}".
        
        Session History:
        {history_text}
        
        Provide a Gap Analysis:
        1. Current State vs Goal State
        2. Key obstacles identified
        3. Recommendations for closing the gap
        """
        print(f"📊 [Coach] Analyzing session gap for goal: '{self.state['goal']}'...")
        response = perform(Generate(LLMRequest(messages=[{"role": "user", "content": prompt}])))
        return response

    def generate_training_plan(self, current_level: str = None, target_level: str = None) -> str:
        # 1. Context-Aware Plan (No arguments)
        if current_level is None and target_level is None:
            if not self.state["history"]:
                return "No session history to base the plan on. Please start a session first."
            
            perform(ReasoningTrace(TraceLog(
                thought="Generating a concrete training plan based on the session insights.",
                plan={"action": "generate_training_plan"}
            )))
            
            history_text = "\n".join(self.state["history"])
            prompt = f"""
            Based on the coaching session below, generate a concrete Training Plan to achieve the goal: "{self.state['goal']}".
            
            Session Context:
            {history_text}
            
            Format the plan with clear steps and milestones.
            """
            print(f"📋 [Coach] Generating context-aware plan for goal: '{self.state['goal']}'...")
            return perform(Generate(LLMRequest(messages=[{"role": "user", "content": prompt}])))

        # 2. Legacy/Explicit Plan (Arguments provided)
        user_msg = f"Coach, I am currently at the {current_level} level and my goal is to reach the {target_level} level. Provide a plan starting with 'Step'."
        response = perform(Generate(LLMRequest(messages=[{"role": "user", "content": user_msg}])))
        return response