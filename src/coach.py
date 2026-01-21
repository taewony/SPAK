from typing import List, Dict
from kernel.semantic_kernel import perform
from kernel.effects import Generate, LLMRequest, Reply, Listen, UserOutput, UserInput, ReasoningTrace, TraceLog

class Coach:
    def __init__(self, goal: str = "becomming a CEO of AI-Powered Content Creation one-person company"):
        self.state = {
            "goal": goal,
            "plan": "Initial engagement and rapport building.",
            "history": [],
            "user_feeling": ""
        }

    def start(self, feeling: str = None) -> str:
        """Starts the interactive multi-turn coaching session."""
        
        # 1. Think & Plan Opening
        perform(ReasoningTrace(TraceLog(
            thought=f"User wants to achieve: {self.state['goal']}. I need to establish a baseline.",
            plan={"action": "init", "strategy": "rapport_building"}
        )))
        
        prompt = f"""
        You are an expert coach. The user's goal is: "{self.state['goal']}".
        Generate a warm, motivating greeting and ask an opening question.
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
            if user_input.lower() in ["quit", "exit", "quit()", "exit()", "end", "end()"]:
                print("\n[Coach]: Ending session and preparing report...")
                
                # Execute end() logic to generate Gap Analysis & Roadmap
                end_report = self.end()
                print(end_report)
                
                return "Session ended gracefully with report."

            self.state["history"].append(f"User: {user_input}")
            
            # --- THINK STEP ---
            # Analyze if the current plan needs revision based on user input
            thought_prompt = f"""
            Goal: {self.state['goal']}
            Current Plan: {self.state['plan']}
            Last User Input: {user_input}
            
            Analyze if the plan needs revision. Should we stick to the plan or shift focus?
            Respond with:
            THOUGHT: [Reasoning]
            REVISION: [New Plan or 'None']
            """
            analysis = perform(Generate(LLMRequest(messages=[{"role": "user", "content": thought_prompt}])))
            
            # Simple parsing for logging
            thought = "Analyzing user input to refine coaching strategy."
            new_plan = "None"
            if "THOUGHT:" in analysis:
                thought = analysis.split("THOUGHT:")[1].split("REVISION:")[0].strip()
            if "REVISION:" in analysis:
                new_plan = analysis.split("REVISION:")[1].strip()

            # --- PLAN/REVISE STEP ---
            if new_plan.lower() != "none":
                old_plan = self.state["plan"]
                self.state["plan"] = new_plan
                perform(ReasoningTrace(TraceLog(
                    thought=thought,
                    plan={"action": "revise", "old_plan": old_plan, "new_plan": new_plan}
                )))
            else:
                perform(ReasoningTrace(TraceLog(
                    thought=thought,
                    plan={"action": "continue", "current_plan": self.state["plan"]}
                )))
            
            # --- EXECUTE STEP ---
            history_text = "\n".join(self.state["history"][-10:])
            prompt = f"""
            Goal: {self.state['goal']}
            Active Strategy: {self.state['plan']}
            History: {history_text}
            
            Respond to the user.
            """
            
            response = perform(Generate(LLMRequest(messages=[{"role": "user", "content": prompt}])))
            
            print(f"\n[Coach]: {response}")
            self.state["history"].append(f"Coach: {response}")
            
            user_input = perform(Listen(UserInput(prompt="\n[User]: ")))
            
        return "Session loop finished. Run 'end()' to generate report."

    def end(self) -> str:
        """Ends the session, performs gap analysis, and generates a roadmap."""
        if not self.state["history"]:
            return "No session history available for analysis."
        
        history_text = "\n".join(self.state["history"])
        
        # 1. Gap Analysis
        perform(ReasoningTrace(TraceLog(
            thought="Performing gap analysis on the session history.",
            plan={"action": "end", "sub_task": "gap_analysis"}
        )))
        
        prompt_gap = f"""
        Analyze the coaching session history below regarding the goal: "{self.state['goal']}".
        
        Session History:
        {history_text}
        
        Provide a Gap Analysis:
        1. Current State vs Goal State
        2. Key obstacles identified
        """
        print(f"📊 [Coach] Running Gap Analysis for goal: '{self.state['goal']}'...")
        gap_analysis = perform(Generate(LLMRequest(messages=[{"role": "user", "content": prompt_gap}])))
        print(f"\n[Gap Analysis]:\n{gap_analysis}")

        # 2. Roadmap
        perform(ReasoningTrace(TraceLog(
            thought="Generating a strategic roadmap to bridge the identified gap.",
            plan={"action": "end", "sub_task": "roadmap"}
        )))
        
        prompt_roadmap = f"""
        Based on the previous analysis, generate a concrete Roadmap to achieve the goal: "{self.state['goal']}".
        
        Context:
        {gap_analysis}
        
        Format as a step-by-step roadmap.
        """
        print(f"🗺️ [Coach] Generating Roadmap...")
        roadmap = perform(Generate(LLMRequest(messages=[{"role": "user", "content": prompt_roadmap}])))
        
        return f"\n[Roadmap]:\n{roadmap}"
