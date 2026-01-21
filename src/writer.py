from kernel.semantic_kernel import perform
from kernel.effects import Generate, LLMRequest, WriteFile, FileWrite

class Writer:
    def draft_content(self, outline: str) -> str:
        prompt = f"""You are a Technical Writer. Draft the full content based strictly on the provided outline.
        
        INSTRUCTIONS:
        1. Follow the structure of the outline exactly.
        2. Expand each bullet point into full, coherent paragraphs.
        3. Use professional academic tone (unless outline suggests otherwise).
        4. Output in Markdown format.
        
        OUTLINE:
        {outline}
        """
        return perform(Generate(LLMRequest(messages=[{"role": "user", "content": prompt}])))

    def refine_content(self, draft: str, feedback: str) -> str:
        prompt = f"""Refine the following draft based on the Reviewer's feedback.
        
        FEEDBACK:
        {feedback}
        
        CURRENT DRAFT:
        {draft}
        
        INSTRUCTIONS:
        1. Address every point in the feedback.
        2. Improve clarity and flow.
        3. Return the fully updated draft.
        """
        return perform(Generate(LLMRequest(messages=[{"role": "user", "content": prompt}])))

    def finalize_artifact(self, content: str, path: str) -> str:
        perform(WriteFile(FileWrite(path=path, content=content)))
        return f"File written to {path}"
