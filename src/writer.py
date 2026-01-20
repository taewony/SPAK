from kernel.semantic_kernel import perform
from kernel.effects import Generate, LLMRequest, WriteFile, FileWrite

class Writer:
    def draft_content(self, outline: str, format_style: str) -> str:
        prompt = f"""Draft a full document based on the outline, following the format style.
        
        OUTLINE:
        {outline}
        
        STYLE:
        {format_style}
        """
        return perform(Generate(LLMRequest(messages=[{"role": "user", "content": prompt}])))

    def finalize_artifact(self, draft: str, file_path: str) -> str:
        perform(WriteFile(FileWrite(path=file_path, content=draft)))
        return f"File written to {file_path}"
