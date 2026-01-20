from typing import List
import os
from kernel.semantic_kernel import perform
from kernel.effects import ListFiles, FileList, ReadFile, FileRead

class Librarian:
    def gather_materials(self, source_path: str, topic_keywords: str) -> List[str]:
        # 1. List all files
        # Note: In a real scenario, source_path should be absolute or relative to CWD.
        all_files = perform(ListFiles(FileList(dir_path=source_path)))
        
        relevant_docs = []
        # Handle case where topic_keywords might be empty or None
        keywords = [k.strip().lower() for k in topic_keywords.split(",")] if topic_keywords else []
        
        # 2. Simple filter and read
        for file_path in all_files:
            if file_path.endswith(".md") or file_path.endswith(".txt"):
                try:
                    content = perform(ReadFile(FileRead(path=file_path)))
                    # Check if any keyword is in content
                    if not keywords or any(k in content.lower() for k in keywords):
                        relevant_docs.append(content)
                except Exception:
                    continue # Skip unreadable files
        
        return relevant_docs
