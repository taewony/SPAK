from typing import List
import os
from kernel.semantic_kernel import perform
from kernel.effects import ListFiles, FileList, ReadFile, FileRead, GrepFiles, GrepRequest

class Librarian:
    def gather_materials(self, source_path: str, topic_keywords: str) -> List[str]:
        """
        Uses Hybrid Tooling (Grep) to quickly find relevant documents.
        """
        if not topic_keywords:
            return []
            
        # Strategy: Use the first keyword as the primary fast-path filter.
        # In a real system, we might run multiple greps or a complex regex.
        keywords = [k.strip() for k in topic_keywords.split(",")]
        primary_keyword = keywords[0] 
        
        print(f"📚 [Librarian] Fast-searching for '{primary_keyword}' in {source_path}...")
        
        # 1. Fast Path: Grep
        relevant_files = perform(GrepFiles(GrepRequest(
            pattern=primary_keyword,
            dir_path=source_path,
            file_pattern="*.md" 
        )))
        
        print(f"    Found {len(relevant_files)} candidate files.")
        
        # 2. Ingestion
        docs = []
        for file_path in relevant_files:
            try:
                # We assume file_path returned by grep is relative or absolute correctly.
                # If findstr returns relative, we might need to join with source_path?
                # findstr /s returns relative path from where it was run?
                # My handler implementation: `findstr ... dir_path\pattern`
                # If dir_path is absolute, output is absolute.
                
                content = perform(ReadFile(FileRead(path=file_path)))
                docs.append(content)
            except Exception as e:
                print(f"    Failed to read {file_path}: {e}")
                continue
        
        return docs
