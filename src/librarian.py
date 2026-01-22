from typing import List
import os
from kernel.semantic_kernel import perform
from kernel.effects import ListFiles, FileList, ReadFile, FileRead, GrepFiles, GrepRequest

class Librarian:
    def gather_materials(self, source_path: str, topic_keywords: str) -> List[str]:
        """
        Uses Hybrid Tooling (Grep) to quickly find relevant documents.
        If topic_keywords is "*" or empty, lists all markdown files.
        """
        relevant_files = []
        
        # Clean keywords
        keywords = [k.strip().strip("'").strip('"') for k in topic_keywords.split(",")]
        primary_keyword = keywords[0] if keywords else ""

        if not primary_keyword or primary_keyword == "*":
            print(f"📚 [Librarian] Listing all .md files in {source_path}...")
            all_files = perform(ListFiles(FileList(dir_path=source_path)))
            relevant_files = [f for f in all_files if f.endswith(".md")]
        else:
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
                content = perform(ReadFile(FileRead(path=file_path)))
                docs.append(content)
            except Exception as e:
                print(f"    Failed to read {file_path}: {e}")
                continue
        
        return docs