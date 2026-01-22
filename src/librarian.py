from typing import List
import os
from kernel.semantic_kernel import perform
from kernel.effects import ListFiles, FileList, ReadFile, FileRead, GrepFiles, GrepRequest

class Librarian:
    def gather_materials(self, source_path: str, topic_keywords: str) -> List[str]:
        """
        Uses Hybrid Tooling (Grep) to quickly find relevant documents.
        Supports multi-keyword search (space or comma separated).
        """
        relevant_files = set()
        
        # 1. Parse keywords
        # If explicitly comma-separated, prioritize that. Otherwise split by space.
        raw_input = topic_keywords.strip().strip("'").strip('"')
        
        if not raw_input:
            return []

        if "," in raw_input:
            keywords = [k.strip() for k in raw_input.split(",")]
        else:
            keywords = [k.strip() for k in raw_input.split(" ")]
            
        print(f"📚 [Librarian] Searching for keywords: {keywords} in {source_path}...")

        # 2. Search Loop
        for kw in keywords:
            if not kw: continue
            
            if kw == "*":
                print(f"    Listing all .md files (wildcard)...")
                files = perform(ListFiles(FileList(dir_path=source_path)))
                relevant_files.update([f for f in files if f.endswith(".md")])
            else:
                # Fast Path: Grep
                # print(f"    Grep: '{kw}'")
                files = perform(GrepFiles(GrepRequest(
                    pattern=kw,
                    dir_path=source_path,
                    file_pattern="*.md" 
                )))
                relevant_files.update(files)
        
        # Convert to list
        unique_files = list(relevant_files)
        print(f"    Found {len(unique_files)} unique candidate files.")
        
        # 3. Ingestion
        docs = []
        for file_path in unique_files:
            try:
                content = perform(ReadFile(FileRead(path=file_path)))
                docs.append(content)
            except Exception as e:
                print(f"    Failed to read {file_path}: {e}")
                continue
        
        return docs
