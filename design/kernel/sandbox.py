import math, re, json, datetime, collections, itertools, functools, statistics
import traceback
from typing import Dict, Any, List

class SandboxExecutor:
    """확장된 Soft Sandbox 실행기"""
    
    def __init__(self):
        # 허용된 모듈 화이트리스트
        self.allowed_modules = {
            "math": math, "re": re, "json": json, 
            "datetime": datetime, "collections": collections,
            "itertools": itertools, "functools": functools, 
            "statistics": statistics
        }
        
    def _create_restricted_environment(self) -> Dict[str, Any]:
        """위험한 내장 함수(__import__, open 등)를 제거한 환경 생성"""
        safe_builtins = {
            "abs": abs, "all": all, "any": any, "bin": bin, "bool": bool,
            "dict": dict, "enumerate": enumerate, "filter": filter, "float": float,
            "format": format, "frozenset": frozenset, "int": int, "len": len,
            "list": list, "map": map, "max": max, "min": min, "range": range,
            "round": round, "set": set, "sorted": sorted, "str": str, "sum": sum,
            "tuple": tuple, "zip": zip,
            "print": print # 로깅용 허용
        }
        
        env = {"__builtins__": safe_builtins}
        env.update(self.allowed_modules)
        return env

    def execute(self, tool_name: str, args: Dict) -> Any:
        # 실제 ToolRegistry 연동 로직은 여기에 위치하거나
        # tool_name이 "python_repl"일 경우 아래 로직 사용
        pass

    def execute_python_code(self, code: str, timeout_sec: int = 5) -> Dict[str, Any]:
        """Python 코드를 격리된 환경에서 실행"""
        env = self._create_restricted_environment()
        
        try:
            # TODO: 실제 프로덕션에서는 multiprocessing으로 Timeout 및 메모리 제한 필요
            # 현재는 exec() 기반의 Soft Sandbox
            exec(code, env)
            
            # 실행 후 결과값 추출 약속 (예: result 변수에 담기)
            return {"success": True, "result": env.get("result", "No result variable set")}
            
        except Exception as e:
            return {
                "success": False, 
                "error": str(e),
                "traceback": traceback.format_exc()
            }