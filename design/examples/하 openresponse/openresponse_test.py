"""
Ollama Open Response API를 사용한 SPAK Kernel 테스트
기존 WorkerLLMCommunicator를 Open Response API로 교체
"""

import asyncio
import aiohttp
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid

# ========== 데이터 타입 정의 (간소화) ==========
class ItemType(Enum):
    REASONING = "reasoning"
    FUNCTION_CALL = "function_call"
    MESSAGE = "message"
    HYPOTHESIS = "hypothesis"

@dataclass
class OpenResponseItem:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: ItemType = ItemType.MESSAGE
    content: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

# ========== Ollama Open Response API 통신기 ==========
class OllamaOpenResponseCommunicator:
    """
    Ollama의 Open Response API(v1/responses)를 사용한 통신기
    """
    
    def __init__(self, 
                 base_url: str = "http://localhost:11434/v1",
                 model: str = "qwen2.5:7b",
                 api_key: str = "ollama"):
        
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.api_key = api_key
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        """비동기 컨텍스트 매니저 시작"""
        self.session = aiohttp.ClientSession(
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """비동기 컨텍스트 매니저 종료"""
        if self.session:
            await self.session.close()
    
    async def send_request(self, 
                          context: Dict[str, Any],
                          trace_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Open Response API로 요청 전송
        
        API 문서: https://github.com/ollama/ollama/blob/main/docs/openai.md#responses-api
        """
        
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        # Open Response API 요청 포맷 구성
        request_payload = {
            "model": self.model,
            "input": self._build_prompt(context),
            "temperature": 0.3,
            "max_completion_tokens": 1024,
            "stream": False,
            "tools": context.get("tools", []),
            "tool_choice": context.get("tool_choice", "auto")
        }
        
        # Metadata 추가 (선택사항)
        if trace_id:
            request_payload["metadata"] = {
                "trace_id": trace_id,
                "agent_system": "SPAK_Kernel"
            }
        
        try:
            # Open Response API 호출
            async with self.session.post(
                f"{self.base_url}/responses",
                json=request_payload,
                timeout=aiohttp.ClientTimeout(total=180)
            ) as response:
                
                if response.status == 200:
                    response_data = await response.json()
                    return await self._process_openresponse(response_data, context)
                else:
                    error_text = await response.text()
                    return {
                        "success": False,
                        "error": f"HTTP {response.status}: {error_text}",
                        "status_code": response.status
                    }
                    
        except aiohttp.ClientError as e:
            return {
                "success": False,
                "error": f"Connection error: {str(e)}"
            }
        except asyncio.TimeoutError:
            return {
                "success": False,
                "error": "Request timeout"
            }
    
    def _build_prompt(self, context: Dict[str, Any]) -> str:
        """컨텍스트를 프롬프트로 변환"""
        prompt_parts = []
        
        # 시스템 역할
        if "system_role" in context:
            prompt_parts.append(f"시스템 역할: {context['system_role']}")
        
        # 전략적 컨텍스트
        if "strategic_context" in context:
            prompt_parts.append(f"\n## 전략적 목표\n{context['strategic_context']}")
        
        # 작업 설명
        if "task_description" in context:
            prompt_parts.append(f"\n## 작업\n{context['task_description']}")
        
        # 응답 형식 요구사항
        response_format = context.get("response_format", {})
        if response_format:
            prompt_parts.append("\n## 응답 형식")
            if "required_items" in response_format:
                prompt_parts.append("필수 응답 항목:")
                for item in response_format["required_items"]:
                    prompt_parts.append(f"- {item['description']}")
            
            if "json_format" in response_format:
                prompt_parts.append(f"\nJSON 형식:\n{json.dumps(response_format['json_format'], indent=2, ensure_ascii=False)}")
        
        return "\n".join(prompt_parts)
    
    async def _process_openresponse(self, 
                                   response_data: Dict[str, Any],
                                   context: Dict[str, Any]) -> Dict[str, Any]:
        """Open Response API 응답 처리"""
        
        # Open Response API의 응답 구조 파싱
        items = []
        
        # 주 텍스트 응답 처리
        if "output_text" in response_data:
            reasoning_item = OpenResponseItem(
                type=ItemType.REASONING,
                content={
                    "text": response_data["output_text"],
                    "usage": response_data.get("usage", {}),
                    "model": response_data.get("model", self.model)
                }
            )
            items.append(reasoning_item)
        
        # Tool Calls 처리
        if "tool_calls" in response_data and response_data["tool_calls"]:
            for tool_call in response_data["tool_calls"]:
                function_item = OpenResponseItem(
                    type=ItemType.FUNCTION_CALL,
                    content={
                        "tool": tool_call.get("name", ""),
                        "arguments": tool_call.get("arguments", {}),
                        "tool_call_id": tool_call.get("id", "")
                    }
                )
                items.append(function_item)
        
        # 메시지 아이템으로 변환
        if not items and "output_text" in response_data:
            # JSON 응답 파싱 시도
            try:
                json_response = json.loads(response_data["output_text"])
                if "hypothesis" in json_response:
                    items.append(OpenResponseItem(
                        type=ItemType.HYPOTHESIS,
                        content=json_response["hypothesis"]
                    ))
                if "reasoning" in json_response:
                    items.append(OpenResponseItem(
                        type=ItemType.REASONING,
                        content={"text": json_response["reasoning"]}
                    ))
            except json.JSONDecodeError:
                # 일반 텍스트 응답
                items.append(OpenResponseItem(
                    type=ItemType.MESSAGE,
                    content={"text": response_data["output_text"]}
                ))
        
        return {
            "success": True,
            "items": items,
            "raw_response": response_data,
            "model": response_data.get("model", self.model),
            "usage": response_data.get("usage", {}),
            "created": response_data.get("created", datetime.now().isoformat())
        }

# ========== 간소화된 테스트 커널 ==========
class TestSPAKKernel:
    """Open Response API를 사용한 테스트 커널"""
    
    def __init__(self, model: str = "qwen2.5:7b"):
        self.model = model
        self.communicator: Optional[OllamaOpenResponseCommunicator] = None
        self.trace_id = f"trace_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    async def initialize(self):
        """통신기 초기화"""
        self.communicator = OllamaOpenResponseCommunicator(model=self.model)
    
    async def execute_cognitive_task(self, task_description: str) -> Dict[str, Any]:
        """인지 작업 실행"""
        if not self.communicator:
            await self.initialize()
        
        # 컨텍스트 구성
        context = {
            "system_role": "당신은 과학적 연구를 지원하는 엔지니어링 에이전트입니다.",
            "task_description": task_description,
            "strategic_context": "정확성과 검증 가능성을 우선시하세요.",
            "response_format": {
                "required_items": [
                    {"type": "reasoning", "description": "문제 해결을 위한 단계별 추론"},
                    {"type": "hypothesis", "description": "검증 가능한 가설"}
                ],
                "json_format": {
                    "reasoning": "추론 과정 텍스트",
                    "hypothesis": {
                        "statement": "가설 진술",
                        "testability": "검증 가능성 설명",
                        "expected_evidence": "기대되는 증거"
                    },
                    "action_plan": [
                        {
                            "step": "단계 설명",
                            "method": "사용 방법",
                            "expected_outcome": "기대 결과"
                        }
                    ]
                }
            }
        }
        
        # Open Response API 호출
        async with self.communicator as comm:
            result = await comm.send_request(context, trace_id=self.trace_id)
        
        return result
    
    async def test_function_calling(self) -> Dict[str, Any]:
        """함수 호출 기능 테스트"""
        
        # 함수 정의
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "analyze_data",
                    "description": "데이터 분석을 수행합니다",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "dataset": {
                                "type": "string",
                                "description": "분석할 데이터셋 이름"
                            },
                            "method": {
                                "type": "string",
                                "enum": ["statistical", "machine_learning", "visualization"],
                                "description": "사용할 분석 방법"
                            },
                            "parameters": {
                                "type": "object",
                                "description": "분석 파라미터"
                            }
                        },
                        "required": ["dataset", "method"]
                    }
                }
            }
        ]
        
        context = {
            "system_role": "당신은 데이터 분석 전문가입니다.",
            "task_description": "고객 데이터셋을 분석하고 주요 인사이트를 찾아주세요.",
            "tools": tools,
            "tool_choice": "auto",
            "response_format": {
                "json_format": {
                    "analysis_plan": "분석 계획",
                    "required_tools": "필요한 도구 목록"
                }
            }
        }
        
        async with self.communicator as comm:
            result = await comm.send_request(context, trace_id=self.trace_id)
        
        return result

# ========== 테스트 실행기 ==========
async def run_openresponse_tests():
    """Open Response API 통합 테스트"""
    
    print("=" * 60)
    print("Ollama Open Response API 테스트 시작")
    print("=" * 60)
    
    # 1. 기본 통신 테스트
    print("\n[1] 기본 통신 테스트...")
    kernel = TestSPAKKernel(model="qwen2.5:7b")
    await kernel.initialize()
    
    # 간단한 작업 테스트
    task = "수질 오염을 모니터링하기 위한 IoT 시스템을 설계하세요. 주요 고려사항을 설명하시오."
    print(f"작업: {task[:50]}...")
    
    result = await kernel.execute_cognitive_task(task)
    
    if result["success"]:
        print("✅ API 호출 성공")
        print(f"모델: {result.get('model', 'N/A')}")
        print(f"응답 항목 수: {len(result['items'])}")
        
        # 응답 내용 출력
        for i, item in enumerate(result["items"], 1):
            print(f"\n항목 {i} ({item.type.value}):")
            if item.type == ItemType.REASONING:
                content = item.content.get("text", "")
                print(f"   {content[:200]}...")
            elif item.type == ItemType.HYPOTHESIS:
                print(f"   가설: {json.dumps(item.content, ensure_ascii=False, indent=2)}")
    else:
        print(f"❌ API 호출 실패: {result.get('error', 'Unknown error')}")
    
    # 2. 함수 호출 테스트
    print("\n" + "=" * 60)
    print("[2] 함수 호출 테스트...")
    
    try:
        func_result = await kernel.test_function_calling()
        
        if func_result["success"]:
            print("✅ 함수 호출 테스트 성공")
            
            # 툴 콜 분석
            tool_calls = func_result.get("raw_response", {}).get("tool_calls", [])
            if tool_calls:
                print(f"발견된 툴 콜: {len(tool_calls)}개")
                for tc in tool_calls:
                    print(f"  - {tc.get('name', 'Unknown')}: {tc.get('arguments', {})}")
            else:
                print("ℹ️ 툴 콜이 없습니다. 텍스트 응답만 반환됨.")
                
                # 텍스트 응답 출력
                if "items" in func_result and func_result["items"]:
                    for item in func_result["items"]:
                        if item.type == ItemType.MESSAGE:
                            print(f"응답: {item.content.get('text', '')[:100]}...")
        else:
            print(f"❌ 함수 호출 테스트 실패: {func_result.get('error')}")
            
    except Exception as e:
        print(f"❌ 함수 호출 테스트 중 오류: {str(e)}")
    
    # 3. 응답 구조 분석
    print("\n" + "=" * 60)
    print("[3] 응답 구조 분석...")
    
    if result["success"] and "raw_response" in result:
        raw_resp = result["raw_response"]
        
        print(f"응답 ID: {raw_resp.get('id', 'N/A')}")
        print(f"생성 시간: {raw_resp.get('created', 'N/A')}")
        
        # 사용량 정보
        usage = raw_resp.get("usage", {})
        if usage:
            print(f"토큰 사용량:")
            print(f"  입력: {usage.get('input_tokens', 0)} 토큰")
            print(f"  출력: {usage.get('output_tokens', 0)} 토큰")
            print(f"  총합: {usage.get('total_tokens', 0)} 토큰")
        
        # 응답 메타데이터
        metadata = raw_resp.get("metadata", {})
        if metadata:
            print(f"메타데이터: {metadata}")
    
    print("\n" + "=" * 60)
    print("테스트 완료")
    print("=" * 60)

# ========== Ollama 서비스 체크 ==========
async def check_ollama_service():
    """Ollama 서비스 상태 확인"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:11434/api/tags", 
                                 timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    models = data.get("models", [])
                    print(f"✅ Ollama 서비스 실행 중")
                    print(f"   사용 가능한 모델: {len(models)}개")
                    for model in models[:3]:  # 처음 3개만 출력
                        print(f"   - {model.get('name', 'Unknown')}")
                    if len(models) > 3:
                        print(f"   ... 외 {len(models)-3}개")
                    return True
                else:
                    print(f"❌ Ollama 서비스 응답 오류: {resp.status}")
                    return False
    except Exception as e:
        print(f"❌ Ollama 서비스 연결 실패: {str(e)}")
        print("   실행 명령어: ollama serve")
        return False

# ========== 메인 실행 ==========
async def main():
    """메인 실행 함수"""
    
    print("Ollama Open Response API 통합 테스트")
    print(f"테스트 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 서비스 확인
    service_ok = await check_ollama_service()
    if not service_ok:
        print("\nOllama 서비스를 시작한 후 다시 시도해주세요.")
        print("명령어: ollama serve")
        return
    
    # 테스트 실행
    try:
        await run_openresponse_tests()
    except Exception as e:
        print(f"\n❌ 테스트 실행 중 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()

# ========== 기존 코드와의 통합 예제 ==========
def create_integration_example():
    """기존 SPAK Kernel과의 통합 예제"""
    
    integration_example = '''
# 기존 ExtendedSPAKKernel과의 통합 방법

class IntegratedWorkerLLMCommunicator:
    """기존 시스템과 Open Response API 통합"""
    
    def __init__(self, endpoint: str = "http://localhost:11434/v1"):
        # 기존 endpoint 형식 변환
        if "worker-llm" in endpoint:
            # 기존 엔드포인트를 Open Response API로 변환
            self.ollama_communicator = OllamaOpenResponseCommunicator(
                base_url=endpoint.replace("/worker-llm", ""),
                model="gemma3:1b"
            )
        else:
            self.ollama_communicator = OllamaOpenResponseCommunicator(
                base_url=endpoint,
                model="gemma3:1b"
            )
    
    async def send_request(self, context: Dict) -> Dict:
        """기존 인터페이스 유지하면서 Open Response API 사용"""
        
        # 기존 컨텍스트를 Open Response API 형식으로 변환
        openresponse_context = {
            "system_role": context.get("system_model", {}).get("description", ""),
            "task_description": context.get("prompt", ""),
            "tools": self._convert_tools(context.get("tools", [])),
            "response_format": context.get("response_format", {})
        }
        
        async with self.ollama_communicator as comm:
            result = await comm.send_request(
                openresponse_context,
                trace_id=context.get("trace_id")
            )
        
        # 결과를 기존 형식으로 변환
        return self._convert_to_legacy_format(result)

# 사용 예시:
# 1. 기존 Kernel 초기화
kernel = ExtendedSPAKKernel(
    spec_path="./spec.spak",
    meta_llm_endpoint="http://localhost:8001/meta-llm",
    worker_llm_endpoint="http://localhost:11434/v1"  # Open Response API로 변경
)

# 2. WorkerLLMCommunicator 교체
kernel.worker_communicator = IntegratedWorkerLLMCommunicator()
    '''
    
    return integration_example

if __name__ == "__main__":
    # 메인 테스트 실행
    asyncio.run(main())
    
    # 통합 예제 출력
    print("\n" + "=" * 60)
    print("기존 시스템 통합 예제")
    print("=" * 60)
    print(create_integration_example())