import ollama
import textwrap
import time
import os

class DocumentSummarizer:
    def __init__(self, model_name="qwen2.5:7b"):
        """초기화 및 모델 설정"""
        print(f"=== 문서 요약 및 구조화 시스템 시작 ===")
        print(f"[1단계] 모델 초기화: {model_name}")
        
        # 사용할 모델 설정
        self.model = model_name
        
        # 테스트용 문서 데이터
        self.text_data1 = """인공지능(AI) 기술의 발전은 현대 사회에 많은 변화를 가져오고 있습니다. 
AI는 의료, 교육, 금융, 교통 등 다양한 분야에서 혁신을 주도하고 있으며, 
특히 딥러닝 기술의 발전으로 인해 이미지 인식, 자연어 처리, 음성 인식 등의 분야에서 
놀라운 성과를 보여주고 있습니다. AI 기술은 또한 기업의 생산성 향상과 비용 절감에도 
기여하고 있어, 많은 기업들이 AI 도입을 적극적으로 검토하고 있습니다.

그러나 AI 기술의 발전은 동시에 윤리적 문제와 일자리 감소에 대한 우려도 함께 가져왔습니다. 
자동화로 인한 일자리 감소 문제는 사회적 논쟁이 되고 있으며, 
AI 알고리즘의 편향성 문제도 중요한 이슈로 떠오르고 있습니다. 
이에 따라 각국 정부와 국제기구는 AI 윤리 가이드라인과 규제를 마련하고 있습니다.

미래에는 AI와 인간의 협력이 더욱 중요해질 것으로 예상됩니다. 
AI는 인간의 보조 도구로서 일상 생활과 업무를 지원하며, 
새로운 창의적인 분야에서 인간의 능력을 확장시켜 줄 것입니다."""

        self.text_data2 = """지구 온난화와 기후 변화는 현재 인류가 직면한 가장 심각한 문제 중 하나입니다. 
과학자들은 지난 100년 동안 지구 평균 기온이 약 1.1도 상승했으며, 
이 추세가 계속된다면 2100년까지 2.7도 이상 상승할 수 있다고 경고하고 있습니다.

기후 변화의 주요 원인은 인간 활동으로 인한 온실가스 배출입니다. 
화석 연료 사용, 산림 파괴, 산업 활동 등이 대기 중 이산화탄소 농도를 
산업화 이전보다 50% 이상 증가시켰습니다. 이로 인해 극지방의 빙하가 녹고, 
해수면이 상승하며, 이상 기후 현상이 빈번해지고 있습니다.

기후 변화의 영향은 전 세계적으로 나타나고 있습니다. 열대성 폭풍의 강도 증가, 
가뭄과 산불 발생 빈도 증가, 농업 생산성 감소, 생물다양성 손실 등이 대표적입니다. 
특히 해안가 지역과 섬 국가들은 해수면 상승으로 인한 침수 위협에 직면해 있습니다.

국제 사회는 파리 협정을 통해 지구 평균 기온 상승을 산업화 이전 대비 
1.5도 이내로 제한하기로 합의했습니다. 이를 위해 각국은 탄소 중립 목표를 설정하고, 
재생 에너지 확대, 전기차 보급, 에너지 효율 향상 등의 정책을 시행하고 있습니다."""
        
        print(f"[1단계] 테스트 문서 2개가 메모리에 로드되었습니다.")
        print(f"[1단계] 초기화 완료\n")
    
    def check_model_availability(self):
        """모델 가용성 확인 - 간소화된 버전"""
        print(f"[2단계] 모델 가용성 확인 중...")
        
        try:
            # 방법 1: ollama.list()의 응답 구조 확인
            print("  방법1: ollama.list()로 확인...")
            model_list = ollama.list()
            
            # 다양한 응답 구조 처리
            if isinstance(model_list, dict):
                if 'models' in model_list:
                    models = model_list['models']
                    print(f"    ✓ 딕셔너리 형식, 'models' 키 존재")
                else:
                    models = model_list
                    print(f"    ⚠ 딕셔너리 형식이지만 'models' 키 없음")
            elif isinstance(model_list, list):
                models = model_list
                print(f"    ✓ 리스트 형식 직접 반환")
            else:
                print(f"    ⚠ 예상치 못한 응답 형식: {type(model_list)}")
                models = []
            
            # 모델 이름 추출
            model_names = []
            if models:
                if isinstance(models, list):
                    for model in models:
                        if isinstance(model, dict):
                            if 'name' in model:
                                model_names.append(model['name'])
                            elif 'model' in model:
                                model_names.append(model['model'])
                        elif isinstance(model, str):
                            model_names.append(model)
            
            print(f"  발견된 모델: {model_names}")
            
            # 모델 존재 여부 확인
            if self.model in model_names:
                print(f"  ✓ '{self.model}' 모델이 사용 가능합니다.")
                return True
            else:
                print(f"  ✗ '{self.model}' 모델을 찾을 수 없습니다.")
                
                # 방법 2: 직접 API 호출로 확인
                print("  방법2: 직접 API 호출로 확인...")
                try:
                    # 짧은 테스트 호출
                    response = ollama.generate(
                        model=self.model,
                        prompt='test',
                        options={'num_predict': 5}
                    )
                    print(f"    ✓ '{self.model}' 모델 응답 성공")
                    return True
                except Exception as api_error:
                    print(f"    ✗ '{self.model}' 모델 응답 실패: {api_error}")
                    return False
                
        except Exception as e:
            print(f"  ✗ 모델 확인 중 오류: {e}")
            print(f"  오류 상세: {type(e).__name__}")
            
            # 에러가 발생해도 직접 시도해보기
            print("  방법3: 직접 시도로 확인...")
            try:
                response = ollama.generate(model=self.model, prompt='test', stream=False)
                print(f"  ✓ '{self.model}' 모델 직접 호출 성공")
                return True
            except:
                print(f"  ✗ '{self.model}' 모델 직접 호출도 실패")
                return False
    
    def create_structure_template(self):
        """구조화 템플릿 생성"""
        print(f"[3단계] 요약 구조 템플릿 생성 중...")
        
        structure_template = """
다음 형식으로 문서를 요약해주세요:

## 제목
[문서의 핵심 주제 제목]

## 요약  
[3-4문장으로 전체 내용 요약]

## 주요 내용
1. [첫 번째 핵심 내용]
2. [두 번째 핵심 내용]
3. [세 번째 핵심 내용]

## 결론/시사점
[문서 내용에서 도출되는 결론이나 시사점]

"""
        print(f"  ✓ 구조 템플릿이 생성되었습니다.")
        print()
        return structure_template
    
    def summarize_document(self, doc_text, doc_name, structure_template):
        """단일 문서 요약 및 구조화"""
        print(f"\n[4단계] '{doc_name}' 문서 처리 시작")
        print("-" * 50)
        
        start_time = time.time()
        
        # 프롬프트 생성
        prompt = f"""아래 문서를 분석하여 요약해주세요. 반드시 주어진 형식에 맞춰 작성해주세요.

문서 제목: {doc_name}
문서 내용:
{doc_text}

{structure_template}

주의사항:
- 주어진 형식을 정확히 따르세요
- 핵심 내용을 빠짐없이 포함하세요
- 객관적이고 명확하게 작성하세요
"""
        
        try:
            print(f"  Ollama API 호출 중...")
            
            # 간단한 API 호출
            response = ollama.chat(
                model=self.model,
                messages=[
                    {'role': 'system', 'content': '당신은 전문 문서 분석가입니다. 문서를 정확히 분석하여 요약해주세요.'},
                    {'role': 'user', 'content': prompt}
                ],
                options={
                    'num_predict': 500,
                    'temperature': 0.3
                }
            )
            
            full_response = response['message']['content']
            elapsed_time = time.time() - start_time
            
            print(f"  ✓ 문서 처리 완료 (소요 시간: {elapsed_time:.1f}초)")
            print(f"  응답 길이: {len(full_response)}자")
            print("-" * 50)
            
            return full_response
            
        except Exception as e:
            print(f"  ✗ 문서 처리 중 오류: {str(e)}")
            return f"[오류] 문서 처리 실패: {str(e)}"
    
    def process_all_documents(self):
        """모든 문서 처리"""
        print(f"\n{'='*60}")
        print(f"문서 요약 및 구조화 프로세스 시작")
        print(f"{'='*60}")
        
        # 1. 모델 확인 (단순화)
        print(f"[2단계] 모델 확인 (간소화)...")
        try:
            # 직접 모델 호출 시도
            test_response = ollama.generate(
                model=self.model,
                prompt='test',
                options={'num_predict': 3}
            )
            print(f"  ✓ '{self.model}' 모델 응답 확인")
        except Exception as e:
            print(f"  ✗ 모델 확인 실패: {e}")
            print(f"  모델 '{self.model}'을(를) 찾을 수 없습니다.")
            print(f"  설치 명령어: ollama pull {self.model}")
            return None
        
        # 2. 구조 템플릿 생성
        structure_template = self.create_structure_template()
        
        # 3. 문서 처리
        print(f"\n[4단계] 문서 처리 시작")
        results = {}
        
        # 문서1 처리
        print(f"\n문서 1/2 처리 중: 'AI 기술 발전과 영향'")
        results['AI 기술 발전과 영향'] = self.summarize_document(
            self.text_data1, 
            "AI 기술 발전과 영향", 
            structure_template
        )
        
        # 잠시 대기
        time.sleep(1)
        
        # 문서2 처리  
        print(f"\n문서 2/2 처리 중: '기후 변화와 대응 방안'")
        results['기후 변화와 대응 방안'] = self.summarize_document(
            self.text_data2,
            "기후 변화와 대응 방안",
            structure_template
        )
        
        # 결과 통계
        print(f"\n{'='*60}")
        print(f"처리 결과 요약")
        print(f"{'='*60}")
        success_count = sum(1 for r in results.values() if not r.startswith('[오류]'))
        print(f"총 문서: {len(results)}개")
        print(f"성공: {success_count}개")
        print(f"실패: {len(results) - success_count}개")
        print(f"{'='*60}\n")
        
        return results
    
    def save_results(self, results, filename="요약_결과.txt"):
        """결과를 파일로 저장"""
        print(f"[5단계] 결과 저장 중...")
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("="*60 + "\n")
                f.write("문서 요약 및 구조화 결과\n")
                f.write(f"생성 시간: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("="*60 + "\n\n")
                
                for doc_name, content in results.items():
                    f.write(f"[{doc_name}]\n")
                    f.write("-"*40 + "\n")
                    f.write(content)
                    if not content.endswith('\n'):
                        f.write('\n')
                    f.write("\n" + "="*60 + "\n\n")
            
            file_size = os.path.getsize(filename)
            print(f"  ✓ '{filename}' 저장 완료 ({file_size} 바이트)")
            return True
        except Exception as e:
            print(f"  ✗ 저장 실패: {e}")
            return False
    
    def display_results(self, results):
        """결과를 화면에 출력"""
        print(f"\n{'='*60}")
        print(f"최종 결과 출력")
        print(f"{'='*60}")
        
        for doc_name, content in results.items():
            print(f"\n[{doc_name}]")
            print("-" * 40)
            print(content)
            if not content.endswith('\n'):
                print()
            print("=" * 60)

# 메인 실행 부분
if __name__ == "__main__":
    print("Ollama 문서 요약 시스템 v1.2")
    print("RTX 1060 3GB 최적화 버전")
    print("=" * 60)
    
    try:
        summarizer = DocumentSummarizer()
        results = summarizer.process_all_documents()
        
        if results:
            summarizer.display_results(results)
            summarizer.save_results(results)
            
            print(f"\n" + "="*60)
            print(f"✅ 처리 완료!")
            print(f"✅ 결과 파일: '요약_결과.txt'")
            print(f"="*60)
        else:
            print(f"\n❌ 처리 실패")
            
    except KeyboardInterrupt:
        print(f"\n⚠️ 중단됨")
    except Exception as e:
        print(f"\n❌ 오류: {e}")

    input("\n계속하려면 엔터 키를 누르세요...")