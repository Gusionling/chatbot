from openai import OpenAI
import json
import os
from config import OPENAI_API_KEY

class FineTuningManager:
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
    
    def create_training_data(self, data_list, output_file="training_data.jsonl"):
        """
        Fine-tuning용 JSONL 파일 생성
        
        Args:
            data_list: [{"prompt": "질문", "completion": "답변"}, ...] 형태의 리스트
            output_file: 출력할 파일명
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in data_list:
                training_example = {
                    "messages": [
                        {"role": "user", "content": item["prompt"]},
                        {"role": "assistant", "content": item["completion"]}
                    ]
                }
                f.write(json.dumps(training_example, ensure_ascii=False) + '\n')
        
        print(f"Training data saved to {output_file}")
        return output_file
    
    def upload_training_file(self, file_path):
        """
        OpenAI에 학습 파일 업로드
        
        Args:
            file_path: 업로드할 JSONL 파일 경로
            
        Returns:
            file_id: 업로드된 파일의 ID
        """
        try:
            with open(file_path, "rb") as f:
                response = self.client.files.create(
                    file=f,
                    purpose="fine-tune"
                )
            
            print(f"File uploaded successfully. File ID: {response.id}")
            return response.id
        
        except Exception as e:
            print(f"Error uploading file: {e}")
            return None
    
    def create_fine_tuning_job(self, training_file_id, model="gpt-3.5-turbo"):
        """
        Fine-tuning 작업 생성
        
        Args:
            training_file_id: 업로드된 학습 파일 ID
            model: 사용할 기본 모델
            
        Returns:
            job_id: 생성된 작업 ID
        """
        try:
            response = self.client.fine_tuning.jobs.create(
                training_file=training_file_id,
                model=model
            )
            
            print(f"Fine-tuning job created. Job ID: {response.id}")
            return response.id
        
        except Exception as e:
            print(f"Error creating fine-tuning job: {e}")
            return None
    
    def check_job_status(self, job_id):
        """
        Fine-tuning 작업 상태 확인
        
        Args:
            job_id: 확인할 작업 ID
        """
        try:
            response = self.client.fine_tuning.jobs.retrieve(job_id)
            
            print(f"Job Status: {response.status}")
            print(f"Model: {response.model}")
            
            if response.status == "succeeded":
                print(f"Fine-tuned model ID: {response.fine_tuned_model}")
                return response.fine_tuned_model
            elif response.status == "failed":
                print(f"Error: {response.error}")
            
            return response.status
        
        except Exception as e:
            print(f"Error checking job status: {e}")
            return None
    
    def list_fine_tuning_jobs(self):
        """
        Fine-tuning 작업 목록 조회
        """
        try:
            response = self.client.fine_tuning.jobs.list()
            
            print("Fine-tuning Jobs:")
            for job in response.data:
                print(f"  Job ID: {job.id}")
                print(f"  Status: {job.status}")
                print(f"  Model: {job.model}")
                if job.fine_tuned_model:
                    print(f"  Fine-tuned Model: {job.fine_tuned_model}")
                print("  ---")
        
        except Exception as e:
            print(f"Error listing jobs: {e}")

def create_sample_training_data():
    """
    샘플 학습 데이터 생성 (인텔 제품 관련)
    """
    sample_data = [
        {
            "prompt": "인텔 CPU의 최신 제품은 무엇인가요?",
            "completion": "인텔의 최신 CPU는 13세대 Intel Core 프로세서 시리즈입니다. 이 제품군은 높은 성능과 전력 효율성을 제공하며, 게이밍과 콘텐츠 제작에 최적화되어 있습니다."
        },
        {
            "prompt": "게이밍용 CPU 추천해주세요",
            "completion": "게이밍용으로는 Intel Core i5-13600K 또는 i7-13700K를 추천합니다. 높은 클록 속도와 멀티코어 성능으로 최신 게임을 원활하게 실행할 수 있습니다."
        },
        {
            "prompt": "사무용 컴퓨터에 어떤 CPU가 좋을까요?",
            "completion": "사무용으로는 Intel Core i3-13100 또는 i5-13400이 적합합니다. 일반적인 오피스 작업, 웹 브라우징, 문서 편집에 충분한 성능을 제공하면서 가성비가 우수합니다."
        },
        {
            "prompt": "인텔 Arc 그래픽카드에 대해 알려주세요",
            "completion": "Intel Arc는 인텔의 데스크톱 및 노트북용 그래픽카드 브랜드입니다. Arc A-시리즈는 1080p 및 1440p 게이밍에 적합하며, 하드웨어 가속 레이 트레이싱과 AI 기능을 지원합니다."
        },
        {
            "prompt": "인텔과 AMD CPU 차이점이 뭔가요?",
            "completion": "인텔 CPU는 일반적으로 게이밍 성능이 우수하고 단일 코어 성능이 강합니다. AMD는 멀티코어 성능과 가성비가 뛰어납니다. 최근에는 두 브랜드 모두 경쟁력 있는 제품을 출시하고 있어 용도에 따라 선택하시면 됩니다."
        }
    ]
    
    return sample_data

if __name__ == "__main__":
    # Fine-tuning 매니저 초기화
    ft_manager = FineTuningManager()
    
    # 샘플 데이터 생성
    sample_data = create_sample_training_data()
    
    # 학습 데이터 파일 생성
    training_file = ft_manager.create_training_data(sample_data, "intel_products_training.jsonl")
    
    print("Fine-tuning을 시작하려면 다음 단계를 따르세요:")
    print("1. 학습 데이터를 더 추가하세요 (최소 10개 이상 권장)")
    print("2. 아래 코드를 실행하여 파일을 업로드하고 fine-tuning을 시작하세요:")
    print(f"""
# 파일 업로드
file_id = ft_manager.upload_training_file("{training_file}")

# Fine-tuning 작업 시작
job_id = ft_manager.create_fine_tuning_job(file_id)

# 작업 상태 확인
ft_manager.check_job_status(job_id)
""")
