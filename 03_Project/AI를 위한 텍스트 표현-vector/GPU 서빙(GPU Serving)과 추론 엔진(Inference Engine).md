#GPU_서빙 #GPU_Serving #추론_엔진 #Inference_Engine

### 1) 개념(Concept)

**GPU 서빙(GPU Serving)**은 **거대언어모델(LLM)**과 같이 막대한 연산량이 필요한 AI 모델을 **그래픽 처리 장치(GPU)**의 병렬 연산 능력을 활용하여 실시간으로 추론(Inference)하는 기술입니다. **NVIDIA Triton Inference Server**와 같은 전문 엔진은 여러 개의 모델을 하나의 GPU에서 효율적으로 실행하거나, 수많은 사용자의 요청을 최적으로 배포하여 응답 속도를 극대화하는 역할을 합니다.

---

### 2) 왜 필요한가(Background & Problem Solving)

- **연산 병목(Computation Bottleneck):** **LLM**은 수십억 개의 파라미터를 가지고 있어 **중앙 처리 장치(CPU)**로 처리하면 응답에 수분이 소요됩니다. 이를 수밀리초(ms) 단위로 줄이기 위해 수천 개의 코어를 가진 GPU가 필수적입니다.
    
- **비효율적인 자원 사용:** 일반적인 API 서버는 GPU 메모리(VRAM) 관리 능력이 부족합니다. 한 번에 하나의 요청만 처리하면 GPU의 남는 자원이 낭비되고 비용 효율성이 떨어집니다.
    
- **다양한 프레임워크 대응:** 실제 프로젝트에서는 PyTorch, TensorFlow, ONNX 등 다양한 형식의 모델을 사용합니다. 이를 통합 관리하고 최적화할 표준화된 서빙 환경이 필요합니다.
    

---

### 3) 어떻게 동작하는가(Mechanism)

**NVIDIA Triton**과 같은 전문 엔진은 다음과 같은 핵심 기능을 통해 GPU 자원을 최적화합니다.

1. **동적 배칭(Dynamic Batching):** 개별적으로 들어오는 여러 요청을 짧은 대기 시간 동안 모아서 하나의 큰 **배치(Batch)**로 묶어 GPU에 전달합니다. 이는 GPU의 처리량(Throughput)을 비약적으로 높입니다.
    
2. **동시 모델 실행(Concurrent Model Execution):** 하나의 GPU 안에 여러 모델 인스턴스를 올려, 자원이 허용하는 한 동시에 여러 추론을 수행합니다.
    
3. **모델 리포지토리(Model Repository):** **S3**와 같은 저장소에 저장된 모델 파일들을 버전별로 관리하고, 서비스 중단 없이 모델을 업데이트(Hot-swap)합니다.
    

---

### 4) 수학적 관점: GPU 처리량(Throughput)과 지연 시간(Latency)

GPU 서빙의 핵심 목표는 정해진 시간 내에 처리하는 데이터 양($\text{Throughput}$)을 최대화하는 것입니다.

$$\text{Throughput} = \frac{\text{Batch Size} \times \text{Number of GPUs}}{\text{Execution Time per Batch}}$$

- **① 수식 전체 의미:** 전체 시스템이 초당 처리할 수 있는 토큰 또는 요청의 수는 배치 크기와 GPU 개수에 비례하며, 한 배치를 계산하는 데 걸리는 시간에 반비례합니다.
    
- **② 각 기호의 의미:**
    
    - $\text{Batch Size}$: 한 번의 연산에 포함된 데이터 묶음의 크기
        
    - $\text{Number of GPUs}$: 투입된 전체 GPU 장비 수
        
    - $\text{Execution Time}$: GPU가 한 배치를 계산하는 데 걸리는 시간(초)
        
- **③ 실제 계산 방식:**
    
    - 배치 크기가 32이고, 2대의 GPU를 사용하며, 한 배치를 처리하는 데 0.5초가 걸린다면:
        
    - $\text{Throughput} = (32 \times 2) / 0.5 = 128 \text{ requests/sec}$
        
- **④ 왜 필요한가:** 실무자는 이 수식을 통해 사용자 트래픽을 감당하기 위해 필요한 최소 GPU 대수와 최적의 배치 크기를 산출하여 인프라 비용을 최적화합니다.
    

---

### 5) 프로젝트 관점: LLM 추론 흐름

실제 시스템에서는 다음과 같은 흐름으로 배치됩니다.

**데이터(Prompt) → API Gateway(부하 분산) → Triton(동적 배칭) → GPU(연산) → 결과(Token) 반환**

- **MLOps 연계:** 모델의 사용량과 GPU 온도를 실시간으로 모니터링하여, 트래픽이 몰리면 **쿠버네티스(Kubernetes)**를 통해 GPU 노드를 자동으로 확장(Auto-scaling)합니다.
    
- **RAG 구조 내 역할:** 사용자가 질문을 하면 **임베딩(Embedding)** 모델과 **LLM** 모델이 각각 Triton 서버에서 호출되어 답변을 생성합니다.
    

---

### 6) 이해를 돕는 비유

- **초등학생 수준:** 한 명씩 태우는 승용차(CPU) 대신, 수많은 사람을 한꺼번에 태우고 빠르게 달리는 고속열차(GPU)를 운행하는 것과 같습니다. **Triton**은 열차 시간표를 짜고 승강장에서 사람들을 모아 효율적으로 태우는 역무원 역할을 합니다.
    
- **실무자 수준:** **VRAM**의 고정 비용을 상쇄하기 위해 **배치 처리(Batch Processing)**와 **모델 병렬화(Model Parallelism)**를 극대화하고, gRPC/HTTP 통신 레이어를 최적화하여 **Inference Latency**를 최소화하는 인프라 솔루션입니다.
    

---

[다음 단계 학습]

- **양자화(Quantization):** 모델의 가중치를 정밀도를 낮춰 저장함으로써 GPU 메모리 사용량을 줄이고 속도를 높이는 기법입니다.
    
- **vLLM:** LLM 추론에 특화된 엔진으로, **KV 캐시(KV Cache)** 관리를 통해 Triton보다 더 높은 처리량을 내는 최신 기술입니다.
    
- **TensorRT:** NVIDIA GPU에서 딥러닝 모델의 실행 속도를 최적화하기 위해 모델 구조를 분석하고 재구성하는 컴파일러입니다.
    
- **멀티-GPU 병렬 처리(Model Parallelism):** 하나의 모델이 너무 커서 GPU 1대에 들어가지 않을 때, 모델을 쪼개어 여러 GPU에 나누어 올리는 기술입니다.
    
