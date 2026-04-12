
# I부 AI에이전트 엔진: 텍스트에서 대규모 언어 모델까지

## 1장 딥러닝으로 텍스트 데이터 분석하기
* 원-핫 인코딩(One-hot encoding)
* 단어 가방(BoW, Bag of Words)
* 단어 빈도-역문서 빈도(TF-IDF, term frequecny-inverse document frequency)
* 워드투벡(word2vec)
* 순환 신경망(RNN, recurrent neural network)
* 장단기 메모리(LSTM, Long short-term memory)
* 게이트 순환 유닛(GRU, Gated recurrent unit)
* 합성곱 신경망(CNN, convolutional neural network)
* 텍스트 뭉치(corpus, 코퍼스)

컴퓨터는 이미지나 표보다 텍스트를 더 처리하기 까다롭니다.
단어의 의미(signified, 기의)와 이를 나타내는 기호(sinifier, 기표) 사이에는 고정된 일대일 대응 관계가 존재하지 않고 작성자의 의도에 따라 문맥이 달라지기 때문

### 첫번째 단계 : 텍스트 코퍼스를 정해진 기본 단위로 분할
#### 토큰화 (tokenization)
문장이 있을 때 공백을 기준으로 단어를 구분(text segmentation)
이때 문장 부호도 하나의 단어로 간주

### 두번째 단계:  전처리
구체적으로 '단어'를 어떻게 정의할 것인지, 코퍼스에 포함된 의미는 같지만 다르게 인식될 수 있는 특정 용어들을 동일한 어휘로 묶을지 여부를 결정하는 것

#### 텍스트 정규화 (text nomalization)
* 소문자 변환 : 'He', 'he' 가 같은 단어임을 인식할 수 있도록 전처리 과정에서 텍스트를 소문자로 변환해서 정규화 할 수 있다
* 표제어 추출(lemmaztization) : 'cams'와 'comes'를 동사 원형인 'come'으로 변경해서 동일하게 인식
* 어간 추출(stemming) : 접미사가 붙은 단어들에서 접미사를 제거하여 원형으로 변경 



