# Music Genre Classification

로그 멜 스펙트럼 데이터를 이용한 ResNet 기반 음악 장르 분류 모델입니다. FastAPI 서버로 배포하여 mp3 파일을 업로드하면 장르 예측 결과를 반환합니다.

## Project

원시 오디오 대신 로그 멜 스펙트럼만으로 장르를 분류할 수 있을지 검증하기 위해 시작한 개인 프로젝트입니다. 시간 축 정보 사용 범위 변화에 따른 증강 효과가 성능에 미치는 영향을 분석하기 위해 다음 세 가지 학습 전략을 비교했습니다: Random 512-frame crop, Random 1024-frame crop, Full sequence
PyTorch로 4개의 residual block을 가지는 ResNet을 직접 구현하여 실험을 수행했습니다.

## Model
- Input: (48, 1876) log-mel spectrogram
- Architecture: Custom ResNet
- Data Augmentation:
  - Random 512 frame crop
- Inference:
  - Sliding window
  - Soft voting aggregation

## Result
- Top-1 Accuracy: 56%
-> 512프레임 랜덤크롭이 타 학습 방식에 비해 최대 1.5% 높은 성능을 보였습니다. 

+ 실제 API에서 예측할 때는 Sliding window와 soft voting을 적용했는데, 단일 구간 기반 Validation이 실제 서비스 환경과 불일치한다는 점을 인지하였고, Evaludation 단계에서도 sliding window + soft voting을 적용해 학습과 추론 전략을 일관되게 맞추어 보다 현실적인 성능 지표를 산출할 수 있었습니다.
-> 3만 개 데이터를 사용했을 때 기존 accuracy 49.12%에 비해 sliding window와 soft voting을 적용했을 때 accuracy가 52.17%로 상승

+ 클래스 불균형을 고려하여 Accuracy뿐만 아니라 Macro F1 Score를 함께 보고하면 좋을 것 같습니다.

Colab GPU Access의 제한으로 인해 대규모 데이터(10만 개)를 사용했을 때 3만 개의 데이터를 사용했을 때보다 10% 이상 높은 초기 Accuracy를 보여주었지만, 충분한 epoch 학습이 이루어지지 못해 underfitting 가능성이 있습니다.

## Dataset 
제3회 카카오 아레나 대회에서 공개된 Melon Playlist Dataset을 사용해 모델을 학습했습니다. 멜 스펙트럼은 1876프레임의 48개 특징값으로 추출되어 있습니다. 

Ferraro A., Kim Y., Lee S., Kim. B., Jo N., Lim S., Lim S., Jan J., Kim S., Serra X. & Bogdanov D. (2021). "Melon Playlist Dataset: a public dataset for audio-based playlist generation and music tagging". International Conference on Acoustics, Speech and Signal Processing (ICASSP 2021).