# BiomedCLIP Fine-tuning Framework

깔끔하게 정리된 BiomedCLIP 파인튜닝 프레임워크입니다. 다양한 contrastive loss 함수를 사용하여 breast tumor 분류 태스크에 BiomedCLIP을 fine-tune할 수 있습니다.

## 특징

- **다양한 Loss 함수 지원**: CLIP, SigLIP, Hard Negative Loss (HNL), MGCA, GLoRIA
- **모듈화된 구조**: 새로운 loss 함수 추가가 쉬움
- **통합된 인터페이스**: 단일 train.py와 test.py로 모든 loss 실험 가능
- **깔끔한 코드**: 명확한 디렉토리 구조와 문서화

## 프로젝트 구조

```
biomedclip_finetuning/
├── README.md
├── requirements.txt
├── config.py                  # 공통 설정 (prompts, hyperparameters)
├── losses/                    # Loss 함수 모듈
│   ├── __init__.py
│   ├── clip_loss.py          # Standard CLIP loss
│   ├── siglip_loss.py        # SigLIP loss
│   ├── hnl_loss.py           # Hard Negative Loss
│   ├── mgca_loss.py          # MGCA loss
│   └── gloria_loss.py        # GLoRIA loss
├── models/                    # 모델 wrapper
│   ├── __init__.py
│   └── biomedclip_wrapper.py
├── data/                      # 데이터셋
│   ├── __init__.py
│   └── tumor_dataset.py
├── train.py                   # 통합 학습 스크립트
├── test.py                    # 통합 테스트 스크립트
└── scripts/                   # 실행 스크립트
    ├── train_clip.sh
    ├── train_siglip.sh
    ├── train_hnl.sh
    ├── train_mgca.sh
    ├── train_gloria.sh
    └── test.sh
```

## 설치

```bash
cd biomedclip_finetuning
pip install -r requirements.txt
```

## 사용법

### 1. 학습 (Training)

#### Option A: Python 스크립트 직접 실행

```bash
# CLIP loss로 학습
python train.py \
    --train-dir /path/to/train_images \
    --output-dir checkpoints/clip_exp \
    --loss clip \
    --batch-size 8 \
    --epochs 50 \
    --lr 2e-5 \
    --gpu 0

# SigLIP loss로 학습
python train.py \
    --train-dir /path/to/train_images \
    --output-dir checkpoints/siglip_exp \
    --loss siglip \
    --batch-size 8 \
    --epochs 50 \
    --lr 2e-5 \
    --gpu 0

# Hard Negative Loss로 학습
python train.py \
    --train-dir /path/to/train_images \
    --output-dir checkpoints/hnl_exp \
    --loss hnl \
    --batch-size 8 \
    --epochs 50 \
    --lr 2e-5 \
    --gpu 0

# MGCA Loss로 학습
python train.py \
    --train-dir /path/to/train_images \
    --output-dir checkpoints/mgca_exp \
    --loss mgca \
    --batch-size 8 \
    --epochs 50 \
    --lr 2e-5 \
    --gpu 0

# GLoRIA Loss로 학습
python train.py \
    --train-dir /path/to/train_images \
    --output-dir checkpoints/gloria_exp \
    --loss gloria \
    --batch-size 8 \
    --epochs 50 \
    --lr 2e-5 \
    --gpu 0
```

#### Option B: Shell 스크립트 사용

```bash
# scripts/*.sh 파일 편집하여 데이터 경로와 하이퍼파라미터 설정
bash scripts/train_clip.sh
bash scripts/train_siglip.sh
bash scripts/train_hnl.sh
bash scripts/train_mgca.sh
bash scripts/train_gloria.sh
```

### 2. 테스트 (Testing)

#### 단일 이미지 분류

```bash
python test.py \
    --checkpoint checkpoints/clip_exp/best_model.pt \
    --image /path/to/image.png \
    --gpu 0
```

#### 배치 분류 (Accuracy 측정)

```bash
python test.py \
    --checkpoint checkpoints/clip_exp/best_model.pt \
    --image-dir /path/to/test_images \
    --pattern "*.png" \
    --gpu 0
```

#### Shell 스크립트 사용

```bash
# scripts/test.sh 파일 편집하여 checkpoint 경로 설정
bash scripts/test.sh
```

## Loss 함수 설명

### 1. CLIP Loss (InfoNCE)
- 표준 CLIP contrastive loss
- Softmax 기반 cross-entropy
- 참고: [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)

### 2. SigLIP Loss
- Sigmoid 기반 pairwise loss
- Batch size에 덜 민감
- 참고: [Sigmoid Loss for Language Image Pre-Training](https://arxiv.org/abs/2303.15343)

### 3. Hard Negative Loss (HNL)
- Hard negative reweighting
- 어려운 negative 샘플에 더 큰 가중치
- 참고: [Hard Negative Noise Contrastive Estimation](https://arxiv.org/abs/2301.02280)

### 4. MGCA Loss
- Multi-Granularity Cross-modal Alignment
- Global + Local + Prototype 정렬
- 가장 복잡하지만 강력한 성능
- 참고: MGCA 논문

### 5. GLoRIA Loss
- Global-Local Representations for Images
- Attention 기반 local alignment
- Global + Local contrastive learning
- 참고: [GLoRIA: A Multimodal Global-Local Representation Learning Framework](https://openaccess.thecvf.com/content/ICCV2021/papers/Huang_GLoRIA_A_Multimodal_Global-Local_Representation_Learning_Framework_for_Label-Efficient_Medical_ICCV_2021_paper.pdf)

## 새로운 Loss 추가하기

1. `losses/` 폴더에 새 loss 모듈 생성:
```python
# losses/my_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyLoss(nn.Module):
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, image_features, text_features):
        # Your loss implementation
        loss = ...
        return loss
```

2. `losses/__init__.py`에 추가:
```python
from .my_loss import MyLoss
__all__ = [..., 'MyLoss']
```

3. `config.py`에 하이퍼파라미터 추가:
```python
LOSS_CONFIGS = {
    ...
    'myloss': {
        'temperature': 1.0,
        # other hyperparameters
    }
}
```

4. `train.py`의 `create_loss_function()`에 추가:
```python
elif loss_name == 'myloss':
    return MyLoss(temperature=config['temperature'])
```

5. 실행 스크립트 생성:
```bash
# scripts/train_myloss.sh
python train.py --loss myloss ...
```

## 하이퍼파라미터

기본 하이퍼파라미터는 `config.py`에 정의되어 있습니다:

```python
DEFAULT_TRAINING_CONFIG = {
    'batch_size': 8,
    'epochs': 50,
    'lr': 2e-5,
    'weight_decay': 0.05,
    'num_workers': 4,
    'warmup_epochs': 20,
    'initial_lr': 1e-8,
    'seed': 42,
    'max_length': 77,
}
```

Loss별 하이퍼파라미터는 `LOSS_CONFIGS` 딕셔너리에서 확인할 수 있습니다.

## TensorBoard

학습 중 TensorBoard 로그가 자동으로 저장됩니다:

```bash
tensorboard --logdir checkpoints/clip_exp/logs
```

## 참고사항

- 데이터셋 형식: 이미지 파일명이 `benign*.png` 또는 `malignant*.png`로 시작해야 합니다
- Checkpoint 형식: PyTorch state dict로 저장됩니다
- GPU 메모리: batch size는 GPU 메모리에 맞게 조정하세요

## 원본 코드

이 프로젝트는 MGCA 레포지토리의 흩어진 파인튜닝 코드들을 깔끔하게 정리한 것입니다:
- `finetuning_with_tumor_mgca_pure.py`
- `finetuning_with_tumor_hnl.py`
- `finetuning_with_tumor_siglip.py`
- `test_tumor_classification.py`
- `losses.py`

## License

MIT License
