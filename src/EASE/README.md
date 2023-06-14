# EASE model

## How to Train
```bash
python train.py
```

## How to inference
```bash
python inference.py
```

## Parameter
- lambda:학습 파라미터
- scale: matrix의 interaction 초기화값의 scale 조정
- cv: cv 평가를 진행 (False: 평가x)
- wandb: 개인 key, name 사용 권장!