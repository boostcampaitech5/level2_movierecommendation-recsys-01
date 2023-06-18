# Ensemble

추천 결과의 유사도를 구하는 코드와 Top-K 앙상블을 진행하는 코드를 구현하였습니다.

## Files

- get_similiarty.py
    추천한 아이템이 얼마나 겹치는지 %를 보여주는 코드입니다. 
<br>

- topk_with_weight.py
    Soft Voting 방식을 사용하여 추천 결과를 앙상블하는 코드입니다.
<br>

-  topk_without_weight.py
    모델별로 상위의 아이템을 그대로 가져와 추천 결과를 앙상블하는 코드입니다.
<br>

- split_by_interaction.py
    훈련 데이터의 유저의 상호작용 횟수별로 유저를 나눠 다른 모델을 적용하는 코드입니다.
<br>

## Directory Structure

submission - 유사도를 구할 submission.csv 파일을 저장하는 디렉토리입니다. (get_similiarity.py 코드에 사용됩니다)
<br>
for_ensemble - 앙상블을 진행할 submission.csv 파일을 저장하는 디렉토리입니다. (topk_ensemble.py 코드에 사용됩니다)
<br><br>

## How to run

```
# Numpy, Pandas, Tqdm 라이브러리가 모두 설치되어 있는 가상환경을 실행합니다.
# submission 디렉토리에 유사도를 구하고 싶은 csv파일을 저장합니다.
# for_ensemble 디렉토리에 앙상블을 진행할 csv파일을 저장합니다.
```

### get_similarity.py
```bash
python get_similarity.py
```

### topk_with_weight.py
```bash
python topk_with_weight.py --weight=[weight1, weight2, ...]
```

### topk_without_weight.py
두 파일을 앙상블합니다.
main 모델과 sub 모델을 파일명을 입력하고 main 모델에서 사용할 item 개수를 지정합니다.
```bash
python topk_without_weight.py --main_file MAIN_FILE --sub_file SUB_FILE --using_topk USING_TOPK
```

### split_by_interaction.py
두 파일을 앙상블합니다.
coldstart에 강한 모델을 less_file에, 상호작용이 많을 경우 잘 동작하는 모델을 much_file에 지정하고 상호작용 횟수를 지정합니다.
### 주의사항
전체 훈련 데이터를 기준으로 유저를 나누므로 기준이되는 훈련 파일을 지정해야 합니다.
기본 설정은 우리 프로젝트의 구조에 주어진 데이터 파일을 넣은 위치입니다!

```bash
python split_by_interaction.py --train_path TRAIN_PATH --file_path FILE_PATH --less_file LESS_FILE --much_file MUCH_FILE --num_interaction NUM_INTERACTION
```

<br><br>

0. submission 디렉토리에 저장할 csv 파일은 aistage에 제출하는 csv 파일과 동일한 형식이어야 합니다. 

1. get_similarity.py 코드를 실행하면 submission 디렉토리에 존재하는 모든 csv 파일 쌍에 대해 유사도를 구합니다.

2. for_ensemble 디렉토리에 저장하는 csv 파일은 추천 아이템이 10개 이상이고, 모든 csv는 동일한 추천 아이템 개수를 가져야 합니다. 

3. topk_ensemble의 weight 인자에는 리스트 형식으로 가중치를 입력해야 합니다. (ex: [0.4,0.4,0.2])

    1. 어떤 파일에 얼마나 가중치를 줄 지는, abcd 순으로 입력하여 결정할 수 있습니다.

    2. weight 인자에 입력할 리스트는 띄어쓰기 없이 입력해야 합니다. 

4. topk_ensemble.py의 로직은 다음과 같습니다.

    1. for_ensemble 디렉토리에 저장된 csv 파일들을 읽어옵니다.

    2. 각 유저별로 추천된 아이템 Rank를 파일 별로 가중치를 주어 합합니다. 

    3. 합한 Rank를 다시 정렬하여 상위 10개의 아이템을 추천합니다.