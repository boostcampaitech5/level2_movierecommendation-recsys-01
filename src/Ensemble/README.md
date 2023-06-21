# Ensemble

- 추천 결과의 유사도를 구하는 코드
- Top-K 앙상블을 진행하는 코드 (soft voting + hard voting)
- 상호작용 횟수를 기준으로 user를 분리하여 다른 모델을 적용하는 코드 

## Files

- get_similiarty.py
    추천한 아이템이 얼마나 겹치는지 %를 보여주는 코드입니다. 
<br>

- topk_ensemble.py
    Soft Voting 방식과 Hard Voting 방식을 선택하여 추천 결과를 앙상블하는 코드입니다.
<br>

- split_by_interaction.py
    훈련 데이터의 유저의 상호작용 횟수별로 유저를 나눠 다른 모델을 적용하는 코드입니다.
<br>

## Directory Structure

- submission
    유사도를 구할 submission.csv 파일을 저장하는 디렉토리입니다. (get_similiarity.py 코드에 사용됩니다)
<br>
- for_ensemble
    앙상블을 진행할 submission.csv 파일을 저장하는 디렉토리입니다. (topk_ensemble.py 코드에 사용됩니다)
<br>
- output
    앙상블이 진행된 후, 최종 결과 파일이 저장되는 디렉토리입니다.

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

### topk_ensemble.py
- option
    soft: 가중치를 주어 최종 결과를 출력한다.
    hard: 모델별로 k개를 뽑아 앙상블한다.
```bash
python topk_ensemble.py --option "soft" --file_path "/for_ensemble" --weight "0.5,0.5"

python topk_ensemble.py --option "hard" --file_path "/for_ensemble" --weight "0.5,0.5"
```

### Notice

0. submission 디렉토리에 저장할 csv 파일은 aistage에 제출하는 csv 파일과 동일한 형식이어야 합니다. 

1. get_similarity.py 코드를 실행하면 submission 디렉토리에 존재하는 모든 csv 파일 쌍에 대해 유사도를 구합니다.

2. for_ensemble 디렉토리에 저장하는 csv 파일은 추천 아이템이 10개 이상이고, 모든 csv는 동일한 추천 아이템 개수를 가져야 합니다. 

3. topk_ensemble의 weight 인자에는 문자열 형식으로 가중치를 입력해야 합니다. (ex: "0.4,0.4,0.2")

    1. 파일명을 기준으로 정렬을 하므로 a,b,c,d 형식으로 가중치에 맞게 지정해야 한다.

    2. weight 인자에 입력할 리스트는 띄어쓰기 없이 입력해야 합니다. 

4. topk_ensemble.py의 로직은 다음과 같습니다.

    1. for_ensemble 디렉토리에 저장된 csv 파일들을 읽어옵니다.

    2. 각 유저별로 추천된 아이템 Rank를 파일 별로 가중치를 주어 합합니다. 

    3. 합한 Rank를 다시 정렬하여 상위 10개의 아이템을 추천합니다.


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
