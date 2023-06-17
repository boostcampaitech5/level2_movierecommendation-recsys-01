# Ensemble

추천 결과의 유사도를 구하는 코드와 Top-K 앙상블을 진행하는 코드를 구현하였습니다.

# Files

get_similiarty.py - 추천한 아이템이 얼마나 겹치는지 %를 보여주는 코드입니다. 

topk_ensemble.py - Soft Voting 방식을 사용하여 추천 결과를 앙상블하는 코드입니다.


# Directory Structure

submission - 유사도를 구할 submission.csv 파일을 저장하는 디렉토리입니다. (get_similiarity.py 코드에 사용됩니다)

for_ensemble - 앙상블을 진행할 submission.csv 파일을 저장하는 디렉토리입니다. (topk_ensemble.py 코드에 사용됩니다)

# How to run

```
# Numpy, Pandas, Tqdm 라이브러리가 모두 설치되어 있는 가상환경을 실행합니다.
# submission 디렉토리에 유사도를 구하고 싶은 csv파일을 저장합니다.
# for_ensemble 디렉토리에 앙상블을 진행할 csv파일을 저장합니다.

python get_similarity.py
python topk_ensemble.py --weight=[weight1, weight2, ...]
```

0. submission 디렉토리에 저장할 csv 파일은 aistage에 제출하는 csv 파일과 동일한 형식이어야 합니다. 

1. get_similarity.py 코드를 실행하면 submission 디렉토리에 존재하는 모든 csv 파일 쌍에 대해 유사도를 구합니다.

2. for_ensemble 디렉토리에 저장하는 csv 파일은 추천 아이템이 10개 이상이고, 모든 csv는 동일한 추천 아이템 개수를 가져야 합니다. 

3. topk_ensemble의 weight 인자에는 리스트 형식으로 가중치를 입력해야 합니다. (ex: [0.4, 0.4, 0.2])

4. topk_ensemble.py의 로직은 다음과 같습니다.

    1. for_ensemble 디렉토리에 저장된 csv 파일들을 읽어옵니다.

    2. 각 유저별로 추천된 아이템 Rank를 파일 별로 가중치를 주어 합합니다. 

    3. 합한 Rank를 다시 정렬하여 상위 10개의 아이템을 추천합니다.


# More information

작업중 ...