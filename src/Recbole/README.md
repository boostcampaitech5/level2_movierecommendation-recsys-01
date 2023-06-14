# RecBole

다양한 모델을 사용하기 위하여 RecBole 파이프라인을 구현하였습니다. 현재 General model, Context Aware model을 사용할 수 있습니다.

# Files

preprocess.py - train에 필요한 dataset 디렉토리와 inter, user item context를 생성합니다.   
main.py - Movie Rec 대회에 맞는 Recbole 모델을 training 합니다. (args: model_name, model_type, make_yaml) 
inference.py - path를 바탕으로 inference 합니다. (args: path)

# Directory Structure

saved - main.py를 통해 학습된 best model이 저장됩니다.
dataset - RecBole에서 사용하는 data 디렉토리입니다. preprocess.py의 output으로 생성되며, 하위에 context_train, general_train 디렉토리가 생성됩니다.   
trainer - 학습 및 추론에 사용되는 파이썬 코드를 저장한 디렉토리입니다. 
utils - 자잘한 파이썬 코드를 저장한 디렉토리입니다. 
general_yaml - general model 훈련에 필요한 yaml 파일이 담긴 디렉토리 입니다. 
context_yaml - context model 훈련에 필요한 yaml 파일이 담긴 디렉토리 입니다. 


# How to run

```
conda create -n recbole python=3.8.5
conda activate recbole
pip install -r requirements.txt 

python preprocess.py
python main.py 
python inference.py --path [path_name]
```

0. 한번만 preprocess.py 파일을 실행해주시면 됩니다. 
1. main.py 파일 시행시에, --model_name 인자에 소문자로 모델명을 입력해주셔야 합니다. (현재 BPR, EASE, LightGCN, MultiVAE, NeuMF, FFM, FM 사용 가능) (default = EASE)
2. main.py 파일의 --model_name과 동일한 --model_type 을 지정해주셔야 합니다. (default = general)
3. main.py 파일의 --make_yaml 인자를 통해 yaml 파일을 생성할 수 있습니다. (모델별로 한번만 생성하시면 됩니다) (default = True)
4. inference.py 파일의 --path 인자는 필수로 입력해주셔야 합니다. (saved에 저장된 파일 이름을 확장자 포함하여 입력해주시면 됩니다) 
5. inference.py를 실행하시면 submission.csv 파일이 생성됩니다. 
6. yaml 디렉토리의 모델 yaml을 조정하여 학습 인자를 조정할 수 있습니다. 


# More information

작업중 ...