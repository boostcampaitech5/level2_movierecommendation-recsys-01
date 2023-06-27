<a name="readme-top"></a>

<div align="center">  

![header](https://capsule-render.vercel.app/api?type=rounded&color=0:81ed89,100:06c007&text=Style%20Bible&height=150&fontSize=80&fontColor=d3ffd8)

---
  <br>
  :sunglasses:곽동호 T5013 :moneybag:권수훈 T5017 :smile_cat:박상우 T5081
  <br><br>
  :smile:이민호 T5140 :stuck_out_tongue_winking_eye:이한정 T5166 :relaxed:이준원 T5237
  <br><br><br>
  
  <p align="center"><strong>Skills</strong>
    <br />

---
<br>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=Python&logoColor=white" alt="Python badge">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=PyTorch&logoColor=white" alt="PyTorch badge">
  <img src="https://img.shields.io/badge/pandas-150458?style=flat-square&logo=pandas&logoColor=white" alt="pandas badge">
  <img src="https://img.shields.io/badge/numpy-013243?style=flat-square&logo=numpy&logoColor=white" alt="numpy badge">
  <img src="https://img.shields.io/badge/scikit learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white" alt="scikitlearn badge">
    <img src="https://img.shields.io/badge/wandb-FFBE00?style=flat-square&logo=weightsandbiases&logoColor=white" alt="weightsandbiases badge">
</p>
  
<br><br>

<h1> Movie Recommendation</h1>
<img src = ./images/main_image.png , width =500, height=250>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#프로젝트-개요">프로젝트 개요</a></li>
    <li><a href="#협업-규칙">협업 규칙</a></li>
    <li><a href="#프로젝트-구조">프로젝트 구조</a></li>
    <li><a href="#로드맵">로드맵</a></li>
    <li><a href="#모델-사용-방법">모델 사용 방법</a></li>
    <li><a href="#모델별-성능">모델별 성능</a></li>
    <li><a href="#순위">순위</a></li>
    <li><a href="#Wrap-up-Report">Wrap up Report</a></li>
  </ol>
</details>
<br>

<!-- 프로젝트 개요 -->
## 프로젝트 개요

위의 그림과 같이 사용자의 영화 시청 이력 데이터를 활용하여 다음에 시청할 영화 및 좋아할 영화를 예측하는 것이 프로젝트의 목표입니다. 이 대회에서는 MovieLens 데이터를 전처리하여 사용되며, **implicit feedback**만 존재합니다. **시간순으로 정렬된 시퀀스**에서 **일부 아이템이 누락**된 상황을 가정합니다. 이를 통해 **실제 상황**과 유사한 예측을 수행하는 것이 목표입니다.

<br>

- **input:** user의 implicit 데이터, item(movie)의 meta데이터
- **output:** user에게 추천하는 item을 user, item이 ','로 구분된 파일(csv) 로 제출합니다.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- 협업 규칙 -->
## 협업 규칙
<br>

- [협업 규칙](/RuleBook.md)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<br>


## 로드맵

<img src = images/Roadmap.png>

<p align="right">(<a href="#readme-top">back to top</a>)</p>
<br>

## 프로젝트 구조

    📦MovieRec
    ┣ 📂data
    ┣ 📂images
    ┣ 📂notebooks
    ┣ 📂src
    ┃ ┣ 📂ALS
    ┃ ┣ 📂EASE
    ┃ ┣ 📂Ensemble
    ┃ ┣ 📂Recbole
    ┃ ┗ 📂Sequential
    ┣ 📜README.md
    ┗ 📜RuleBook.md

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## 모델 사용 방법

각 모델별 사용 방법입니다.

- [ALS](/src/ALS/README.md)

- [EASE](/src/EASE/README.md)

- [Recbole](/src/EASE/Recbole.md)

- [Sequential](/src/Sequential/README.md)


## 앙상블 방법

- [Ensemble](/src/Ensemble/README.md)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## 모델별 성능

- RecBole

<img width =400px src = images/RecBole.png>

- 최종 제출 결과

<img width = 400px src = images/PublicScore.png>

<img width = 400px src = images/PrivateScore.png>


<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- 순위 -->
## 순위

- Public 4/12

<img width="600" alt="Pasted Graphic 1" src="https://github.com/boostcampaitech5/level2_movierecommendation-recsys-01/assets/69078499/eee01baf-2b5f-47c9-afcb-0323abfa3d9e">

- Private 7/12

<img width="600" alt="Pasted Graphic 2" src="https://github.com/boostcampaitech5/level2_movierecommendation-recsys-01/assets/69078499/6d7e9bcb-3fee-4abe-9e54-361631b2170d">

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## Wrap-up Report

[Wrap-up Report](images/WrapUpReport.pdf)

<p align="right">(<a href="#readme-top">back to top</a>)</p>