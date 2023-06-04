# <span style="color:green">Style Bible</span>
<br>
:sunglasses:곽동호 T5013 :sunglasses:권수훈 T5017 :sunglasses:박상우 T5081
<br><br>
:sunglasses:이민호 T5140 :sunglasses:이한정 T5166 :sunglasses:이준원 T5237
<br>

# GitHub 협업 규칙
이 문서는 RecSys 1조 Style Bible의 GitHub 협업 규칙을 담고 있습니다.
<br>

## 1. 커밋 컨벤션
커밋 메시지 구조는 header, body, footer 세 가지 파트로 나누고, 각 파트는 빈 줄을 두어 구분한다.
<br>

### Header
머리말은 아래와 같은 구조로 이루어져 있습니다.
- tag: subject
- tag는 소문자로 시작
- subject는 소문자로 시작
<br>

#### Tag
- 태그: 커밋의 종류를 나타냅니다. 주요 태그 아래와 같습니다:

 - `feat`: 새로운 기능 추가
 - `fix`: 버그 수정
 - `docs`: 문서 관련 변경
 - `setting`: 프로젝트 초기 설정
 - `style`: 코드 스타일 변경 (공백, 포맷 등)
 - `refactor`: 코드 리팩토링
 - `test`: 테스트 관련 변경
 - `chore`: 그 외 자잘한 변경
 <br>

#### Subject
제목은 다음의 규칙을 지킨다.

- 최대 50글자가 넘지 않도록 하고 마침표 및 특수기호는 사용하지 않는다.
- 명령문으로 작성한다.
<br>

### Body
본문은 다음의 규칙을 지킨다.

- 본문은 한 줄당 72자 내로 작성한다.
- 본문 내용은 양에 구애받지 않고 최대한 상세히 작성한다.
- 본문 내용은 어떻게 변경했는지보다 무엇을 변경했는지 또는 왜 변경했는지를 설명한다.
<br>

### Footer
꼬리말은 다음의 규칙을 지킨다.

- 꼬리말은 optional이고 이슈 트래커 ID를 작성한다.
- 꼬리말은 " #이슈 번호" 형식으로 사용한다.
- 여러 개의 이슈 번호를 적을 때는 쉼표(,)로 구분한다.
<br>

### 커밋 메시지 예시
>setting: add issue template
>
>To use when implementing new features.
>
> #1
---
<br>

## 2. ISSUE / PR
ISSUE / PR은 다음의 규칙을 지킨다.
- ISSUE 기반으로 커밋을 트래킹한다.
- PR은 최소 3명 이상의 확인을 받고 승인해야 한다.
- ISSUE / PR은 템플릿을 만들어서 사용하고 아래의 규칙에 따라 제목을 설정해야 한다.
<br>

### 제목

 - [Feat] 새로운 기능 추가
 - [Fix] 버그 수정
 - [Docs] 문서 관련 변경
 - [Setting] 프로젝트 초기 설정
 - [Style] 코드 스타일 변경 (공백, 포맷 등)
 - [Refactor] 코드 리팩토링
 - [Test] 테스트 관련 변경
 - [Chore] 그 외 자잘한 변경
<br>

### ISSUE/PR 제목 예시
> [Feat] Saint+ 모델 구현
---
<br>

## 3. 대회를 위한 Git Flow

### 모델링 가이드

- 새로운 모델을 만들 때, ISSUE를 생성한다.
 >[FEAT] Saint+ 모델 구현
- master에서 브랜치를 base: 모델명으로 생성한다.
 >base: saint+
- base: 모델명 브랜치에서 사용자명: 모델명으로 브랜치를 생성해서 작업한다.
 >junwon: saint+
- 정상적으로 동작하는 기능이나 공유하고 싶은 자료가 있을 때, base: 모델명 브랜치에 사용자의 branch를 merge 한다.

- 모델이 완성되었을 때, master에 base: 모델명을 merge 한다.
<br>

## 4. 프로젝트 폴더 구조 
<br>
<img width="400" alt="image" src="https://github.com/boostcampaitech5/level2_movierecommendation-recsys-01/assets/69078499/ec79e214-73d7-4623-8ea5-edeb0949aa3b">
<br>

- README.md

    README 파일은 프로젝트의 개요를 제공합니다.<br>
    목적, 설치 지침, 사용 가이드라인 및 기타 관련 정보가 포함됩니다.

- data

    csv 파일이나 데이터를 저장하여 사용하는 폴더입니다.<br>
    파일은 빈 폴더로 존재해야 하며 데이터 위치를 지정해서 사용하기 위해 빈 폴더를 생성하였습니다.

- notebooks

    EDA, 데이터 전처리, 시각화와 관련된 주피터 노트북을 저장하는 곳입니다.

- src

    해당 위치에 모델별로 폴더를 생성하여 작업합니다.
<br>

### 파이토치 템플릿
파이토치 템플릿은 아래 링크를 참고하여 작성한다.
- 참조 링크
[기본 템플릿](https://github.com/victoresque/pytorch-template)
[간단한 버전](https://smha-61749.medium.com/pytorch-%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-%EB%94%A5%EB%9F%AC%EB%8B%9D-%ED%94%84%EB%A1%9C%EC%A0%9D%ED%8A%B8-%EC%84%A4%EA%B3%84%ED%95%98%EA%B3%A0-%ED%85%9C%ED%94%8C%EB%A6%BF-%EA%B5%AC%EC%84%B1%ED%95%98%EA%B8%B0-ccf222552e63)

<br>