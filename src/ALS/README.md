## 할 일

-   [x] docker python image들의 차이에 대해서 공부하기
-   [ ] multi build 공부하기
-   [ ] ipynb를 py로 변경하기
-   [x] dockerfile 짜기 (with volume)
    -   [ ] volume flag

## docker와 data

docker container에서 머신러닝 알고리즘을 수행하려고 합니다.
이 때 큰 용량의 csv가 존재하는데 여기서 질문이 있습니다.

1. data를 volumes로 연결하여 사용하는 것은 좋지 않습니까?
2. data는 컨테이너 내부에 있는 것이 좋습니까 바깥에 있는 것이 좋습니까? 만약 바깥에 있는 것이 옳은 패턴이라면 데이터 로드는 어떤 방식이 좋습니까?

## 환경 구성에 대한 참고 자료

-   python 환경에서 alpine 이미지는 왜 권장되지 않는가?

    -   [Using Apline can make Python Docker builds 50x slower](https://pythonspeed.com/articles/alpine-docker-python/)
    -   Alpine 리눅스는 용량이 작은 이유는 C 라이브러리로 musl을 채택했고, 다양한 유닉스 도구들을 탑재한 busybox를 기반으로 하고 있기 때문이다.
    -   Alpine 리눅스는 Wheel 포맷을 지원하지 않는다!! 그래서 직접 소스 코드(.tar.gz)를 내려받아 컴파일 해야 하는 경우가 생긴다.
    -   Wheel 포맷을 지원하지 않는 이유는 [musl](https://ko.wikipedia.org/wiki/Musl) 라이브러리가 GNU C 라이브러리(glibc)로 컴파일된 wheel 바이너리를 지원하지 않기 때문. 그래서 PyPI에서 라이브러리를 받을 때 .whl 패키지가 아닌 .tar.gz를 다운받음

-   multi build는 무엇인가?

    -   https://docs.docker.com/build/building/multi-stage/

-   volume flags는 무엇인가?

    -   https://mytory.net/2018/06/21/docker-volume-flags-description.html

-   compose 관련 참고

    -   [compose-sample](https://docs.docker.com/samples/)
    -   [공식문서) compose overview](https://docs.docker.com/compose/)

-   https://luis-sena.medium.com/creating-the-perfect-python-dockerfile-51bdec41f1c8
-   https://jonnung.dev/docker/2020/04/08/optimizing-docker-images/#gsc.tab=0

-   docker compose and hot-reload

    -   docker compose 쓸 때 volume을 현재 dir로 해놔야 hot reload처럼 받아다가 개발 가능함.
    -   https://jonnung.dev/docker/2020/04/08/optimizing-docker-images/#gsc.tab=0
    -   https://olshansky.medium.com/hot-reloading-with-local-docker-development-1ec5dbaa4a65
    -   https://www.freecodecamp.org/news/how-to-enable-live-reload-on-docker-based-applications/

-   플랫폼 관련
    -   export DOCKER_DEFAULT_PLATFORM=linux/amd64
    -   docker build --platform linux/amd64 -t dev-lec .
    -   [buildx](https://github.com/docker/buildx)

## dvc

-   https://kaushikshakkari.medium.com/hands-on-data-versioning-code-versioning-with-dvc-git-and-aws-s3-99237d2baa23

git init
dvc init

dvc add data/ -> .dvc 생성 + .gitignore에 data/ 추가 + .dvcignore 생성 + data.dvc 생성
git add data/data.xml.dvc

dvc

## data

```bash
    wget https://aistages-prod-server-public.s3.amazonaws.com/app/Competitions/000243/data/data.tar.gz
    tar -xf data.tar.gz
    rm data.tar.gz
```
