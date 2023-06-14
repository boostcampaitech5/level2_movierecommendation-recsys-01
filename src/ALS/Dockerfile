FROM --platform=linux/amd64 python:3.8-slim-buster AS builder

LABEL description="ALS model with implicit feedback"
LABEL maintainer="darrenkwondev46@gmail.com"

WORKDIR /app

RUN apt-get update && apt-get install --no-install-recommends -y wget libgomp1 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# deps install cache
COPY ./requirements.txt /app
RUN pip3 install -r requirements.txt

FROM --platform=linux/amd64 builder
LABEL maintainer="darrenkwondev46@gmail.com"

COPY . /app

CMD ["sh", "ENTRYPOINT.sh"]