FROM nvidia/cuda:11.7.0-runtime-ubuntu18.04


# 기본적인 파이썬 패키지 설치
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3-pip \
    python3.9-dev

# 파이썬이 기본 python 명령어로 사용되도록 설정
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1

# pip를 최신 버전으로 업그레이드
RUN python -m pip install --upgrade pip

COPY requirement.txt .

RUN pip install -r requirement.txt

RUN python download_models.py

COPY . .

EXPOSE 5000

CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]