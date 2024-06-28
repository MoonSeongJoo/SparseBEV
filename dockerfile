# 기본 이미지로 NVIDIA에서 제공하는 CUDA 11.8 베이스 이미지 사용
FROM nvidia/cuda:11.8.0-base-ubuntu20.04

# 필수 패키지 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cuda-command-line-tools-11-8 \
        cuda-cudart-dev-11-8 \
        cuda-libraries-dev-11-8 \
        cuda-minimal-build-11-8 \
        libcudnn8 \
        libcudnn8-dev \
        libnvinfer8 \
        libnvinfer-dev \
        libnvinfer-plugin8 \
        wget \
        && \
    rm -rf /var/lib/apt/lists/*

# Miniconda 설치
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /miniconda.sh && \
    bash /miniconda.sh -b -p /miniconda && \
    rm /miniconda.sh

# 환경 변수 설정
ENV CUDA_HOME /usr/local/cuda-11.8
ENV PATH /miniconda/bin:/usr/local/cuda-11.8/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/cuda-11.8/lib64:${LD_LIBRARY_PATH}

# Conda 환경 설정 및 PyTorch 설치
RUN conda create -n sparsebev python=3.8 -y && \
    echo "source activate sparsebev" > ~/.bashrc && \
    /bin/bash -c "source activate sparsebev && \
    conda install pytorch==2.0.0 torchvision==0.15.0 cudatoolkit=11.8 -c pytorch -c nvidia"

# 작업 디렉토리 설정
WORKDIR /workspace

# 컨테이너 실행 시 기본 명령
CMD [ "/bin/bash" ]



