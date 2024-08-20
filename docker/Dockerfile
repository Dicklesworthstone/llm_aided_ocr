# syntax=docker/dockerfile:1
# NOTE: Building this image require's docker version >= 23.0.
#
# For reference:
# - https://docs.docker.com/build/dockerfile/frontend/#stable-channel
ARG TAG_VERSION="12.1.1" 
FROM nvidia/cuda:${TAG_VERSION}-cudnn8-devel-ubuntu22.04 
ARG HTTP_PROXY
ARG HTTPS_PROXY
ENV http_proxy=${HTTP_PROXY}
ENV https_proxy=${HTTPS_PROXY}
ARG DEBIAN_FRONTEND="noninteractive"
ENV DEBIAN_FRONTEND=${DEBIAN_FRONTEND}

WORKDIR /root
SHELL ["/bin/bash", "-c"]

ARG ROOT_PASSWD="root"
ENV ROOT_PASSWD=${ROOT_PASSWD}
ARG SSH_PORT="2323"
ENV SSH_PORT=${SSH_PORT}
# base tools
RUN <<EOT
#!/bin/bash
apt-get update
apt-get install -y libgl1-mesa-glx bash-completion wget curl htop jq vim bash libaio-dev build-essential openssh-server openssh-client python3 python3-pip python3-venv bzip2
apt-get install -y --no-install-recommends software-properties-common build-essential autotools-dev nfs-common pdsh cmake g++ gcc curl wget vim tmux emacs less unzip htop iftop iotop ca-certificates openssh-client openssh-server rsync iputils-ping net-tools sudo llvm-dev re2c
sudo apt-get install -y tesseract-ocr
add-apt-repository ppa:git-core/ppa -y
apt-get install -y git libnuma-dev wget
pip install pipx
pipx install nvitop
pipx ensurepath 
. ~/.bashrc
# Configure SSH for password and public key authentication
cp /etc/ssh/sshd_config /etc/ssh/sshd_config.bak
sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config
sed -i 's/#PubkeyAuthentication yes/PubkeyAuthentication yes/' /etc/ssh/sshd_config
sed -i 's/^\(\s*\)GSSAPIAuthentication yes/\1GSSAPIAuthentication no/' /etc/ssh/ssh_config
sed -i "s/^#Port 22/Port ${SSH_PORT}/" /etc/ssh/sshd_config
cat /root/.ssh/id_rsa.pub >> /root/.ssh/authorized_keys
cat /root/.ssh/id_rsa.pub >> /root/.ssh/authorized_keys2
chmod 600 /root/.ssh/authorized_keys
chmod 600 /root/.ssh/authorized_keys2
mkdir /var/run/sshd
echo "root:${ROOT_PASSWD}" | chpasswd
mkdir -p ~/.pip
# install miniconda
wget -qO- https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -p /opt/conda 
rm /tmp/miniconda.sh 
ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh
echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc 
. /opt/conda/etc/profile.d/conda.sh 
conda init bash
conda config --set show_channel_urls true
# 配置 .condarc 文件
cat <<EOF > ~/.condarc
channels:
  - conda-forge
  - bioconda
  - pytorch
  - pytorch-nightly
  - nvidia
  - defaults
show_channel_urls: true
EOF
EOT

ARG BRANCH="main"
ENV BRANCH=${BRANCH}
ARG CONDA_ENV_NAME="llmocr"
ENV CONDA_ENV_NAME=${CONDA_ENV_NAME}
ARG PYTHON_VERSION="3.12"
ENV PYTHON_VERSION=${PYTHON_VERSION}
# https://github.com/opendatalab/PDF-Extract-Kit
RUN <<EOT
#!/bin/bash
source /opt/conda/etc/profile.d/conda.sh
conda create -n ${CONDA_ENV_NAME} python=${PYTHON_VERSION} -y
conda activate ${CONDA_ENV_NAME}
git clone https://github.com/Dicklesworthstone/llm_aided_ocr
git checkout ${BRANCH}
cd llm_aided_ocr
python -m pip install --upgrade pip
python -m pip install wheel
python -m pip install --upgrade setuptools wheel
pip install -r requirements.txt
EOT

CMD ["/usr/sbin/sshd", "-D"]