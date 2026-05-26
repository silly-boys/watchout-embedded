#!/bin/bash
set -e

echo "[1/3] 시스템 패키지 설치 중..."
sudo apt-get update -qq
sudo apt-get install -y \
  libavformat-dev libavcodec-dev libavdevice-dev \
  libavutil-dev libswscale-dev libswresample-dev \
  libopus-dev libvpx-dev \
  pkg-config python3-dev python3-venv

echo "[2/3] 가상환경 생성 중..."
python3 -m venv venv --system-site-packages
source venv/bin/activate

echo "[3/3] Python 패키지 설치 중..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "완료! 실행 방법:"
echo "  ./start.sh"
