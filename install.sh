#!/bin/bash

# AI 챗봇 프로젝트 설치 스크립트

echo "AI 제품 추천 챗봇 설치를 시작합니다..."

# Python 버전 확인
python_version=$(python3 --version 2>/dev/null)
if [ $? -eq 0 ]; then
    echo "[OK] Python 확인: $python_version"
else
    echo "[ERROR] Python3가 설치되어 있지 않습니다."
    echo "Python 3.7 이상을 설치해주세요."
    exit 1
fi

# pip 업그레이드
echo "pip 업그레이드 중..."
python3 -m pip install --upgrade pip

# 필요한 패키지 설치
echo "필요한 패키지들을 설치합니다..."
python3 -m pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "[OK] 패키지 설치 완료!"
else
    echo "[ERROR] 패키지 설치 중 오류가 발생했습니다."
    exit 1
fi

# .env 파일 생성 (없는 경우)
if [ ! -f ".env" ]; then
    echo ".env 파일을 생성합니다..."
    cp .env.example .env
    echo "주의: .env 파일에 실제 OpenAI API 키를 입력해주세요!"
else
    echo "[OK] .env 파일이 이미 존재합니다."
fi

echo ""
echo "설치가 완료되었습니다!"
echo ""
echo "다음 단계를 진행하세요:"
echo "1. OpenAI API 키를 발급받으세요: https://platform.openai.com/api-keys"
echo "2. .env 파일을 편집하여 API 키를 입력하세요"
echo "3. 챗봇을 실행하세요: python3 run.py"
echo ""
