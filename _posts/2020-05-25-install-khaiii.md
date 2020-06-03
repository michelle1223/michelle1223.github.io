---
layout: post
title: How to install khaiii on python
category: NLP
tag: python
---

# What is khaiii?  
khaiii: 카카오에서 개발한 형태소 분석기. 흔히 konlpy 라이브러리에 있는 komoran, twitter(현 Okt) 등을 사용하지만 이번에 카카오 아레나 주관 Melon Playlist Continuation 대회에 참가하면서 강제로(?) 사용하게 되었다. (카카오에서 개최한 대회여서 그런지 text tokenizer 라이브러리로 khaiii만 허용했다.) khaiii는 처음 사용해서 설치부터 차근차근 진행하게 되었는데, 생각보다 에러도 많이 떴고, khaiii 사용자와 맥 사용자가 워낙 적어서 참고할 만한 게시글이 많이 없어서 메모해놓는다. 곧 새 맥북이 오기 때문에...조만간 또 설치해야 한다. 그땐 어려움 없이 설치할 수 있기를...  

## 1. Install Xcode  
khaiii 설치 과정에서 cmake을 사용해야 하는데, 이때 맥의 경우 이런 에러가 나온다.  

```python
xcrun: error: invalid active developer path (/Library/Developer/CommandLineTools), missing xcrun at: /Library/Developer/CommandLineTools/usr/bin/xcrun  
```

이건 hunter나 cmake의 문제가 아니라 맥 자체의 developer path 지정의 문제인 것 같다.  

```python
xcode-select --install
```

이 코드를 터미널에 입력해서 해결해주자.  


## 2. Clone khaiii  
```python
git clone https://github.com/kakao/khaiii.git
```

## 3. Install & bind with python

```python
cd khaiii  #khaiii 폴더로 이동 (cd = change directory)
mkdir build  #khaiii폴더 아래에 build 폴더 만들기 (mkdir = make directory)
cd build  #build 폴더로 이동
sudo cmake ..  #약 10분 소요
sudo make all  #빌드 실행. 약 5분 소요
sudo make resource  #리소스 빌드
sudo make install  #khaiii (드디어) 설치
sudo make package_python  #python과 바인딩
cd package_python
sudo pip3 install . #마지막 점 찍는 것 주의! 약 5분 소요.
```
