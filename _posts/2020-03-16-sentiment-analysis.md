---
layout: post
title: Sentiment Analysis
category: NLP
tag: NLP
---

# 1. What is Sentiment Analysis?  
Sentiment Analysis, 한국어로 '감성 분석'은 어떤 텍스트 데이터가 긍정적인 의미를 담고 있는지, 부정적인 의미를 담고 있는지 판단하는 분류 방법이다. 감성 분석은 영화나 음악 리뷰, 식당 평가, 제품 리뷰, 뉴스 데이터 등 다양한 분야에서 사용될 수 있다. 해당 프로젝트에서는 영화 리뷰를 선택하여 감성 분석을 진행해보았다.  

# 2. Coding  
아래 사용된 데이터는 한국 영화 10편(리틀 포레스트, 아가씨, 살인의 추억, 너의 결혼식, 장화 홍련, 부산행, 써니, 초능력자, 신과 함께, 마당을 나온 암탉)에 대한 네이버 영화 리뷰를 1000개씩 크롤링해서 1만 개의 row를 지닌 데이터프레임 형식으로 만든 것이다. 크롤링과 데이터 전처리에 관한 코드는 [Data Science]라는 카테고리로 따로 포스팅할 예정이니 이에 대한 것은 그 게시물을 참고하길 바란다.  

'''python
import numpy as np
import pandas as pd
import re
import konlpy
'''

**미완**  
