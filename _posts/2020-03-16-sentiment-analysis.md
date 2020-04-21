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

```python
import numpy as np
import pandas as pd
import re
import konlpy

# 크롤링한 영화 리뷰들을 하나의 dataframe으로 합치고 불필요한 column 제거.
# 각각의 영화 리뷰 dataframe들 10개의 이름은 영어 제목대로 만듦 (과정 생략)
movie = pd.concat([littleforest, handmaiden, murder, yourwedding, twosisters, busan, sunny, haunters, god, hen])
movie = movie.drop('Unnamed: 0', axis=1)
movie
```

### Tokenization  
```python
from ckonlpy.tag import Twitter
twitter = Twitter()
```
여기서 토큰화를 할 때는, stopwords라는 library를 만들어서 처리해줄 불용어들을 지정한다. 사람들이 많이 쓰는 불용어 사전을 인터넷에서 검색해서 사용하는 방법도 있지만, 일단은 간단한 불용어 사전을 만들어서 stopwords로 지정해주었다.  

```python
stopwords = ['근데','의','가','이','은','들','는','좀','걍','과','도','를','으로','자','에','와','한','하다','관람객', 'ㅋ', 'ㅎ', 'ㅠ', '다', '더', '싶다', '지금', '하지만', '네', '요', '다가', '해서']
```
stopwords에 '관람객'이 들어간 이유는, 네이버 영화의 댓글 리뷰는 리뷰 작성자가 관람객인지 아닌지 여부를 따로 보여주기 때문에 크롤링을 할 때 관람객이 작성한 리뷰의 경우 '관람객'이라는 단어가 리뷰 맨 앞에 항상 들어갔기 때문이다. 이 점을 이용해서 관람객의 리뷰에 가중치를 더 주고 분석에 중요한 리뷰와 중요하지 않은 리뷰를 구분할 수도 있겠지만, 이번 프로젝트에서는 이러한 과정을 생략하였다.  

```python
token = []
for sentence in movie['review']:
    temp_X = []
    temp_X = twitter.morphs(sentence, stem=True) # 토큰화
    temp_X = [word for word in temp_X if not word in stopwords] # 불용어 제거
    token.append(temp_X)
movie['token'] = token
```
이와 같이 토큰화를 끝낸 후 movie 데이터에 token column을 붙여준다.  

### Retrieve train data  
추후 설명할 모델링 과정이 모두 끝난 후 정확도를 산출해보니, 앞서 크롤링한 데이터만을 사용하여 모델을 구축할 경우, 만족스럽지 않은 정확도(약 65%)가 산출됨이 확인되었다. 이는 모델 학습에 쓰이는 훈련 데이터가 부족하기 때문으로 결론을 내려서, 외부에서 train data로 사용할 수 있는 한국어 영화 리뷰 데이터를 불러와서 모델을 학습시키는 데 사용하였다.  

```python
import urlib.request
urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt", filename="ratings_train.txt")
urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt", filename="ratings_test.txt")
train_data = pd.read_csv('ratings_train.txt',sep='\t',error_bad_lines=False)
```
이렇게 불러온 훈련 데이터에서 null 값이 있는지 확인하고, 전처리 및 토큰화를 거친 후 X_train라고 저장하여 사용하였다. 이 데이터의 경우 각 문장의 긍정/부정 여부가 label로 표시되어 있는데, 이 label column은 이후 y_train으로 저장하여 사용하였다. 그리고 위에서 크롤링하여 모은 데이터는 X_test로 저장하여 쓰였다.   

### 정수 인코딩  
토큰화된 데이터를 모델에 돌리기 위해서는 keras에서 제공하는 Tokenizer, pad_sequences 함수를 이용하여 정수로 이루어진 벡터로 바꿔줘야 한다. 이 과정을 정수 인코딩이라고 한다.  

```python
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

max_words = 35000
tokenizer = Tokenizer(num_words=max_words) # 상위 35,000개의 단어만 보존
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

max_len = 30
# 전체 데이터의 길이는 30으로 맞춘다.
X_train = pad_sequences(X_train, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)
y_train = np.array(train_data['label'])
```

### Modeling  
모델링은 LSTM으로 진행하였다. LSTM은 자연어처리에서 자주 쓰이는 모델로, RNN을 한 단계 발전시켜 단점을 보완한 모델이다.
모델에 대한 설명:  
1. 긍정/부정의 이진 분류를 수행하기 위해 시그모이드 함수와 binary_crossentropy 사용  
2. 에포크 4회 수행  
3. train data의 20%를 검증 데이터로 사용하여 모델의 정확도 계산  
4. 최종 정확도: 88.32%  

```python
from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.models import Sequential

model = Sequential()
model.add(Embedding(max_words, 100))
model.add(LSTM(128))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(X_train, y_train, epochs=4, batch_size=60, validation_split=0.2)
```

### Calculate accuracy  
movie 데이터에 'label' column을 추가해서, 이 column을 y_test라 이름 짓고 정확도를 산출한 것이다.  
이렇게 한 이유는, 모델이 판단한 긍정/부정 여부와 평점을 토대로 한 긍정/부정 여부를 비교하기 위함이다. 이때, 대부분의 평점들이 다소 상향 평준화되어 있기 때문에 "1~5점 부정 / 6~10점 긍정"으로 판단하지 않고, 6점 이하부터 부정적 리뷰라고 설정하였다.  

```python
movie['label'] = ""
#row index number를 초기화하지 않으면 겹치는 index number가 10개씩 생김.(10개의 df를 concat했기 때문)
movie.reset_index(drop=True, inplace=True)

#평점 5점, 6점, 7점 리뷰를 비교한 후 평점 6점 이하의 리뷰를 부정 리뷰라고 정의함. (점수의 상향평준화 때문)
for i in range(0,10000):
    if movie.loc[i,'score'] > 6 :
        movie.loc[i,'label'] = 1
    else: movie.loc[i,'label'] = 0

y_test = np.array(movie['label'])
print("\n 테스트 정확도: %.4f" % (model.evaluate(X_test, y_test)[1]))
```

이렇게 정확도를 산출한 결과, 85.65% 의 정확도가 나왔다.  
정확도를 높이기 위해서는, LSTM대신 ELMo 모델을 사용하는 것도 방법이다. ELMo 모델의 경우, 에포크를 1회만 수행해도 80%라는 꽤 높은 정확도가 나온다. ELMo나 BERT 등 다른 최신 모델을 사용했을 때의 결과도 이후 추가하여 결과를 비교해보겠다.  
