---
layout: post
title: Making a ChatBot with seq2seq - 1.이론편
category: NLP
tag: NLP
---

딥러닝으로 챗봇을 만드는 방법에는 여러 가지가 있다. 그 중 상대적으로 간단한 방법이 seq2seq 모델을 이용하는 방법이다. seq2seq 모델은 sequence-to-sequence 모델로 풀어 쓸 수 있다. 단어의 시퀀스, 즉 문장을 입력으로 받아서 바로 문장이 출력되도록 하는 방식이다.  이 모델은 주로 챗봇이나 번역기 구현을 위해 쓰인다.  
seq2seq 모델은 Encoder와 Decoder로 이루어져 있다. 그리고 이 Encoder와 Decoder는 각각의 RNN 구조를 지니고 있다. Encoder는 입력 문장의 모든 단어들을 순차적으로 입력 받은 후 이 모든 단어 정보들을 압축해서 하나의 벡터로 만드는데, 이 벡터를 Context Vector라고 한다. (모든 단어 정보를 하나의 벡터로 압축하다 보니, 보통 이 Context Vector는 수 백 차원 이상이 된다.) 그 후 이 Context Vector는 Decoder로 전송된다.  
이렇게만 보면 구조는 매우 간단해보인다! 하지만 이게 끝은 아니다. 앞에서도 말했다시피, Encoder와 Decoder는 두 개의 RNN 구조이다. 그 내부를 살펴보면 다음 그림과 같다.  

[seq2seq 구조 그림]

위 그림에서는 Encoder와 Decoder가 LSTM 셀들로 가정되어 있다. 성능의 문제로 인해 보통은 바닐라 RNN이 아닌 LSTM이나 GRU의 구조를 갖게 되기 때문이다.  

우선 Encoder를 살펴 보면, 입력 문장은 토큰화를 통해 단어 단위로 쪼개지고 각 단어 토큰이 RNN 셀의 각 시점의 입력이 된다. 이렇게 모든 단어를 입력받은 후에 Encoder RNN 셀의 마지막 시점의 은닉 상태(hidden state)를 Context Vector로 만들어 Decoder에 넘겨준다. 그러면 이 Context Vector는 Decoder RNN 셀의 첫 번째 은닉 상태로 사용된다.  
Decoder에 Context Vector가 입력되면, Decoder는 이 벡터와 <sos>(문장을 시작한다는 뜻의 입력값)을 이용하여 다음에 등장할 확률이 가장 높은 단어를 예측하고 그 예측된 단어를 다음 시점의 RNN셀의 입력으로 사용한다. 이 과정에서 출력 단어(즉 예측된 단어)로 나올 수 있는 단어는 무궁무진하다. seq2seq에서는 선택 가능한 모든 단어들로부터 하나의 단어를 골라서 예측할 때, 소프트맥스 함수를 사용한다.

[소프트맥스 함수 그래프]

각 시점의 출력 벡터는 소프트맥스 함수를 통해 각 단어별 확률값으로 변하고, 이 값을 바탕으로 출력 단어가 결정된다. 이러한 과정을 반복하여 문장의 마지막 단어까지 예측하고, 완성된 문장과 함께 <eos>(문장이 끝났다는 뜻의 심볼)을 출력해낸다.  
여기까지가 seq2seq 모델의 기본적 구조이다. Context Vector를 어떻게 사용할 지, 얼마나 자주 입력으로 사용할 지(ex. 매 시점마다 Context Vector가 하나의 입력으로 또 사용되도록 설정할 수 있다. 이렇게 하면 매 시점마다 앞 뒤 문맥을 더욱 반영하여 단어를 예측한다.)에 따라 세부적인 구조는 더 복잡해질 수 있다. 하지만 전체적인 구조는 위의 설명에서 크게 벗어나지 않을 것이다.  

[ChatBot Github Link](https://github.com/golbin/TensorFlow-Tutorials/tree/master/10%20-%20RNN/ChatBot)
