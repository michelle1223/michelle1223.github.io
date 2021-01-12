---
layout: post
title: The Exponomial Choice Model - A New Alternative for Assortment and Price Optimization (1)
category: Papers
tag: Statistics
---

이번 글에서 정리하고자 하는 논문은 Discrete Choice Model의 review paper인 Berbeglia Review Paper에서 마르코프 체인 모델 (Markov Chain model) 과 함께 좋은 성능을 보인 모델로 소개된 Exponomial model의 2014년 논문이다.  

## Discrete Choice Models
Exponomial model에 대해서 설명하기 위해서는, 먼저 Discrete Choice Model에 대한 설명이 필요하다. 말 그대로 discrete한 선택 (choice) 를 모델링한 것인데, 예를 들면 쇼핑몰에서 여러 가지 특징과 가격을 지닌 제품들이 주어졌을 때, 그 중 a라는 제품을 선택하는 경우를 말하는 것이다. 이렇게 어떤 제품을 선택하거나, 혹은 어떤 제품군을 선택할 때, 이 선택의 확률을 모델링할 수 있는데, 이러한 모델을 discrete choice model이라고 한다. 이때, 어떤 확률 모형을 사용하는 지에 따라 모델의 종류를 나눌 수 있는데, 대표적으로 logistic regression의 형태를 사용하는 multinomial logit이나 nested logit과 같은 모델들이 있다. 이 모델들의 경우, 30년도 전에 제안되어 discrete choice model 분야의 조상님(?) 같은 존재가 되었다. 지금도 여전히 사용되는 모델이지만, 그 단점이 분명하여 이를 극복하기 위해 여러 새로운 모델들이 제안되고 있다. 그 중 하나가 이 글에서 소개하고자 하는 exponomial model이다.  
어떤 분야든, 새로운 모델이 나온다는 것은 이전 모델에 어떠한 '결점'이 있었다는 것이고, 그 결점을 극복하기 위해 새로운 모델이 제안된다. Exponomial 모델의 연구진 또한 이전에 제안된 multinomial logit이나 nested logit 등의 모델들의 결점을 지적하며 Exponomial 모델을 소개한다. 그 결점은 바로 logit 모델의 가정 사항에 있다.  

## Skewness of WtP
Multinomial Logit (MNL) 모델이나 Nested Logit (NL) 모델에서는 소비자의 willingness to pay (WtP; 소비자가 어떤 제품을 사는데 지불할 최대 금액) 의 분포가 positively skewed (right-skewed) 되어 있다고 가정한다. 이는 logit 모델의 error term 이 Gumbel distribution을 따른다고 가정하기 때문인데, 이러한 가정 사항이 적절한 상황도 있지만, 현대 사회에서는 현실과 이 가정 사항이 맞지 않을 경우가 매우 많다. 인터넷의 발달로 인해 대부분의 제품에 대한 정보가 소비자들에게 풍부하게 제공되고 있는 상황에서, 소비자들의 WtP는 오히려 negatively skewed distribution을 따를 것이라고 가정하는 것이 더 적절하다. 왜냐하면, 여러 제품의 정보를 수집한다면, 그 정보를 바탕으로 특정한 가격 p 이상은 지불하고자 하는 willingness가 급격히 떨어질 것이기 때문이다. 예를 들어, 어떤 소비자가 자동차를 구매하고자 하는 상황을 생각해보자. 자동차의 경우, 각 브랜드의 차량 가격 및 성능이 인터넷 등에 풍부하게 공유되어 있다. 그렇다면, 어떤 자동차의 기준 가격 (benchmark price) 가 p라고 했을 때, p보다 10~20% 높은 가격에 구매하고자 할 확률은 p보다 10~20% 낮은 가격에 구매하고자 할 확률보다 훨씬 작을 것이다. 즉, p를 기준으로, p 이상의 가격에서는 소비자의 WtP가 급격히 감소할 것이다. 이를 고려하면 WtP의 분포는 negatively skewed 인 것이 더 적절하다고 할 수 있다.  

**Plot of WtP**  

이러한 점을 고려하여, 위 논문에서는 negatively skewed 된 consumer utility 분포를 바탕으로 만들어진 새로운 discrete choice model을 제안하고자 한다. 그리고 이 새로운 모델은 exponomial, 즉 exponential term의 선형 결합 (linear function) 형태로 소비자가 특정 제품을 선택할 확률을 모델링하기 때문에 "Exponomial Model"이라고 부른다.  

## Exponomial Choice Model  
기존의 discrete choice model에서는 보통 특정 제품 (product i) 이 주는 utility를 아래와 같은 공식으로 나타낸다.  

**U_i = u_i + e_i (u_i: nominal utility, e_i: error term. unobservable/not encoded but affects utility)**  

위 논문에서는 이 식을 조금 변형하여 아래와 같이 소비자 k가 i라는 선택을 할 때의 utility를 나타낸다.  

**U_k(i) = u_i - z_ik (u_i: ideal utility, z_ik: iid exponential random variables with mean 1/lambda)**  

이때, z_ik는 random term으로서 소비자의 개인적인 성향 (personal preference) 을 반영할 수 있도록 하는 확률 변수이다.  
이와 같이 모델링된 utility는 consumer utility의 분포가 negatively skewed 되도록 하는 결과를 낳는다. 여기서 ideal utility를 나타내는 u_i는, 제품의 특성, 소비자의 특성, 그리고 기타 환경/맥락에 따른 특성 (product, environmental, consumer attributes) 등에 대한 function일 것이라고 생각할 수 있다. 그러나 우선 위 연구에서는 product level attributes만을 고려하고, consumer heterogeneity는 새로운 random variable로 나타낼 수 있도록 위 식을 아래와 같이 변형한다.   

**U_k(i) = (u_i + d_k) - z_ik (d_k ~ N(0, 0.5), iid)**  

이와 같이 utility를 모델링했을 때, U_k(i)의 density는 다음과 같이 plot 할 수 있다.  

**Plot from page 7**  

여기서 더 나아가, ideal utility를 다음과 같이 각 제품의 가격을 이용해 나타낼 수 있고, 이를 이용하여 소비자 k의 제품 i에 대한 WtP를 아래와 같이 모델링할 수 있다. utility를 그대로 사용하지 않고 WtP에 대한 식으로 변형하는 이유는, 현실에서 utility는 관찰하기 힘들지만, WtP는 상대적으로 실제 데이터를 가지고 observe하기 쉽기 때문이다. (즉 observability의 관점에서 더 선호되기 때문이다)  

**Page 7, u_i의 식과 WtP의 식**  

연구진이 실제 중국 난징의 소비자들의 유전자 변형 콩기름 (GM soybean oil) 제품에 대한 WtP 데이터를 이용하여 (Hu et al. 2006) Multinomial Logit 모델과 Exponomial Choice 모델 중 어느 것이 실제 데이터의 cdf를 더 잘 나타내는지 시험해봤는데, 아래의 plot 을 보면 Exponomial 모델이 실제 데이터 (empirical 이라고 라벨링 되어 있음) 의 cdf와 훨씬 근접하게 모델링하는 데 성공하였음을 알 수 있다.  

**Page 8, plot of CDF**  

위의 경우처럼 실제 데이터에 Exponomial Choice Model을 적용하여 parameter estimation을 진행한 사례를 더 구체적으로 살펴보고자 한다면, 같은 저자 Alptekinoglu의 [논문](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3232788) 을 살펴보면 좋을 것이다. 
