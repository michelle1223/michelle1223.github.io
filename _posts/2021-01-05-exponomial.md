---
layout: post
title: The Exponomial Choice Model - A New Alternative for Assortment and Price Optimization
category: Papers
tag: Statistics
---

이번 글에서 정리하고자 하는 논문은 Discrete Choice Model의 review paper인 Berbeglia Review Paper에서 마르코프 체인 모델 (Markov Chain model) 과 함께 좋은 성능을 보인 모델로 소개된 Exponomial model의 2014년 논문이다.  

## Discrete Choice Models
Exponomial model에 대해서 설명하기 위해서는, 먼저 Discrete Choice Model에 대한 설명이 필요하다. 말 그대로 discrete한 선택 (choice) 를 모델링한 것인데, 예를 들면 쇼핑몰에서 여러 가지 특징과 가격을 지닌 제품들이 주어졌을 때, 그 중 a라는 제품을 선택하는 경우를 말하는 것이다. 이렇게 어떤 제품을 선택하거나, 혹은 어떤 제품군을 선택할 때, 이 선택의 확률을 모델링할 수 있는데, 이러한 모델을 discrete choice model이라고 한다. 이때, 어떤 확률 모형을 사용하는 지에 따라 모델의 종류를 나눌 수 있는데, 대표적으로 logistic regression을 사용하는 multionomial logit이나 nested logit과 같은 모델들이 있다. 이 모델들의 경우, 30년도 전에 제안되어 discrete choice model 분야의 조상님(?) 같은 존재가 되었다. 지금도 여전히 사용되는 모델이지만, 그 단점이 분명하여 이를 극복하기 위해 여러 새로운 모델들이 제안되고 있다. 그 중 하나가 이 글에서 소개하고자 하는 exponomial model이다.  

**수정 중 (To Be Updated)**  
