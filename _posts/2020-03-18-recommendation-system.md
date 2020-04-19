---
layout: post
title: Recommendation System Part 1
category: Data Science
tag: python
---

# Intro & Video  
추천 시스템은 내가 데이터 사이언스에 관심을 갖게 된 가장 큰 계기였다. 하지만 추천 시스템 쪽을 일개 학생이 공부하기 쉽지 않은데, 그 이유 중 하나는 많은 사용자에 대한 방대한 데이터가 있어야 이에 관한 추천 시스템 구축을 시도해볼 수 있기 때문이다. 이러한 방대한 데이터는 보통 찾기 힘들다. 그리고 한국 사용자 데이터는 더더욱 찾기 힘들다. 그래서 나 또한 이쪽을 건드려보지 못하다가, 유튜브에서 우연히 관련 강의가 올라와 있는 것을 보게 되었고, 코드와 데이터셋까지 제공되어 있어서 이 강의를 따라가면서 공부를 시작해보게 되었다.  

{% include video.html id="kiInh5STnyQ" title="Recommendation System" %}   

해당 강의는 (많은 데이터 사이언스 강의들이 그렇듯이...) 영어로 진행되었고, 코드에 대한 설명 또한 영어로 되어 있다. 이를 적절히 번역해서 올려보겠다.  

# About the Dataset  
이 강의에 사용된 데이터셋은 미네소타 대학교 컴퓨터공학과 연구 그룹인 GroupLens에서 제공된 데이터셋으로, MovieLens라는 영화 추천 서비스에서 5점 만점의 rating을 토대로 9742개의 영화에 대한, 610명의 유저들이 만든 100836개의 리뷰를 모은 데이터셋이다.  

# 2 Ways to Make a Recommendation System  
추천 시스템 구축에서 사용하는 method는 보통 두 가지가 있는데, Collaborative Filtering과 Content Based Filtering이 그 두 가지이다. 전자는 사용자 평점 기반, 후자는 컨텐츠 자체의 특성을 기반으로 하여 추천하는 방식이다. 이 강의에서는 Collaborative Filtering, 한국어로는 '협업 필터링' 기법을 사용하여 모델을 구축하였다.  

# Coding  

1. Import Dataset

'''python
import numpy as np
import pandas as pd
import
'''

**미완**  

...때문에 추천 시스템에 대해서는 꾸준히, 새로운 데이터셋을 발견하는 대로 프로젝트 형태로 올려볼 생각이다. 다음에는 Kaggle에 올라와있는 유튜브 데이터셋을 가지고 무언가를 해볼 생각이다. (추천 시스템 카테고리를 따로 개설할까도 생각 중이다. 더 많은 포스트를 올리게 되면 카테고리도 따로 생길 것 같다.)
