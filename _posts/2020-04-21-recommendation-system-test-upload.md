---
layout: post
title: Recommendation System Part 1 test upload
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

**[Process]**  
1. Get a dataset of movie ratings and understand how the dataset is structured.  
2. Try to get just a non-personalized set of recommendations for John-Green-bot and me.  
3. Get personalized ratings for John-Green-bot and me, and import them into the system in the correct format.  
4. Train a User-User collaborative filtering model to provide personalized recommendations based on John-Green-bot's and my prior ratings.  
5. Combine ratings to generate a single ranked recommendation list.

## 1. Data Import


```python
!pip install lenskit
!pip install -U numba
```

    Collecting lenskit
      Downloading lenskit-0.8.4.zip (79 kB)
    [K     |████████████████████████████████| 79 kB 296 kB/s eta 0:00:011
    [?25hRequirement already satisfied: pandas>=0.24 in /Users/Dongeun_Min/anaconda3/lib/python3.7/site-packages (from lenskit) (0.24.2)
    Requirement already satisfied: numpy in /Users/Dongeun_Min/anaconda3/lib/python3.7/site-packages (from lenskit) (1.16.2)
    Requirement already satisfied: scipy in /Users/Dongeun_Min/anaconda3/lib/python3.7/site-packages (from lenskit) (1.2.1)
    Requirement already satisfied: numba<0.48,>=0.43 in /Users/Dongeun_Min/anaconda3/lib/python3.7/site-packages (from lenskit) (0.43.1)
    Collecting pyarrow
      Downloading pyarrow-0.16.0-cp37-cp37m-macosx_10_9_intel.whl (40.5 MB)
    [K     |████████████████████████████████| 40.5 MB 37 kB/s  eta 0:00:01     |█████████████████▉              | 22.5 MB 2.1 MB/s eta 0:00:09     |█████████████████████████████▉  | 37.7 MB 2.3 MB/s eta 0:00:02     |██████████████████████████████  | 38.1 MB 2.3 MB/s eta 0:00:02
    [?25hRequirement already satisfied: cffi in /Users/Dongeun_Min/anaconda3/lib/python3.7/site-packages (from lenskit) (1.12.2)
    Requirement already satisfied: joblib in /Users/Dongeun_Min/anaconda3/lib/python3.7/site-packages (from lenskit) (0.13.2)
    Requirement already satisfied: pytz>=2011k in /Users/Dongeun_Min/anaconda3/lib/python3.7/site-packages (from pandas>=0.24->lenskit) (2018.9)
    Requirement already satisfied: python-dateutil>=2.5.0 in /Users/Dongeun_Min/anaconda3/lib/python3.7/site-packages (from pandas>=0.24->lenskit) (2.8.0)
    Requirement already satisfied: llvmlite>=0.28.0dev0 in /Users/Dongeun_Min/anaconda3/lib/python3.7/site-packages (from numba<0.48,>=0.43->lenskit) (0.28.0)
    Requirement already satisfied: six>=1.0.0 in /Users/Dongeun_Min/anaconda3/lib/python3.7/site-packages (from pyarrow->lenskit) (1.12.0)
    Requirement already satisfied: pycparser in /Users/Dongeun_Min/anaconda3/lib/python3.7/site-packages (from cffi->lenskit) (2.19)
    Building wheels for collected packages: lenskit
      Building wheel for lenskit (setup.py) ... [?25ldone
    [?25h  Created wheel for lenskit: filename=lenskit-0.8.4-py3-none-any.whl size=74346 sha256=f8715eea1a55516e730feda6acd99c7926ce82c1fbc451ad7a672875e18b76d4
      Stored in directory: /Users/Dongeun_Min/Library/Caches/pip/wheels/89/12/ce/b72e165030d4a7d39e0ac5ac8331ae97e767074acb1c87a135
    Successfully built lenskit
    Installing collected packages: pyarrow, lenskit
    Successfully installed lenskit-0.8.4 pyarrow-0.16.0
    Collecting numba
      Downloading numba-0.48.0-cp37-cp37m-macosx_10_9_x86_64.whl (2.0 MB)
    [K     |████████████████████████████████| 2.0 MB 227 kB/s eta 0:00:01
    [?25hRequirement already satisfied, skipping upgrade: setuptools in /Users/Dongeun_Min/anaconda3/lib/python3.7/site-packages (from numba) (40.8.0)
    Requirement already satisfied, skipping upgrade: numpy>=1.15 in /Users/Dongeun_Min/anaconda3/lib/python3.7/site-packages (from numba) (1.16.2)
    Collecting llvmlite<0.32.0,>=0.31.0dev0
      Downloading llvmlite-0.31.0-cp37-cp37m-macosx_10_9_x86_64.whl (15.9 MB)
    [K     |████████████████████████████████| 15.9 MB 15.8 MB/s eta 0:00:01
    [31mERROR: lenskit 0.8.4 has requirement numba<0.48,>=0.43, but you'll have numba 0.48.0 which is incompatible.[0m
    [?25hInstalling collected packages: llvmlite, numba
      Attempting uninstall: llvmlite
        Found existing installation: llvmlite 0.28.0
    [31mERROR: Cannot uninstall 'llvmlite'. It is a distutils installed project and thus we cannot accurately determine which files belong to it which would lead to only a partial uninstall.[0m



```python
import lenskit.datasets as ds
import pandas as pd
import numpy as np
```


```python
!git clone https://github.com/crash-course-ai/lab4-recommender-systems.git
data = ds.MovieLens('lab4-recommender-systems/')
print("Successfully installed dataset.")
```

    'lab4-recommender-systems'에 복제합니다...
    remote: Enumerating objects: 25, done.[K
    remote: Counting objects: 100% (25/25), done.[K
    remote: Compressing objects: 100% (25/25), done.[K
    remote: Total 25 (delta 12), reused 0 (delta 0), pack-reused 0[K
    오브젝트 묶음 푸는 중: 100% (25/25), 981.92 KiB | 1016.00 KiB/s, 완료.
    Successfully installed dataset.



```python
data.movies.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>genres</th>
    </tr>
    <tr>
      <th>item</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Toy Story (1995)</td>
      <td>Adventure|Animation|Children|Comedy|Fantasy</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Jumanji (1995)</td>
      <td>Adventure|Children|Fantasy</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Grumpier Old Men (1995)</td>
      <td>Comedy|Romance</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Waiting to Exhale (1995)</td>
      <td>Comedy|Drama|Romance</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Father of the Bride Part II (1995)</td>
      <td>Comedy</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.ratings.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user</th>
      <th>item</th>
      <th>rating</th>
      <th>timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>4.0</td>
      <td>964982703</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>3</td>
      <td>4.0</td>
      <td>964981247</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>6</td>
      <td>4.0</td>
      <td>964982224</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>47</td>
      <td>5.0</td>
      <td>964983815</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>50</td>
      <td>5.0</td>
      <td>964982931</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.tags.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user</th>
      <th>item</th>
      <th>tag</th>
      <th>timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>60756</td>
      <td>funny</td>
      <td>1445714994</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>60756</td>
      <td>Highly quotable</td>
      <td>1445714996</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>60756</td>
      <td>will ferrell</td>
      <td>1445714992</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>89774</td>
      <td>Boxing story</td>
      <td>1445715207</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>89774</td>
      <td>MMA</td>
      <td>1445715200</td>
    </tr>
  </tbody>
</table>
</div>



* tags might need NLP


```python
movie_data = data.ratings.join(data.movies['genres'], on='item')
movie_data = movie_data.join(data.movies['title'], on='item')
movie_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user</th>
      <th>item</th>
      <th>rating</th>
      <th>timestamp</th>
      <th>genres</th>
      <th>title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>4.0</td>
      <td>964982703</td>
      <td>Adventure|Animation|Children|Comedy|Fantasy</td>
      <td>Toy Story (1995)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>3</td>
      <td>4.0</td>
      <td>964981247</td>
      <td>Comedy|Romance</td>
      <td>Grumpier Old Men (1995)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>6</td>
      <td>4.0</td>
      <td>964982224</td>
      <td>Action|Crime|Thriller</td>
      <td>Heat (1995)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>47</td>
      <td>5.0</td>
      <td>964983815</td>
      <td>Mystery|Thriller</td>
      <td>Seven (a.k.a. Se7en) (1995)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>50</td>
      <td>5.0</td>
      <td>964982931</td>
      <td>Crime|Mystery|Thriller</td>
      <td>Usual Suspects, The (1995)</td>
    </tr>
  </tbody>
</table>
</div>



* data fully joined!

## 2. 전처리


```python
# highest rated films
average_ratings = (data.ratings).groupby(['item']).mean()
sorted_avg_ratings = average_ratings.sort_values(by="rating", ascending=False)
joined_data = sorted_avg_ratings.join(data.movies['genres'], on='item')
joined_data = joined_data.join(data.movies['title'], on='item')
joined_data = joined_data[joined_data.columns[1:]]

print("RECOMMENDED FOR ANYBODY:")
joined_data.head(10)
```

    RECOMMENDED FOR ANYBODY:





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>rating</th>
      <th>timestamp</th>
      <th>genres</th>
      <th>title</th>
    </tr>
    <tr>
      <th>item</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>88448</th>
      <td>5.0</td>
      <td>1.315438e+09</td>
      <td>Comedy|Drama</td>
      <td>Paper Birds (Pájaros de papel) (2010)</td>
    </tr>
    <tr>
      <th>100556</th>
      <td>5.0</td>
      <td>1.456151e+09</td>
      <td>Documentary</td>
      <td>Act of Killing, The (2012)</td>
    </tr>
    <tr>
      <th>143031</th>
      <td>5.0</td>
      <td>1.520409e+09</td>
      <td>Comedy|Drama|Romance</td>
      <td>Jump In! (2007)</td>
    </tr>
    <tr>
      <th>143511</th>
      <td>5.0</td>
      <td>1.526207e+09</td>
      <td>Documentary</td>
      <td>Human (2015)</td>
    </tr>
    <tr>
      <th>143559</th>
      <td>5.0</td>
      <td>1.520410e+09</td>
      <td>Comedy|Crime|Fantasy</td>
      <td>L.A. Slasher (2015)</td>
    </tr>
    <tr>
      <th>6201</th>
      <td>5.0</td>
      <td>1.100120e+09</td>
      <td>Drama|Romance</td>
      <td>Lady Jane (1986)</td>
    </tr>
    <tr>
      <th>102217</th>
      <td>5.0</td>
      <td>1.443200e+09</td>
      <td>Comedy</td>
      <td>Bill Hicks: Revelations (1993)</td>
    </tr>
    <tr>
      <th>102084</th>
      <td>5.0</td>
      <td>1.493422e+09</td>
      <td>Action|Animation|Fantasy</td>
      <td>Justice League: Doom (2012)</td>
    </tr>
    <tr>
      <th>6192</th>
      <td>5.0</td>
      <td>1.063275e+09</td>
      <td>Romance</td>
      <td>Open Hearts (Elsker dig for evigt) (2002)</td>
    </tr>
    <tr>
      <th>145994</th>
      <td>5.0</td>
      <td>1.526207e+09</td>
      <td>Comedy</td>
      <td>Formula of Love (1984)</td>
    </tr>
  </tbody>
</table>
</div>



* these movies don't look familiar. are these really recommended? let's see what's going on.


```python
!pip install pandas --upgrade
```

    Collecting pandas
      Downloading pandas-1.0.3-cp37-cp37m-macosx_10_9_x86_64.whl (10.0 MB)
    [K     |████████████████████████████████| 10.0 MB 1.2 MB/s eta 0:00:01
    [?25hRequirement already satisfied, skipping upgrade: python-dateutil>=2.6.1 in /Users/Dongeun_Min/anaconda3/lib/python3.7/site-packages (from pandas) (2.8.0)
    Requirement already satisfied, skipping upgrade: pytz>=2017.2 in /Users/Dongeun_Min/anaconda3/lib/python3.7/site-packages (from pandas) (2018.9)
    Requirement already satisfied, skipping upgrade: numpy>=1.13.3 in /Users/Dongeun_Min/anaconda3/lib/python3.7/site-packages (from pandas) (1.16.2)
    Requirement already satisfied, skipping upgrade: six>=1.5 in /Users/Dongeun_Min/anaconda3/lib/python3.7/site-packages (from python-dateutil>=2.6.1->pandas) (1.12.0)
    Installing collected packages: pandas
      Attempting uninstall: pandas
        Found existing installation: pandas 0.24.2
        Uninstalling pandas-0.24.2:
          Successfully uninstalled pandas-0.24.2
    Successfully installed pandas-1.0.3



```python
movie_data.loc[movie_data['title']=='Paper Birds (Pájaros de papel) (2010)']
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user</th>
      <th>item</th>
      <th>rating</th>
      <th>timestamp</th>
      <th>genres</th>
      <th>title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>77875</th>
      <td>483</td>
      <td>88448</td>
      <td>5.0</td>
      <td>1315437602</td>
      <td>Comedy|Drama</td>
      <td>Paper Birds (Pájaros de papel) (2010)</td>
    </tr>
  </tbody>
</table>
</div>



**There is only 1 user that rated for 'Paper Birds'! So, this movie can't be confidently recommended.**  
**What to do: Choose a minimum number of people to include.**


```python
minimum_to_include = 20 #<-- You can try changing this minimum to include movies rated by fewer or more people

average_ratings = (data.ratings).groupby(['item']).mean()
rating_counts = (data.ratings).groupby(['item']).count()
average_ratings = average_ratings.loc[rating_counts['rating'] > minimum_to_include]
sorted_avg_ratings = average_ratings.sort_values(by="rating", ascending=False)
joined_data = sorted_avg_ratings.join(data.movies['genres'], on='item')
joined_data = joined_data.join(data.movies['title'], on='item')
joined_data = joined_data[joined_data.columns[3:]]

print("RECOMMENDED FOR ANYBODY:")
joined_data.head(10)
```

    RECOMMENDED FOR ANYBODY:





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>genres</th>
      <th>title</th>
    </tr>
    <tr>
      <th>item</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>318</th>
      <td>Crime|Drama</td>
      <td>Shawshank Redemption, The (1994)</td>
    </tr>
    <tr>
      <th>922</th>
      <td>Drama|Film-Noir|Romance</td>
      <td>Sunset Blvd. (a.k.a. Sunset Boulevard) (1950)</td>
    </tr>
    <tr>
      <th>898</th>
      <td>Comedy|Drama|Romance</td>
      <td>Philadelphia Story, The (1940)</td>
    </tr>
    <tr>
      <th>475</th>
      <td>Drama</td>
      <td>In the Name of the Father (1993)</td>
    </tr>
    <tr>
      <th>1204</th>
      <td>Adventure|Drama|War</td>
      <td>Lawrence of Arabia (1962)</td>
    </tr>
    <tr>
      <th>246</th>
      <td>Documentary</td>
      <td>Hoop Dreams (1994)</td>
    </tr>
    <tr>
      <th>858</th>
      <td>Crime|Drama</td>
      <td>Godfather, The (1972)</td>
    </tr>
    <tr>
      <th>1235</th>
      <td>Comedy|Drama|Romance</td>
      <td>Harold and Maude (1971)</td>
    </tr>
    <tr>
      <th>168252</th>
      <td>Action|Sci-Fi</td>
      <td>Logan (2017)</td>
    </tr>
    <tr>
      <th>2959</th>
      <td>Action|Crime|Drama|Thriller</td>
      <td>Fight Club (1999)</td>
    </tr>
  </tbody>
</table>
</div>




```python
average_ratings = (data.ratings).groupby(['item']).mean()
rating_counts = (data.ratings).groupby(['item']).count()
average_ratings = average_ratings.loc[rating_counts['rating'] > minimum_to_include]
average_ratings = average_ratings.join(data.movies['genres'], on='item')
average_ratings = average_ratings.loc[average_ratings['genres'].str.contains('Action')]

sorted_avg_ratings = average_ratings.sort_values(by="rating", ascending=False)
joined_data = sorted_avg_ratings.join(data.movies['title'], on='item')
joined_data = joined_data[joined_data.columns[3:]]
print("RECOMMENDED FOR AN ACTION MOVIE FAN:")
joined_data.head(10)
```

    RECOMMENDED FOR AN ACTION MOVIE FAN:





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>genres</th>
      <th>title</th>
    </tr>
    <tr>
      <th>item</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>168252</th>
      <td>Action|Sci-Fi</td>
      <td>Logan (2017)</td>
    </tr>
    <tr>
      <th>2959</th>
      <td>Action|Crime|Drama|Thriller</td>
      <td>Fight Club (1999)</td>
    </tr>
    <tr>
      <th>58559</th>
      <td>Action|Crime|Drama|IMAX</td>
      <td>Dark Knight, The (2008)</td>
    </tr>
    <tr>
      <th>1197</th>
      <td>Action|Adventure|Comedy|Fantasy|Romance</td>
      <td>Princess Bride, The (1987)</td>
    </tr>
    <tr>
      <th>260</th>
      <td>Action|Adventure|Sci-Fi</td>
      <td>Star Wars: Episode IV - A New Hope (1977)</td>
    </tr>
    <tr>
      <th>3275</th>
      <td>Action|Crime|Drama|Thriller</td>
      <td>Boondock Saints, The (2000)</td>
    </tr>
    <tr>
      <th>1208</th>
      <td>Action|Drama|War</td>
      <td>Apocalypse Now (1979)</td>
    </tr>
    <tr>
      <th>1196</th>
      <td>Action|Adventure|Sci-Fi</td>
      <td>Star Wars: Episode V - The Empire Strikes Back...</td>
    </tr>
    <tr>
      <th>1233</th>
      <td>Action|Drama|War</td>
      <td>Boot, Das (Boat, The) (1981)</td>
    </tr>
    <tr>
      <th>1198</th>
      <td>Action|Adventure</td>
      <td>Raiders of the Lost Ark (Indiana Jones and the...</td>
    </tr>
  </tbody>
</table>
</div>




```python
average_ratings = (data.ratings).groupby(['item']).mean()
rating_counts = (data.ratings).groupby(['item']).count()
average_ratings = average_ratings.loc[rating_counts['rating'] > minimum_to_include]
average_ratings = average_ratings.join(data.movies['genres'], on='item')
average_ratings = average_ratings.loc[average_ratings['genres'].str.contains('Romance')]

sorted_avg_ratings = average_ratings.sort_values(by="rating", ascending=False)
joined_data = sorted_avg_ratings.join(data.movies['title'], on='item')
joined_data = joined_data[joined_data.columns[3:]]
print("RECOMMENDED FOR A ROMANCE MOVIE FAN:")
joined_data.head(10)
```

    RECOMMENDED FOR A ROMANCE MOVIE FAN:





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>genres</th>
      <th>title</th>
    </tr>
    <tr>
      <th>item</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>922</th>
      <td>Drama|Film-Noir|Romance</td>
      <td>Sunset Blvd. (a.k.a. Sunset Boulevard) (1950)</td>
    </tr>
    <tr>
      <th>898</th>
      <td>Comedy|Drama|Romance</td>
      <td>Philadelphia Story, The (1940)</td>
    </tr>
    <tr>
      <th>1235</th>
      <td>Comedy|Drama|Romance</td>
      <td>Harold and Maude (1971)</td>
    </tr>
    <tr>
      <th>912</th>
      <td>Drama|Romance</td>
      <td>Casablanca (1942)</td>
    </tr>
    <tr>
      <th>1197</th>
      <td>Action|Adventure|Comedy|Fantasy|Romance</td>
      <td>Princess Bride, The (1987)</td>
    </tr>
    <tr>
      <th>933</th>
      <td>Crime|Mystery|Romance|Thriller</td>
      <td>To Catch a Thief (1955)</td>
    </tr>
    <tr>
      <th>908</th>
      <td>Action|Adventure|Mystery|Romance|Thriller</td>
      <td>North by Northwest (1959)</td>
    </tr>
    <tr>
      <th>4973</th>
      <td>Comedy|Romance</td>
      <td>Amelie (Fabuleux destin d'Amélie Poulain, Le) ...</td>
    </tr>
    <tr>
      <th>356</th>
      <td>Comedy|Drama|Romance|War</td>
      <td>Forrest Gump (1994)</td>
    </tr>
    <tr>
      <th>7361</th>
      <td>Drama|Romance|Sci-Fi</td>
      <td>Eternal Sunshine of the Spotless Mind (2004)</td>
    </tr>
  </tbody>
</table>
</div>



**Import Jabril & JGB's movie ratings**


```python
import csv

jabril_rating_dict = {}
jgb_rating_dict = {}

with open("/Users/Dongeun_Min/Downloads/lab4-recommender-systems/jabril-movie-ratings.csv", newline='') as csvfile:
  ratings_reader = csv.DictReader(csvfile)
  for row in ratings_reader:
    if ((row['ratings'] != "") and (float(row['ratings']) > 0) and (float(row['ratings']) < 6)):
      jabril_rating_dict.update({int(row['item']): float(row['ratings'])})

with open("/Users/Dongeun_Min/Downloads/lab4-recommender-systems/jgb-movie-ratings.csv", newline='') as csvfile:
  ratings_reader = csv.DictReader(csvfile)
  for row in ratings_reader:
    if ((row['ratings'] != "") and (float(row['ratings']) > 0) and (float(row['ratings']) < 6)):
      jgb_rating_dict.update({int(row['item']): float(row['ratings'])})

print("Rating dictionaries assembled!")
print("Sanity check:")
print("\tJabril's rating for 1197 (The Princess Bride) is " + str(jabril_rating_dict[1197]))
print("\tJohn-Green-Bot's rating for 1197 (The Princess Bride) is " + str(jgb_rating_dict[1197]))
```

    Rating dictionaries assembled!
    Sanity check:
    	Jabril's rating for 1197 (The Princess Bride) is 4.5
    	John-Green-Bot's rating for 1197 (The Princess Bride) is 3.5


## 3. Model Train: Collaborative Filtering Model


```python
from lenskit.algorithms import Recommender
from lenskit.algorithms.user_knn import UserUser

num_recs = 10  #<---- This is the number of recommendations to generate. You can change this if you want to see more recommendations

user_user = UserUser(15, min_nbrs=3) #These two numbers set the minimum (3) and maximum (15) number of neighbors to consider. These are considered "reasonable defaults," but you can experiment with others too
algo = Recommender.adapt(user_user)
algo.fit(data.ratings)

print("Set up a User-User algorithm!")
```

    Set up a User-User algorithm!



```python
jabril_recs = algo.recommend(-1, num_recs, ratings=pd.Series(jabril_rating_dict))  #Here, -1 tells it that it's not an existing user in the set, that we're giving new ratings, while 10 is how many recommendations it should generate

joined_data = jabril_recs.join(data.movies['genres'], on='item')      
joined_data = joined_data.join(data.movies['title'], on='item')
joined_data = joined_data[joined_data.columns[2:]]
print("\n\nRECOMMENDED FOR JABRIL:")
joined_data
```



    RECOMMENDED FOR JABRIL:





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>genres</th>
      <th>title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Comedy|Drama</td>
      <td>Last Detail, The (1973)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Comedy</td>
      <td>Love and Death (1975)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Drama</td>
      <td>Before Night Falls (2000)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Drama</td>
      <td>Magdalene Sisters, The (2002)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Drama|Horror|Mystery|Sci-Fi|Thriller</td>
      <td>Black Mirror: White Christmas (2014)</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Action|Animation|Drama|Fantasy|Sci-Fi</td>
      <td>Neon Genesis Evangelion: The End of Evangelion...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Action|Adventure|Thriller</td>
      <td>Raiders of the Lost Ark: The Adaptation (1989)</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Comedy|Drama|Romance</td>
      <td>Submarine (2010)</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Adventure|Drama</td>
      <td>Nebraska (2013)</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Documentary</td>
      <td>Endless Summer, The (1966)</td>
    </tr>
  </tbody>
</table>
</div>




```python
jgb_recs = algo.recommend(-1, num_recs, ratings=pd.Series(jgb_rating_dict))  #Here, -1 tells it that it's not an existing user in the set, that we're giving new ratings, while 10 is how many recommendations it should generate

joined_data = jgb_recs.join(data.movies['genres'], on='item')      
joined_data = joined_data.join(data.movies['title'], on='item')
joined_data = joined_data[joined_data.columns[2:]]
print("RECOMMENDED FOR JOHN-GREEN-BOT:")
joined_data
```

    RECOMMENDED FOR JOHN-GREEN-BOT:





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>genres</th>
      <th>title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Comedy</td>
      <td>The Night Before (2015)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Adventure|Drama|Sci-Fi</td>
      <td>Day of the Doctor, The (2013)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Drama|Fantasy|Romance</td>
      <td>Wristcutters: A Love Story (2006)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Comedy|Musical</td>
      <td>Holiday Inn (1942)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Comedy</td>
      <td>Outside Providence (1999)</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Comedy|Romance</td>
      <td>Adam's Rib (1949)</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Drama</td>
      <td>Reign Over Me (2007)</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Drama</td>
      <td>Guess Who's Coming to Dinner (1967)</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Drama</td>
      <td>Half Nelson (2006)</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Comedy</td>
      <td>Fired Up (2009)</td>
    </tr>
  </tbody>
</table>
</div>



**making a combined recommendation list**


```python
combined_rating_dict = {}
for k in jabril_rating_dict:
  if k in jgb_rating_dict:
    combined_rating_dict.update({k: float((jabril_rating_dict[k]+jgb_rating_dict[k])/2)})
  else:
    combined_rating_dict.update({k:jabril_rating_dict[k]})
for k in jgb_rating_dict:
   if k not in combined_rating_dict:
      combined_rating_dict.update({k:jgb_rating_dict[k]})

print("Combined ratings dictionary assembled!")
print("Sanity check:")
print("\tCombined rating for 1197 (The Princess Bride) is " + str(combined_rating_dict[1197]))
```

    Combined ratings dictionary assembled!
    Sanity check:
    	Combined rating for 1197 (The Princess Bride) is 4.0



```python
combined_recs = algo.recommend(-1, num_recs, ratings=pd.Series(combined_rating_dict))  #Here, -1 tells it that it's not an existing user in the set, that we're giving new ratings, while 10 is how many recommendations it should generate

joined_data = combined_recs.join(data.movies['genres'], on='item')      
joined_data = joined_data.join(data.movies['title'], on='item')
joined_data = joined_data[joined_data.columns[2:]]
print("\n\nRECOMMENDED FOR JABRIL / JOHN-GREEN-BOT HYBRID:")
joined_data
```



    RECOMMENDED FOR JABRIL / JOHN-GREEN-BOT HYBRID:





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>genres</th>
      <th>title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Comedy|Drama|Romance</td>
      <td>Submarine (2010)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Drama|Romance</td>
      <td>Call Me by Your Name (2017)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Drama|Sci-Fi</td>
      <td>Man Who Fell to Earth, The (1976)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Comedy|Romance</td>
      <td>Adam's Rib (1949)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Drama|War</td>
      <td>Gallipoli (1981)</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Drama</td>
      <td>Before Night Falls (2000)</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Adventure|Drama|Sci-Fi</td>
      <td>Day of the Doctor, The (2013)</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Action|Adventure|Thriller</td>
      <td>Raiders of the Lost Ark: The Adaptation (1989)</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Adventure|Drama|Western</td>
      <td>True Grit (1969)</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Comedy</td>
      <td>Love and Death (1975)</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
