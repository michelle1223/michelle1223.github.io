---
layout: post
title: What is RNN?
category: NLP
tag: NLP
---

Recurrent Neural Network, also known as RNN, is a neural network commonly used for NLP(Natural Language Processing). To understand NLP and deep learning, it is crucial to know what RNN is and how to use it.  
To start off, we must know that RNN is a **sequence model**. This means that the model takes in a 'sequence' of inputs to find the next element of this sequence. When attempting NLP, we must use this 'sequence model'. In a sequence model, a neural network works as a function that predicts the next element of a sequence given the current element of the sequence.  

![sequence model preview](https://indico.io/wp-content/uploads/2016/04/sequence-nathan-fig2.jpg)

Language models generate the next data by feeding their previous outputs back into the model. The representative of these language models is RNN. Basic RNNs take each element of a sequence, multiply the element by a matrix, and then sum the result with the previous output from the network. This is described by the following equation.

ht = activation(XtWx + ht-1Wh )

As mentioned previously, RNN is one of the most commonly used language models. Then why is RNN effective as a language model? This is because when making predictions, RNN does not use an element one at a time. For example, let's say that we need to predict the next word for the following sentence:  

'Tom is not good at science, so he will ___ .'  

The next word may be something like 'study'. How do we know that? To predict the next word of this sentence, we have to consider every word that came before the blank. It is hard to accurately predict the next word if we only consider the word right before the blank, such as 'will', or 'he'. Only when our model considers every element before the blank, including 'science', or 'not', 'good', our model will accurately predict the next word. This is where RNN shows its value. Unlike simple language models that just try to predict the next word given the current word, RNN models capture the **entire context** of the input sequence. Therefore, RNN can predict the probability of generating the next word based on all previous words.  

In the next post, we will see how to code RNN in python, using jupyter notebook.

<!-- more -->

_Reference from:  
[Sequence Modeling with Neural Networks] (https://indico.io/blog/sequence-modeling-neuralnets-part1/)  
[Recurrent Neural Network] (https://wikidocs.net/22886)_
