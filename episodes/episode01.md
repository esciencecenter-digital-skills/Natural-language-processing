---
title: "Episode 1: Introduction to Natural Language Processing (NLP)"
teaching: 0
exercises: 4
---

:::::::::::::::::::::::::::::::::::::: questions

- What is natural language processing (NLP)?
- Why is it important to learn about NLP?
- What are some classic tasks associated with NLP?

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Recognise the importance and benefits of learning about NLP
- Identify and describe classic tasks and challenges in NLP 
- Explore practical applications of natural language processing in industry and research

::::::::::::::::::::::::::::::::::::::::::::::::

## Who is this lesson for?
Natural Language Processing (or NLP) refers to a set of techniques involving the application of statistical methods, 
with or without insights from linguistics, to understand natural (i.e, human) language for the sake of solving real-world tasks.

This course is designed to equip researchers in the humanities and social sciences with the foundational
skills needed to carry over text-based research projects. 

## What will we be covering in this lesson?

This lesson provides a high-level introduction to NLP with particular emphasis on applications in the humanities and the social
sciences.

After following this lessons, learners will be able to:

- Explain and differentiate what are the core topics in NLP
- Identify what kinds of tasks NLP techniques excel at, and what are their limitations
- Summarise and apply the practical, technical steps involved in preparing text (preprocessing)
- Extract vector representations of individual words, visualise it and manipulate (word embeddings)
- Solve Named Entity Recognition (NER) with BERT
- Summarise what other tasks are solvable with large language models
 

::: challenge

Before starting this exercise, a few packages have to be imported. To do this, execute the following:

Import and download the following packages:
```python
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
```

In this exercise we will do some preprocessing on the text: 
"Natural language processing (NLP) is an interdisciplinary subfield of computer science and linguistics."

- As a first step; apply lower casing on the given text.

:::::: solution

Then, lower case the text:

```python
text = "Natural language processing (NLP) is an interdisciplinary subfield of computer science and linguistics."
text.lower()
```
::::::
:::

::: challenge
A second step in preprocessing the text, apply tokenisation on the lower-cased text.
If you do not have the lower-cased text available, you can use the input text.

:::::: solution

```python
words = word_tokenize(text)
```
::::::
:::