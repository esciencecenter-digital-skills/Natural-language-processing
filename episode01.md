---
title: "Episode 1: Apply tokenization, clean and pre-process textual data"
teaching: 0
exercises: 4
---

:::::::::::::::::::::::::::::::::::::: questions

- What different types of preprocessing steps are there?

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Apply tokenization, lower-casing and stopwords removal

::::::::::::::::::::::::::::::::::::::::::::::::

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