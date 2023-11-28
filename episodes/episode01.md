---
title: "Episode 1: Apply tokenization, clean and pre-process textual data"
teaching: 0
exercises: 4
---

:::::::::::::::::::::::::::::::::::::: questions

- What is a document embedding?

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Explain what document embedding and TF-IDF is

::::::::::::::::::::::::::::::::::::::::::::::::

::: challenge

In this exercise we will do some preprocessing on the text: 
"Natural language processing (NLP) is an interdisciplinary subfield of computer science and linguistics."

As a first step; apply lower casing on the given text.

:::::: solution

Import and download the following packages:
```python
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')
```
Then, lower case the text:

```python
text = "Natural language processing (NLP) is an interdisciplinary subfield of computer science and linguistics."
text.lower()
```
::::::
:::
