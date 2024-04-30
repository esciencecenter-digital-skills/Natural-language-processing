---
title: "Episode 2"
teaching: 
exercises: 
---

:::::::::::::::::::::::::::::::::::::::::::::::: Learning Objectives
> ## Learning Objectives
> After following this lesson, learners will be able to:
> - Explain what preprocessing means.
> - Explain why we need preprocessing.
> - Do various preprocessing steps: lowercasing, handling new lines, tokenizing, stop word removal, parts-of-speech tagging, stemming/lemmatizing such that they: 
have a list of words in a piece of text arranged with their parts of speech and lemmas or stems, and with very frequent words removed

:::::::::::::::::::::::::::::::::::::::::::::::: Preprocessing
In order to start analyzing out text we will first do some preprocessing. Preprocessing means that we take a number of steps to put our data in a form that is analyzable; that there is no noise in the data and that we have some features of the text.

Examples of preprocessing steps are:
- tokenization: this means splitting up the text into individual tokens. You can for example create sentence tokens or words tokens.
- lowercasing
- stop words removal, where you remove common words such as `the` or `a` that you would note need in some further analysis steps.
- lemmatization/stemming: here you would get a root of each word, such that you do not get different versions of the same word: you would convert `words` into `word` and `talking` into `talk`
- part of speech tagging. This means that you identify what type of word each is; such as nouns and verbs.

In order to start the preprocssing we first load in the data:
# The corpus
```python
# import packages
import spacy
import io
import matplotlib.pyplot as plt
```

```python
# Load the book The case-book of Sherlock Holmes by Arthur Conan Doyle
path = "../pg69700.txt"
f = io.open(path, mode="r", encoding="utf-8-sig")
corpus = f.read()
```

```python
# Print the text
print(corpus[0:1000])
```

```python
# Select the first story
corpus = corpus[5049:200000]
# Show the text formatted
print(corpus)
```

```python
# Unformatted text
corpus
```

# Tokenization
We will now start by splitting the text up into individual sentences and words. This process is referred to as tokenizing; instead of having one long text we will create individual tokens.

Tokens can be defined in different ways: here we will first split text up into  sentence tokens, so that each token represents one sentence in the text. Then we will extract the word tokens, where each token is one word.

## Individual words and sentences 
Sentences are separated by points, and words are separed by spaces, we can use this information to split the text. However, we saw that when we printed the corpus, that the text is not so 'clean'. If we were to split the text now using points, there would be a lot of redundant symbols that we do not want to include in the individual sentences and words, such as the \n symbols, but also we do not want to include punctuation symbols in our sentences and words. So let's remove these from the text before splitting it up based on. 

### Sentences
The text can be split into sentences based on points. From the corpus as we have it, we do not want to include the end of line symbols, backslashes before apostrophes, and any double spaces that might occur from new alineas or new pages.

We will define `corpus_sentences` to do all preprocessing steps we need to split the text into individual sentences. First we replace the end of lines and backslashes.

```python
# Replace newlines with spaces:
corpus_sentences = corpus.replace("\n", " ")

# Replace backslashes
corpus_sentences = corpus_sentences.replace("\"", "")
```

Then we can replace the double spaces with single spaces. However, there might be multiple double spaces in the text after one another. To catch these, we can repeat the action of replacing double spaces a couple of times, using a loop 
```python
# Replace double spaces with single spaces
for i in range(10):
      corpus_sentences = corpus_sentences.replace("  ", " ")
      i = i + 1
```

```python
# Check that there a no more double spaces
"  " in corpus_sentences
```
Indeed there a no more double spaces.

Now we are ready to split the text into sentences based on points.
```python
sentences = corpus_sentences.split(". ")
```
What this does it that the corpus_sentences is split up every time a `. `is found, and the results are stored in a python list.

If we print the first 20 items in the resulting list, we can see that indeed the data is split up into sentences, but there are some mistakes, where for example a new sentence is defined because the word 'mister' was abbreviated which also resulted in a new sentence definition. This shows that these kind of steps will never be fully perfect, but it good enough to proceed.

```python
sentences[0:20]
```

### Words
We can now procede from `corpus_sentences` to split the corpus into individual words based on spaces. To get 'clean words' we need to replace some more punctuation marks, so that these are not included in the list of words.

Let's first define the punctuation marks we want to remove:
```python
# Punctuation symbols
punctuation = (".", ",", ":", ";", "(", ")", "!", "?", "\"")
```

Then we go over all these punctuation symbols one by one using a loop to replae them:

```python
# Loop over the punctuation symbols to remove them
corpus_words = corpus_sentences

for punct in punctuation:
      corpus_words = corpus_words.replace(punct, "")

# Again replace double spaces with single spaces
for i in range(10):
      corpus_words = corpus_words.replace("  ", " ")
      i = i + 1
```

Next, we should lowercase all text so that we don't get a word in two forms in the list, once with captial, once without, and have a consistent list of words:
```python
# Lowercase the text
corpus_words = corpus_words.lower()
```

Now can we split the text into individual words based by splitting them up on every space:
```python
word_list = corpus_words.split(" ")
word_list
```
The list of the words that we now have contains a lot of duplicates. We can get the unique words by converting the list into a set:
```python
words = set(word_list)
```


# Break

# Using a spacy pipeline to analyse texts
Before the break we did a number of preprocessing steps to get the sentence tokens and word tokens. We took the following steps:

- We loaded the corpus into one long string and selected the part of the string that we wanted to analyse, which is the first story
- We replaced new lines with spaces and removed all double spaces.
- We split the string into sentences based on points
To continue getting the individual words:
- we removed punctuation marks
- removed double spaces
- we lowercased the text
- We split the text into a list of words based on spaces.
- We selected all indidivual words by converting the list into a set.
