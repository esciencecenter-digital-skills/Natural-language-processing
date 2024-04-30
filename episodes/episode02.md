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
- lemmatization: here you would get the lemma of each word. You would get the form in which you find the word in the dictionary, such as the singular version of a plural of a noun, or the first person present tense of a verb instead of the past principle. You are making sure that you do not get different versions of the same word: you would convert `words` into `word` and `talking` into `talk`
- part of speech tagging. This means that you identify what type of word each is; such as nouns and verbs.

# The corpus
In order to start the preprocssing we first load in the data. For that we need a number of python packages
```python
# import packages
import spacy
import io
import matplotlib.pyplot as plt
```

We can then open the text file that contains the text and save it in a variable called corpus_full.
```python
# Load the book The case-book of Sherlock Holmes by Arthur Conan Doyle
path = "../pg69700.txt"
f = io.open(path, mode="r", encoding="utf-8-sig")
corpus_full = f.read()
```
Let's check out the start of the corpus
```python
# Print the text
print(corpus_full[0:1000])
```
This shows that the corpus contains a lot of text before the actual first story starts. Let's therefore select the part of the corpus_full that contains the first story. We determined beforehand which part of the string corpus_full catches the first story, and we can save it in the parameter corpus:
```python
# Select the first story
corpus = corpus[5049:200000]
```
Let's again have a look at what the text looks like:
```python
print(corpus)
```

The print statement automatically formats the text. We can also have a look at what the unformatted text looks like:
```python
corpus
```
This shows that there are things in there sich as `\n` which defines new lines. This is one of the things we want to eliminate from the text in the preprocessing steps so that we have a more analyzable text to work with.

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
words = corpus_words.split(" ")
word
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

We did all these steps by hand, to get an understanding of what is needed to create the tokens. However these steps can also be done with a Python package, where these things happen behind the scenes. We will now start using this package to look at the resilts of the preprocessing steps of stop word removal, stemming and part-of-speech tagging.

## Spacy NLP pipeline
There are multiple python packages that can be used to for NLP, such as Spacy, NLTK, XXX. Here we will be using the Spacy package.

Let's first load a few packages that we will be using.
```Python
import spacy
from collections import Counter
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
```

We load the pipeline that we are going to use which is `en_code_web_md`. This a pipeline that is pretrained to do a number of NLP tasks for English texts. It is important to realize that in many cases you can use a pretrained model such as this one. You then do not have to do any training of your data. This is very nice, because training a model takes requires a whole lot of data, that would have to  analyzed by hand before you can start. It also requires a lot of specific understanding of NLP and a lot of time, and often it is simply not neccesary. These available models are trained on a lot of data, and have very good accuracies.

<em>Maybe something on when you do not want to use pretrained models and train of finetune.</em>

```python
nlp = spacy.load('en_code_web_md')

```

We can check out which components are available in this pipeline:
```python
# loaded components
print("components:", nlp.component_names)
```

Let's now apply this pipeline on our data. Remember that we as a first step before the break we loaded the data in the variable called corpus. We now give this as an argument to the pipeline; this will apply the model to our specific data; such that we have all components of the pipeline available on our specific corpus.
```python
# apply model to our corpus
doc = nlp(corpus)
```

On of the things that the pipeline does, is tokenization as we did in the first part. We can now check out the sentence tokens like this:

```python
# Get sentences
for sentence in doc.sents:
    print(sentence)
```

and the word tokens like this:
```python
# Get word tokens
for token in doc[0:6]:
    print(token.text)
```

## Stop word removal
If we want to get an idea of what the text is about we can visualize the word tokens in a word cloud to see which words are most common.

PUT THIS  IN A SEPARATE FUNCTION THAT IS LOADED INTO THE NOTEBOOK
```python
from wordcloud import WordCloud
# put this in a separate function
def plot_wordcloud(sw = (""), doc = doc):
      wc = WordCloud(stopwords=sw).generate(str(doc))
      plt.imshow(wc, interpolation='bilinear')
      plt.axis("off")
      return plt.show()
```

```python
plot_wordcloud(doc = doc)
```

From this we get no idea what the text is about because the most common words are word such as 'the', 'a', and 'I', which are referred to as stopwords. We could therefore want to remove these stopwords. The spacy package has a list of stopwords available. Let's have a look at these:

```python
# Stop word removal
stopwords = nlp.Defaults.stop_words

print(len(stopwords))
print(stopwords)
```
The pipeline has 326 stopwords, and if we have a look at them you could indeed say that these words do not add much if we want to get an idea of what the text is about. So let's create the word cloud again, but without the stopwords:

```python
plot_wordcloud(sw= stopwords, doc = doc)
```

This shows that Holmes is the most common word in the text, as one might expect. There are also words in this word cloud that would also consider as stop words in this case, such as `said` and `know`. If you would want to remove these as well you can add them to the list of stopwords that we used.

## Lemmatization
Let's now have a look at the lemmatization. From the wordcloud, we can see that one of the most common words in the text is the word `said`. This is past tense of the word `say`. If we want all the words referring to the word `say` we should look at the lemmatized text. We saw in the pipeline that this is also one of the components of the pipeline, so we already have all the lemmas available. We can check them out using:
```python
# Lemmas
for token in doc:
      print(token.text, token.lemma_)
```

Here we can for example see that even the `n't` is recognized as not.

## Part-of-speech tagging 
The last thing we want to look at right now is part-of-speech tagging. The loaded model can tell for each word token what type of word it is grammatically. We can acces these as follows:

```python
# Part-of-speech tags
for token in doc:
    print(token.text, token.pos_)
```

It recognizes determiners, nouns, adpositions, and more. But we can also see that it is not perfect and mistakes are made. That is something important to remember; any model, pretrained or if you train it yourself: there are always mistakes in it.

:::::::::: Wrap up
