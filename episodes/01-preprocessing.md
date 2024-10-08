---
title: "Episode 1: From text to vectors - preprocessing and word embeddings"
teaching: 10
exercises: 0
---
:::::::::::::::::::::::::::::::::::::::::::::::: questions

- What problem are we going to solve in this episode?

- What is preprocessing and why do we need it?
- What different types of preprocessing steps are there?
- What are the consequences of applying data preprocessing on our text?

- What are word embeddings?
- What properties word embeddings have?
- What is a word2vec model?
- Can we inspect word embeddings?
- (Optional) How do we train a word2vec model?

:::::::::::::::::::::::::::::::::::::::::::::::: 

:::::::::::::::::::::::::::::::::::::::::::::::: objectives

After following this lesson, learners will be able to:

- Explain what preprocessing means.
- Perform lowercasing, handling new lines, tokenization, stop words removal, part-of-speech tagging, stemmatization/lemmatization.
- Apply and use a spacy pretrained model.


::::::::::::::::::::::::::::::::::::::::::::::::

# Delpher newspaper articles
In this epsiode we are going to process Dutch newpaper texts. Newspapers make an interesting dataset for scientific research, as it contains information about current events and the language it uses is clear and reflective of its time.

::: Instructor
It would be useful to teach this while showing the website.
:::

We will be using data from [Delpher](https://www.delpher.nl/) in this episode. Delpher is a public database developed by the [KB - the National Library of the Netherlands](https://www.kb.nl/) and contains digitalised historic Dutch newspapers, books, and magazines. The online newspaper collection covers data spanning from 1618 up to 1995 and of many local, national and international publishers. 
What we will be looking into is to examine the sematic shifts of specific words over various decades. We will look at the context in which a word is used between the 1950s and 1990s.

<span style="color:red">Expand a bit more?</span>


::::::::::::::::::::::::::::::::::::::: challenge
## Exploring Delpher
Before we will collect the data, let's play around a bit in Delpher. Go to the [Delpher](https://www.delpher.nl/) and look around what data they all have. Can you find anything in the data that you find interesting or didn't know yet? For example about your living area, sports club, or an historic event?

::::::::::::::: solution

Next we will be looking into text preprocessing. The task we will tackly is to get an overview of the most common words in a piece of text. We will be using the first page of the edition of de [Algemeen Dagblad from July 21 1969](https://www.delpher.nl/nl/kranten/view?coll=ddd&query=&cql%5B%5D=%28date+_gte_+%2220-07-1969%22%29&redirect=true&sortfield=date&resultscoll=dddtitel&identifier=KBPERS01:002846018:mpeg21&rowid=3) for this exercise that can be downloaded as txt-file from Delpher.

It is possible to download the data in different formats; jpg, pdf, and txt. The jpg file is the original scan that was made of the page, and can thus be considered the raw data. Because we want to use the text itself we download the txt file. This file contains only the text without formatting and images, and is produced by Delpher using a technique called Optical Character Recognition (OCR). This is a technique in which text from an image is converted into text. Very simply said, OCR software identifies which parts of an image contain text, and identifies the individual caracters to reconstruct the text. Although in research you might also have to start from images rather than text files, we will not go into optical character recognition. It is quite a difficult process that is beyond the scope of this course.


# Preprocessing
NLP models work by learning the statistical regularities within the constituent parts of the language (i.e, letters, digits, words and sentences) in a text. Before applying these models, the input text must often be modified to make it into a format that is better interpretable by the model. This operation is known as `data preprocessing` and its goal is to make the text ready to be efficiently processed by the model. Applying preprocessing steps will give better results in the end.

Examples of preprocessing steps are:

- cleaning the text: remove symbols/special characters, or other things that "sneaked" into the text while loading the original version.
- lowercasing.
- remove punctuation
- stop word removal, where you remove common words such as `the` or `a` that you would note need in some further analysis steps.
- tokenization: this means splitting up the text into individual tokens. You can for example create sentence tokens or words tokens, or any others.
- lemmatization: with this step you obtain the lemma of each word. You would get the form in which you find the word in the dictionary, such as the singular version of a plural of a noun, or the first person present tense of a verb instead of the past principle. You are making sure that you do not get different versions of the same word: you would convert `words` into `word` and `talking` into `talk`
- part of speech tagging: This means that you identify what type of word each is; such as nouns and verbs.

The above examples of techniques of data preprocessing modify the input text to make it interpretable and analyzable by the NLP model of our choice. Here we will go through several steps to be aware of which steps can be performed and what their consequences are. However, It is important to realize that you do not always need to do all the preprocessing steps, and which ones you should do depends on what you want to do. 
For example, if you want to extract entities from the text using named entity recognition, you explicitly do not want to lowercase the text, as capitals are a component in the identification process.
Another important thing is that NLP tasks and the preprocessing can be very diffent for different languages. This is both in terms of which steps to apply, but also which methods to use for a specific step.

<b> Right now, we are going to apply a number of preprocessing steps to obtain a list of all distinct word tokens from the newspaper page. </b>

## Loading the corpus
In order to start the preprocessing we first load in the data. For that we need a number of python packages.

```python
# import packages
import spacy
import io
import string
import matplotlib.pyplot as plt
```

We can then open the text file that contains the text and save it in a variable called `corpus`.

```python
# Load the newspaper text
path = "./ad.txt"
with open(path) as myfile:
    corpus = myfile.read()
```
Let's check out the start of the corpus
```python
# Print the text
print(corpus)
```

Looking at the text, we can see that in this case the OCR that has been applied to the original image, has given a pretty good result. However, there are still mistakes in the recognized text. For example, on the first line the word 'juli' has misinterpreted as 'iuli'.

### Clean the text
A first thing to do is to clean the text. As said, in this case the text is in pretty good state, close to the original. However, we can still improve the interpretability of the text by removing a number of special characters. Let's define these:

### Lowercase

### Remove punctuation

### Remove stop words

### Tokenisation

### Lemmatisation

<s> ## Tokenization
We will now start by splitting the text up into individual sentences and words. This process is referred to as tokenizing; instead of having one long text we will create individual tokens.

Tokens can be defined in different ways: here we will first split text up into sentence tokens, so that each token represents one sentence in the text. Then we will extract the word tokens, where each token is one word.
</s>

</s>
## Spacy NLP pipeline
There are multiple python packages that can be used to for NLP, such as `Spacy`, `NLTK`, `Gensim` and `PyTorch`. Here we will be using the `Spacy` package.

Let's first load a few packages that we will be using:

```Python
import spacy
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
```

The pipeline that we are going to 

The pipeline that we are going to use is called `en_core_web_md`. This a [pipeline from Spacy](https://spacy.io/models/en/) that is pretrained to do a number of NLP tasks for English texts. We first have to download this model model from the Spacy library:

```python
!python -m spacy download en_core_web_sm
```

It is important to realize that in many cases you can use a pretrained model such as this one. You then do not have to do any training of your data. This is very nice, because training a model requires a whole lot of data, that would have to be analyzed by hand before you can start. It also requires a lot of specific understanding of NLP and a lot of time, and often it is simply not neccesary. These available models are trained on a lot of data, and have very good accuracy.

Free available pretrained models can be found on [Hugging Face](https://huggingface.co/), along with instructions on how to use them. This website contains a lot of models trained for specific tasks and use cases. It also contains data sets that can be used to train new models.

Let's load the model:
```python
nlp = spacy.load('en_core_web_md')

```

We can check out which components are available in this pipeline:
```python
# loaded components
print("components:", nlp.component_names)
```

Let's now apply this pipeline on our data. Remember that we as a first step before the break we loaded the data in the variable called corpus. We now give this as an argument to the pipeline; this will apply the model to our specific data; such that we have all components of the pipeline available on our specific corpus:

```python
# apply model to our corpus
doc = nlp(corpus)
```

One of the things that the pipeline does, is tokenization as we did in the first part. We can now check out the sentence tokens like this:

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
If we want to get an idea of what the text is about we can visualize the word tokens in a word cloud to see which words are most common. To do this we can define a function:

```python
from wordcloud import WordCloud

# Define function that returns a word cloud
def plot_wordcloud(sw = (""), doc = doc):
      wc = WordCloud(stopwords=sw).generate(str(doc))
      plt.imshow(wc, interpolation='bilinear')
      plt.axis("off")
      return plt.show()
```

```python
plot_wordcloud(doc=doc)
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
The last thing we want to look at right now is part-of-speech tagging. The loaded model can tell for each word token what type of word it is grammatically. We can access these as follows:

```python
# Part-of-speech tags
for token in doc:
    print(token.text, token.pos_)
```

It recognizes determiners, nouns, adpositions, and more. But we can also see that it is not perfect and mistakes are made. That is something important to remember; any model, pretrained or if you train it yourself: there are always mistakes in it.


:::::::::::::::::::: challenge

We have gone through various data preprocessing techniques in this episode. Now that you know how to apply them all, let's see how they affect each other.

* Above we removed the stopwords from the text before lemmatization. What happens if you use the lemmatized text? Create a word cloud of your results.
* The word clouds that we created can give an idea on what the text is about. However, there are still some terms in the word cloud that are not so useful to do this aim. Which further words would you remove? Add them to the stop words to improve your word cloud so that it better represents to subject of the text.

::::::::: solution

* Lemmatized word cloud

The doc can be created to consist only of lemma's as follows:
```python
lemmas = ' '.join([token.lemma_ for token in doc])
```

Create the word cloud using the lemmatized text and the stopwords we defined earlier.
```python
plot_wordcloud(doc=lemmas, sw=stopwords)
```

* Additional stop words

Add some more words to the stopwords set:
```python
add_stopwords = ['ask', 'tell', 'like', 'want', 'case', 'come']
new_stopwords = stopwords.update(set(add_stopwords))
```

Create the word cloud:
```python
plot_wordcloud(doc=lemmas, sw=new_stopwords)
```

:::::::::


::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: keypoints

- Preprocessing involves a number of steps that one can apply to their text to prepare it for further processing.
- Preprocessing is important because it can improve your results
- You do not always need to do all preprocessing steps. It depends on the task at hand which preprocessing steps are important.
- A number of preprocessing steps are: lowercasing, tokenization, stop word removal, lemmatization, part-of-speech tagging.
- Often you can use a pretrained model to process and analyse your data.

::::::::::::::::::::::::::::::::::::::::::::::::
