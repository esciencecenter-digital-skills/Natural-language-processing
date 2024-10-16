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

Next we will be looking into text preprocessing. <span style="color:red"> The task we will tackly is to get an overview of the most common words in a piece of text </span>. We will be using the first page of the edition of de [Algemeen Dagblad from July 21 1969](https://www.delpher.nl/nl/kranten/view?coll=ddd&query=&cql%5B%5D=%28date+_gte_+%2220-07-1969%22%29&redirect=true&sortfield=date&resultscoll=dddtitel&identifier=KBPERS01:002846018:mpeg21&rowid=3) for this exercise that can be downloaded as txt-file from Delpher.

It is possible to download the data in different formats; jpg, pdf, and txt. The jpg file is the original scan that was made of the page, and can thus be considered the raw data. Because we want to use the text itself we download the txt file. This file contains only the text without formatting and images, and is produced by Delpher using a technique called Optical Character Recognition (OCR). This is a technique in which text from an image is converted into text. Very simply said, OCR software identifies which parts of an image contain text, and identifies the individual caracters to reconstruct the text. Although in research you might also have to start from images rather than text files, we will not go into optical character recognition. It is quite a difficult process that is beyond the scope of this course.


# Preprocessing
NLP models work by learning the statistical regularities within the constituent parts of the language (i.e, letters, digits, words and sentences) in a text. Before applying these models, the input text must often be modified to make it into a format that is better interpretable by the model. This operation is known as `data preprocessing` and its goal is to make the text ready to be efficiently processed by the model. Applying preprocessing steps will give better results in the end.

Examples of preprocessing steps are:

- cleaning the text: remove symbols/special characters, or other things that "sneaked" into the text while loading the original version.
- lowercasing
- removing punctuation
- stop word removal, where you remove common words such as `the` or `a` that you would note need in some further analysis steps.
- tokenization: this means splitting up the text into individual tokens. You can for example create sentence tokens or words tokens, or any others.
- lemmatization: with this step you obtain the lemma of each word. You would get the form in which you find the word in the dictionary, such as the singular version of a plural of a noun, or the first person present tense of a verb instead of the past principle. You are making sure that you do not get different versions of the same word: you would convert `words` into `word` and `talking` into `talk`
- part of speech tagging: This means that you identify what type of word each is; such as nouns and verbs.

The above examples of techniques of data preprocessing modify the input text to make it interpretable and analyzable by the NLP model of our choice. Here we will go through several steps to be aware of which steps can be performed and what their consequences are. However, It is important to realize that you do not always need to do all the preprocessing steps, and which ones you should do depends on what you want to do. 
For example, if you want to extract entities from the text using named entity recognition, you explicitly do not want to lowercase the text, as capitals are a component in the identification process.
Another important thing is that NLP tasks and the preprocessing can be very diffent for different languages. This is both in terms of which steps to apply, but also which methods to use for a specific step.

<span style="color:red"> Right now, we are going to apply a number of preprocessing steps to obtain a list of all distinct word tokens from the newspaper page. </span>

## Loading the corpus
In order to start the preprocessing we first load in the data. For that we need a number of python packages.

```python
# import packages
import io
import re
import spacy
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

### Cleaning the text
A first thing to do is to clean the text. As said, in this case the text is in pretty good state, close to the original. However, we can still improve the interpretability of the text by removing a number of special characters, bringing the text closer to the original. First we'll remove any special symbols using a regex. Then we'll also remove the three dashes separating different news articles and the vertical bars separating some of the columns.

```python
clean = re.sub(r'[^\n -~\u00C0-\u017F]', "", corpus)
cleaner = clean.replace("---", "")
corpus_clean = cleaner.replace("|", "")

print(corpus_clean)
```

### Lowercasing
Our next step is to lowercase the text. Our goal here is to generate a list of unique words from the text, so in order to not have words twice in the list - once normal and once capitalised when it is at the start of a sentence for example- we can lowercase the full text. 

```python
corpus_lower = corpus_clean.lower()

print(corpus_lower)
```
It is important to keep in mind that in doing this, some information is lost. As mentioned before, models that are trained to identify named entities use information on capitalisation. As another example, there are a lot of names and surnames that carry meaning. "Bakker" is a common Dutch surname, but is also a noun (baker). In lowercasing the text you loose the distinction between the two.

### Tokenisation
A very important step in NLP is tokenisation. Tokenisation is breaking up text into smaller, segments referred to as tokens. Tokens can split text at different levels, such as sentences, words, subwords or characters. This used level depends on the task at hand and the model and technique that is used. Tokenisation is essential in NLP, as it helps to creating structure from raw text. Once a text is split into tokens, these can be transformed into vectors (i.e. numbers), which can used for further processing in an efficient manner. Although a token that is transformed into a vector is then represented by numbers, it can still carry linguistic meaning, as we will discuss later on.

You can tokenise your text with Python using various existing tokenisers. There are tokenisers available for different languages, as each language has their own intricacies that should be taken into account when splitting up text. A good word tokeniser for example, does not simply break up a text based on spaces and punctuation, but it should be able to distinguish:

- abbreviations that include points (e.g.: *e.g.*)
- times (*11:15*) and dates written in various formats (*01/01/2024* or *01-01-2024*)
- word contractions such as *don't*, these should be split into *do* and *n't*
- URLs

Many older tokenisers are rule-based, meaning that they iterate over a number of predefined rules to split the text into tokens, which is useful for splitting text into word tokens for example. Modern large language models use subword tokenisation, which are more flexible.

#### Spacy
There are multiple python packages that can be used for NLP, such as `Spacy`, `NLTK`, `Gensim` and `PyTorch`. Here we will be using the `Spacy` package to create word tokens.

The model that we are going to use is called `nl_core_news_sm`. This a [model from Spacy](https://spacy.io/models/nl/) that is pretrained to do a number of NLP tasks for Dutch texts. We first have to download this model model from the Spacy library:

```python
# download the Dutch spacy model
! python -m spacy download nl_core_news_sm
```

We can then load the model into the pipeline function. This function connects the pretrained model to various preprocessing steps, including the tokenisation.

```python
# Load the Dutch model in a pipeline
nlp = spacy.load("nl_core_news_sm")
```

We can now input our corpus to the pipeline to apply the tokenisation to the text.
```
# Input our corpus
doc = nlp(corpus_lower)
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
# remove stopwords
tokens_no_stopwords = tokens_no_punct

for stopword in stopwords:
    tokens_no_stopwords = [token for token in tokens_no_stopwords if token != stopword]

print(len(tokens_no_stopwords))
```

### Token word cloud

```python
wordcloud = WordCloud().generate(' '.join(tokens_no_stopwords))

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
```

## Statistical-based tokenizers
We have now created a list of word tokens and afterwards removed stopwords and punctuation. Large language models use a more sophisticated way of tokenisation. If we look at the distinct words from out vocabulary:

```python
print(len(set(tokens)))

print(set(tokens))
```

This set is XX tokens long. If we were to process a larger piece of text, the number of distinct words would grow very large. When looking at the inidividual tokens here, we can see that there are various words that are similar in the sense that they are a plural form, or XXX form of the same word. It would be redundant to process the words as completely distinct tokens. This is why many models use sub-word tokenisation. 

:::::::::::::::::::: challenge

We have gone through various data preprocessing techniques in this episode. Now that you know how to apply them all, let's see how they affect each other.

::::::::: solution

:::::::::


::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: keypoints

- Preprocessing involves a number of steps that one can apply to their text to prepare it for further processing.
- Preprocessing is important because it can improve your results
- You do not always need to do all preprocessing steps. It depends on the task at hand which preprocessing steps are important.
- A number of preprocessing steps are: lowercasing, tokenization, stop word removal, lemmatization, part-of-speech tagging.
- Often you can use a pretrained model to process and analyse your data.

::::::::::::::::::::::::::::::::::::::::::::::::
