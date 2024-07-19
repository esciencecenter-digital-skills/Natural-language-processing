---
title: "Episode 3: Word embeddings"
teaching: 10 
exercises: 0
---

:::::::::::::::::::::::::::::::::::::::::::::::: questions

- What are word embeddings?
- What properties word embeddings have?
- What is a word2vec model?
- Can we inspect word embeddings?
- (Optional) How do we train a word2vec model?

:::::::::::::::::::::::::::::::::::::::::::::::: 

:::::::::::::::::::::::::::::::::::::::::::::::: objectives

After following this lesson, learners will be able to:

- Explain what word embeddings are
- Get familiar with using vectors to represent things
- Compute the cosine similarity to get the most similar words 
- Use Word2vec to generate word embeddings
- Extract word embeddings from a pre-trained Word2vec
- Explore properties of word embeddings
- Visualise word embeddings
- Solve analogies
- (Optional) Train your own word2vec model

::::::::::::::::::::::::::::::::::::::::::::::::

# What are word embeddings?

**You shall know a word by the company it keeps** - J. R. Firth, 1957

In this episode, we’ll go over the concept of embedding, and the steps to generate and explore word embeddings with Word2vec. 

We know that computers understand the language of numbers, so in order to let the computer process natural language, we must encode words in a sentence to numbers (i.e., vectors). Ideally, you can "transform" text in numbers in many ways. For instance, take the sentence `the cat sat on the mat`.

We could transform this sentence into a matrix in Python. The matrix will have the number of columns equals to the length of unique words in the corpus (i.e., 5 in our case) and number of words we are encoding (i.e., 6 in this example). 

```python
import numpy as np
import matplotlib.pyplot as plt

sentence = ["the", "cat", "sat", "on", "the","mat"]

words = ["cat", "mat", "on", "sat", "the"]

data = np.array([
    [0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1],
    [0, 1, 0, 0, 0],
])

fig, ax = plt.subplots()

table = ax.table(cellText=data, rowLabels=sentence, colLabels=words, cellLoc='center', loc='center')
ax.axis('off')
plt.title('the cat sat on the mat')
plt.show()
```

and count how many times we encounter a word by putting 1s (present) and 0s (absent). This approach is described as one-hot encoding:


|     | cat | mat | on  | sat | the |
|-----|-----|-----|-----|-----|-----|
| the |  0  |  0  |  0  |  0  |  1  |
| cat |  1  |  0  |  0  |  0  |  0  |
| sat |  0  |  0  |  0  |  1  |  0  |
| on  |  0  |  0  |  1  |  0  |  0  |
| the |  0  |  0  |  0  |  0  |  1  |
| mat |  0  |  1  |  0  |  0  |  0  |


To encode the word `cat` into a vector, you may concatenate each value in the vector cat = [1, 0, 0, 0, 0]. This would give us a unique fingerprint describing the word cat, and differentiating it from the other ones in the sentence. The downside of this approach is that you get a vector that is very sparse, i.e., it will contain a lot of 0s and only very few 1s. For very long documents (imagine billion of words) this approach becomes inefficient very quickly. In addition, we don't get any information about the syntactic and semantic relationship of the words.

Another strategy, is to map each word to a number. This approach is referred to as ordinal encoding:

| cat | mat | on  | sat | the |
|-----|-----|-----|-----|-----|
|   0 |  1  |  2  |  3  |  4  | 

So that the sentence can be mapped as:

```python
sentence = [4, 0, 3, 2, 4, 1]
```

This approach is much more efficient as it links each word to one numeric identifier. However, the choice for this identifier is quite arbitrary -- why does it mean that cat is 0? Does it change anything if it is encoded as 2? Moreover, there is no way to represent the relationship among words, e.g., how `cat`/0 relates to `mat`/1 ? So with this approach we gain in efficiency, but still we don't solve the problem of encoding semantic and syntactic information present in the text.

## Word embeddings
A Word Embedding is a word representation type that maps words (and whole sentences) in a numerical manner, however, differently from the approaches above, it organises information into an *efficient*, i.e., *dense*, representation in which semantic and syntactic features in the text are preserved. In this representation, words are described in a multidimensional space whereby similar words have a similar encoding. This allows us to describe them fully, and make comparisongs among words.

Let's see an example to get familiar with the concept of word embeddings. Let's use again the word `cat`. We want to describe this animal based on its characteristics. For instance, its furriness. Let's say that we measure the cat's furriness (in some magical way) and we found out that a cat has a score of `70` in furriness.

![Embedding of a cat - We measured its furriness and found out it's 70!](fig/emb1.png){alt=""}


Do you think we have described sufficiently this animal? Perhaps we can add another characteristic: Number of legs.

![Embedding of a cat - We have described it along two dimensions: furriness and number of legs](fig/emb3.png){alt=""}

We have now at least two characteristics that describe this animal. This is certainly not enough to describe this animal in **full**, however this approximation becomes helpful when we have to compare other animals, such as a dog.

![Embeddings of a cat and a dog](fig/emb4.png){alt=""}

And what about a caterpillar?

![Embeddings of a cat and a dog and a caterpillar](fig/emb5.png){alt=""}

Which of these two animals (dog vs caterpillar) is more similar to a cat? We can compute the similarity among those vectors with the function `cosine_similarity()` in Python, from the `sklearn` library.


```python
from sklearn.metrics.pairwise import cosine_similarity

cat = np.asarray([[70, 4]])
dog = np.asarray([[56, 4]])
caterpillar = np.asarray([[70, 100]])

cosine_similarity(cat, dog)

cosine_similarity(cat, caterpillar)

```

Output:

```python
array([[0.9998988]])

array([[0.61926538]])
```

- Cosine similarity between cat and dog: 0.9998988
- Cosine similarity between cat and caterpillar: 0.61926538

So the similarity between a cat and a dog is *higher* than a cat and a caterpillar. Therefore, based on these two traits, we can conclude that a cat is much more similar to a dog than a caterpillar. 

We can of course add other dimensions to describe these animals:

![Embeddings of a cat and a dog and a caterpillar - We can describe these animals in many dimensions!](fig/emb6.png){alt=""}

:::::::::::::::::::: challenge
- Add one of two other dimensions. What characteristics could they map?
- Add another animal and map their dimensions
- Compute again the cosine similarity among those animals and find the couple that is the least similar and the most similar

:::::: solution

1. Add one of two other dimensions

We could add the dimension of "velocity" or "speed" that goes from 0 to 100 meters/second. 

- Caterpillar: 0.001 m/s
- Cat: 1.5 m/s
- Dog: 2.5 m/s

(just as an example, actual speeds may vary)

```python
cat = np.asarray([[70, 4, 1.5]])
dog = np.asarray([[56, 4, 2.5]])
caterpillar = np.asarray([[70, 100, .001]])

```

Another dimension could be weight in Kg:

- Caterpillar: .05 Kg
- Cat: 4 Kg
- Dog: 15 Kg

(just as an example, actual weight may vary)

```python
cat = np.asarray([[70, 4, 1.5, 4]])
dog = np.asarray([[56, 4, 2.5, 15]])
caterpillar = np.asarray([[70, 100, .001, .05]])

```

Then the cosine similarity would be:

```python
cosine_similarity(cat, caterpillar)

cosine_similarity(cat, dog)
```

Output:

```python
array([[0.61814254]])
array([[0.97893809]])
```
2. Add another animal and map their dimensions

Another animal that we could add is the Tarantula!

```python
cat = np.asarray([[70, 4, 1.5, 4]])
dog = np.asarray([[56, 4, 2.5, 15]])
caterpillar = np.asarray([[70, 100, .001, .05]])
tarantula = np.asarray([[80, 6, .1, .3]])
```

3. Compute again the cosine similarity among those animals - find out the most and least similar couple

Given the values above, the least similar couple is the dog and the caterpillar, whose cosine similarity is `array([[0.60855407]])`. 

The most similar couple is the cat and the tarantula: `array([[0.99822302]])`

::::::

::::::::::::::::::::

Once we add multiple dimensions the animals' description become more complex, but also richer, therefore our comparisons become much more precise. 

The downside with this approach is that once we get more than 3 dimensions it becomes very difficult to represent the relationships among words with little arrows. However, the `cosine_similarity()` will always work, regardless of the number of dimensions. Bearing this example of an embedding in mind, let's move now to explore a model that returns us those embeddings.


:::: callout

In this example we have built our "embeddings" of a cat, dog and caterpillar with dimensions that we chose arbitrarily. Those dimensions were chosen because they were easy to measure and to see with our own eyes. However, when we deal with word embeddings trained on a corpus, it's difficult to know what the dimensions stand for. These can be many (the number must be limited by us) and it's unknown what they map to in the text. 

::::

::::::::::::::::::::::::::::::::::::: keypoints

- We can represent text as vectors of numbers (which makes it interpretable for machines)
- The most efficient and useful way is to use word embeddings
- We can easily compute how words are similar to each other with the cosine similarity
- Dimensions in word embeddings are many and not transparent

:::::::::::::::::::::::::::::::::::::

# Word2vec model

Word2Vec is a two-layer neural network that processes raw text and returns us the respective word-vectors (i.e., *word embeddings*). To use it, we have two choices: Train a word2vec model on our own or use a pre-trained one.

Word embeddings become better and better at representing the words (i.e., a word vector becomes more specific) within the text with the size of the training material. Think of all the books, articles, Wikipedia content, and other forms of text data we have lying around. These massive amount of text can be used to train a word2vec model and extract the relative embeddings, which will be particularly informative due to the size of the training input. If we were to train a word2vec model on this amount of data, we would first need the raw text, a powerful machine to process it, and some spare time to wait for the model to complete training. However, luckily for us someone else has done this training already, and we can load the output of this training (i.e., their pretrained word2vec model) on our local machine. 

In this section we are going to look at a pre-trained word2vec model (that is, pre-computed *word embeddings*) and at some of their properties. Towards the end of the section we're going to compare this model with a word2vec model that we trained on our own on a small subset of the Gutenberg books we introduced in the previous episode.

We use the trained word2vec model named `GloVe` from the `gensim` library. The GloVe embeddings was trained on an English Wikipedia dump and English Gigaword 5th Edition dataset. It has 6B tokens. The original source of the embeddings can be found here: https://nlp.stanford.edu/projects/glove/

We can download this model locally with this code:

```python
import gensim.downloader
google_vectors = gensim.downloader.load('glove-wiki-gigaword-50')
```

::::::: callout

You can see how many (and which) pretrained model the `gensim` library contains with the following code snippet:

```python
print(list(gensim.downloader.info()['models'].keys()))
```

Output:

```python
['fasttext-wiki-news-subwords-300', 'conceptnet-numberbatch-17-06-300', 'word2vec-ruscorpora-300', 'word2vec-google-news-300', 'glove-wiki-gigaword-50', 'glove-wiki-gigaword-100', 'glove-wiki-gigaword-200', 'glove-wiki-gigaword-300', 'glove-twitter-25', 'glove-twitter-50', 'glove-twitter-100', 'glove-twitter-200', '__testing_word2vec-matrix-synopsis']
```
:::::::

Next, we can look at the embedding of the word `king`:

```python
print(google_vectors['king'])
```

Output:
```python
[ 0.50451   0.68607  -0.59517  -0.022801  0.60046  -0.13498  -0.08813
  0.47377  -0.61798  -0.31012  -0.076666  1.493    -0.034189 -0.98173
  0.68229   0.81722  -0.51874  -0.31503  -0.55809   0.66421   0.1961
 -0.13495  -0.11476  -0.30344   0.41177  -2.223    -1.0756   -1.0783
 -0.34354   0.33505   1.9927   -0.04234  -0.64319   0.71125   0.49159
  0.16754   0.34344  -0.25663  -0.8523    0.1661    0.40102   1.1685
 -1.0137   -0.21585  -0.15155   0.78321  -0.91241  -1.6106   -0.64426
 -0.51042 ]
```

This vector has 50 entries, i.e., 50 dimensions. We can't say much about what those dimensions map. 
We can represent this vector with a heatmap:

```python
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

vectors= google_vectors['king']

fig, ax = plt.subplots(1, 1, figsize=(15, 1))

cmap = ax.imshow([vectors], aspect='auto',cmap=plt.cm.coolwarm)
ax.set_yticks([0])
ax.set_xticks([])
ax.set_yticklabels(['king'])
ax.set_xlabel('Embedding dimension')
plt.colorbar(cmap)

plt.tight_layout()

plt.show()

```
![Embedding of king - Glove word2vec model](fig/emb7.png){alt=""}


And compare it with other words, like "queen":

```python
vectors= google_vectors['king','queen']


fig, ax = plt.subplots(1, 1, figsize=(15, 1))

cmap = ax.imshow(vectors, aspect='auto',cmap=plt.cm.coolwarm)
ax.set_yticks([0,1])
ax.set_xticks([])
ax.set_yticklabels(['king','queen'])
ax.set_xlabel('Embedding dimension')
plt.axhline(y=.5, c='k', linestyle='--')
plt.colorbar(cmap)

plt.tight_layout()

plt.show()
```

![Embedding of king vs queen - Glove word2vec model](fig/emb8.png){alt=""}

We can see that although both vectors have the same number of dimensions, they seem to have similar values (i.e., colours) for some of them. Does this mean that they map the same features?

:::::::::::::::::::: challenge
Let's explore the dimensions of the embeddings.

- Add the vectors for ['boy','king','man', 'queen', 'woman', 'girl', 'daughter'] and plot it using the code above
- Compare the vectors by vertically scanning the columns looking for columns with similar colors. What similarities do you see? What characteristics do you think they map?

- Can you find the column that in your opinion points to `royalty`? and the one that codes for `gender`?

:::::: solution

1. add vectors ['boy','king','man', 'queen', 'woman', 'girl', 'daughter'] and plot it

```python

vectors= google_vectors['boy', 'king', 'man', 'queen', 'woman', 'girl', 'daughter']

fig, ax = plt.subplots(1, 1, figsize=(15, 3))

cmap = ax.imshow(vectors, aspect='auto',cmap=plt.cm.coolwarm)
ax.set_yticks([0,1,2,3, 4, 5, 6])
ax.set_xticks([])
ax.set_yticklabels(['boy','king','man', 'queen', 'woman', 'girl', 'daughter'])
ax.set_xlabel('Embedding dimension')
plt.axhline(y=.5, c='k', linestyle='--')
plt.axhline(y=1.5, c='k', linestyle='--')
plt.axhline(y=2.5, c='k', linestyle='--')
plt.axhline(y=3.5, c='k', linestyle='--')
plt.axhline(y=4.5, c='k', linestyle='--')
plt.axhline(y=5.5, c='k', linestyle='--')

plt.colorbar(cmap)

plt.tight_layout()

plt.show()
```
2. Compare the vectors by vertically scanning the columns looking for columns with similar colors.

![Exercise solution - Glove word2vec model](fig/emb9.png){alt=""}

While we don't know which dimension code for what, we can see that some columns are similar for all words, while other seem to distinguish the characteristic of being royal and gender. We could add more words and get a better understanding of what they code, however ultimately these would be always guesses.

3. Can you find the column that in your opinion points to a vague concept of `royalty`? and the one that codes for `gender`?

Royalty might be tracked by the dimension n. 5, but I can't find the dimension that code for gender. 



::::::

::::::::::::::::::::

## Analogies
A word analogy is a statement of the type: “*a* is to *b* as *x* is to *y*”, which means that *a* and *x* can be transformed in the same way to get *b* and *y*, respectively. Vice versa, *b* and *y* can be inversely transformed to get *a* and *x*. 

A famous analogy (King - man + woman ~= queen) shows an incredible property of word embeddings: That is, since words are encoded as vectors, we can often solve analogies with vector arithmetic. 

Let's consider this in detail:

$$
\overrightarrow{\text{king}} - \overrightarrow{\text{man}} + \overrightarrow{\text{woman}} \approx \overrightarrow{\text{queen}}
$$

Using the `Gensim` library in python, we can translate the above analogy into code:

```python
google_vectors.most_similar(positive=['king', 'woman'], negative=['man'], topn=1)
```

In the line of code above, we have added the word vectors of: $$ \overrightarrow{\text{king}}$$ and: $$\overrightarrow{\text{woman}}$$ and subtracted: $$\overrightarrow{\text{man}}$$ `Topn` was set to 1 to output the most similar (in terms of cosine similarity) word to the resulting vector. 

The output:

```python
[('queen', 0.8523604273796082)]
```

We can visualize this analogy as we did previously:

![Analogy of King - Man + woman ~= Queen](fig/emb10.png){alt="Analogy of King - Man + woman ~= Queen"}

This analogy works very well, as it matches our own expectation. However, the match is not 100%, indeed the embedding "queen" is just the closest embedding that this specific pre-trained model has in its vocabulary. That's the reason why we don't use the `=` symbol, but `~=`.

:::::::::::::::::::: challenge

Try other analogies with the code above. Find at least one analogy that works, and another that in your opinion is not exactly what you expected.

:::::: solution

An analogy that works, i.e., it matches my logic:

$$
\overrightarrow{\text{dollar}} - \overrightarrow{\text{US}} + \overrightarrow{\text{Italy}} \approx \overrightarrow{\text{euro}}
$$

```python
google_vectors.most_similar(positive=['dollar', 'Italy'], negative=['US'], topn=1)
```

Output:

```python
[('euro', 0.5166667103767395)]
```

This analogy also works, and it is based on the orthography of the words:

$$
\overrightarrow{\text{apple}} - \overrightarrow{\text{apples}} + \overrightarrow{\text{cars}} \approx \overrightarrow{\text{car}}
$$

```python
google_vectors.most_similar(positive=['apple', 'cars'], negative=['apples'], topn=1)
```

Output:

```python
[('car', 0.696682333946228)]
```

An analogy that doesn't exactly match my expectation:

$$
\overrightarrow{\text{doctor}} - \overrightarrow{\text{hospital}} + \overrightarrow{\text{school}} \approx \overrightarrow{\text{teacher}}
$$

```python
google_vectors.most_similar(positive=['doctor', 'school'], negative=['hospital'], topn=1)
```

Output:

```python
[('guidance_counselor', 0.5969595313072205)]
```

So, in this case this analogy is not solved very well by our model. I expected the model to give me the word `teacher` but instead it gave me `guidance_counselor`.

::::::

::::::::::::::::::::


## Linguistic categories, dimensionality reduction and the challenge of polysemy

In addition to analogies, we can explore how good word2vec is in capturing the syntactic and semantic similarity between words (and pairs of words), via the exploration of linguistic categories.

Linguistic categories are groups of words that describe high-level properties that all those words have in common. Consider the following words:

```python
['car', 'truck', 'bus', 'bicycle', 'motorcycle', 'scooter', 'train', 'airplane', 
                 'helicopter', 'boat', 'ship', 'submarine', 'van', 'taxi', 'ambulance', 'tractor', 
                 'trailer', 'jeep', 'minivan', 'skateboard', 'tank', 'bobcat']
```

What do they have in common? They are all `vehicles`. Consider now these words:

```python
['dog', 'cat', 'horse', 'lion', 'tiger', 'elephant', 'bear', 'wolf', 'fox', 'deer', 
                'rabbit', 'mouse', 'rat', 'bird', 'eagle', 'hawk', 'fish', 'shark', 'whale', 'dolphin',
               'fly', 'crane', 'bug','seal','cougar', 'jaguar']
```

All those words belong to the category of `animals`. Let's group those vectors in their respective category label:

```python
animal_words = ['dog', 'cat', 'horse', 'lion', 'tiger', 'elephant', 'bear', 'wolf', 'fox', 'deer', 
                'rabbit', 'mouse', 'rat', 'bird', 'eagle', 'hawk', 'fish', 'shark', 'whale', 'dolphin',
               'fly', 'crane', 'bug','seal','cougar', 'jaguar']
vehicle_words = ['car', 'truck', 'bus', 'bicycle', 'motorcycle', 'scooter', 'train', 'airplane', 
                 'helicopter', 'boat', 'ship', 'submarine', 'van', 'taxi', 'ambulance', 'tractor', 
                 'trailer', 'jeep', 'minivan', 'skateboard', 'tank', 'bobcat']
```

Let's extract now their vectors from our pre-trained word2vec model:

```python
all_words = animal_words + vehicle_words 
word_vectors = np.array([google_vectors[word] for word in all_words])
```

Intuitively, if we were to represent *visually* those categories in a two-dimensional (i.e., 2D) space, we would represent each word as a point in the plot, and separate in space those words belonging to the category of `animal` from those belonging to the category of `vehicles`. To test our intuition against the model, we can plot it.

### Dimensionality reduction

However, a point is made of two coordinate: *x* and *y*, while each word in those categories contains 50 coordinates, one for each dimension. We cannot represent more than e.g., 4 dimensions in our plot (x, y, z and colour). What's the solution then?

We must "squeeze" the dimensions into 2 (x and y). This process is called dimensionality reduction. The idea behind it is to represent a set of high-dimensional vectors as 2D points in such a way that the distances between pairs of points are preserved as much as possible. Of course this will be an approximation, however in most cases is good enough to test our intuition. There are many methods of dimensionality reduction, in this case we use `UMAP` from the homonymous library:

```python
# Reduce dimensions using UMAP
import umap.umap_ as umap

reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=40)
embedding = reducer.fit_transform(word_vectors)

```

- Setting `n_neighbors` to 15 means UMAP will consider each point in the context of its 15 nearest neighbors. Lower values of `n_neighbors` can capture more local structure, which might be useful for clustering closely related points.

- `min_dist`: This parameter controls how tightly UMAP packs points together. It defines the minimum distance between points in the embedding space. Lower values lead to more tightly packed embeddings, while higher values result in a more spread out embedding. A value of `0.1` allows some flexibility while keeping the embeddings packed.

- Setting `random_state` to 40 ensures that the results are reproducible. This means that each time the code is run with the same data and parameters, the output will be the same.

- `fit_transform` picks the high-dimentional data and transforms it into a lower-dimensional space.

We plot the result of the dimensionality reduction:

```python
plt.figure(figsize=(12, 8))

# animal words
for i, word in enumerate(animal_words_in_vocab):
    plt.scatter(embedding[i, 0], embedding[i, 1], color='blue')
    plt.text(embedding[i, 0] + 0.1, embedding[i, 1] + 0.1, word, fontsize=9, color='blue')

# vehicle words
for i, word in enumerate(vehicle_words_in_vocab, start=len(animal_words_in_vocab)):
    plt.scatter(embedding[i, 0], embedding[i, 1], color='red')
    plt.text(embedding[i, 0] + 0.1, embedding[i, 1] + 0.1, word, fontsize=9, color='red')

plt.title('2D Visualization of Animal and Vehicle Words using UMAP')
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.grid(True)
plt.show()
```

![2D visualisation of animal and vehicle word embeddings](fig/emb11.png){alt=""}

The visualisation confirms our real world knowledge:

1. Words belonging to the animal realm are closer and form a group together. Same for the vehicle words.

2. Those two categories form two clusters, i.e., it would be easy to draw with a pen their perimeter

### Polysemy 

At a closer inspection, however, there is also an interesting phenomenon visible: The words `fly` (for animals) and `bobcat` (for vehicles) are somewhat confused. They are represented closer to the vehicles and animals, respectively. The reason for this "confusion" is due to the fact that `fly` and `bobcat` are polysemous words, i.e., they have multiple meanings. A vast majority of words, especially frequent ones, are polysemous, with each word taking on anywhere from two to a dozen different senses in many natural languages. We are able to disambiguate words based on the context in which they occur. Word2vec however is not able to do it.

Note this problem in the word `crane`. A crane is a large, tall machine used for moving heavy objects and a tall, long-legged, long-necked bird. This word is categorised as animal, and it falls closer indeed to the animal realm. However it would have been better to represent it half-way through a vehicle and an animal, since it's also some sort of vehicle. 

::::::::::::::::::::::::::::::::::::: keypoints

- We can explore linguistic categories via word2vec by extracting the vectors of words belonging to some category we wish to investigate

- To visualise word embeddings we must reduce their dimensions to 2

- Word2vec does not deal very efficiently with polysemy as it does not allow to extract a different embedding to a word depending on its context
:::::::::::::::::::::::::::::::::::::


# (Optional) Training a word2vec model on our dataset

We import `spacy` for a light pre-processing of the text and nltk to get a subset of the books present in the Gutenberg dataset.

```python
import spacy
import nltk
from nltk.corpus import gutenberg

# this is a log setting, useful for printing during training
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

```

::::::: callout

If you haven't installed `spaCy` before, and the proper english model, run the code below from your terminal:

```python
pip install spacy
python -m spacy download en_core_web_sm

```
:::::::

We load the `en_core_web_sm` model.It is a pre-trained statistical model provided by `spaCy` for processing English language text. It includes vocabulary and syntax already.

```python
nlp = spacy.load("en_core_web_sm")
```

We download the books:

```python
nltk.download('gutenberg')
```

Let's take a look at which books we have in this dataset:

```python
available_books = gutenberg.fileids()
```

Output:

```python
available_books

['austen-emma.txt',
 'austen-persuasion.txt',
 'austen-sense.txt',
 'bible-kjv.txt',
 'blake-poems.txt',
 'bryant-stories.txt',
 'burgess-busterbrown.txt',
 'carroll-alice.txt',
 'chesterton-ball.txt',
 'chesterton-brown.txt',
 'chesterton-thursday.txt',
 'edgeworth-parents.txt',
 'melville-moby_dick.txt',
 'milton-paradise.txt',
 'shakespeare-caesar.txt',
 'shakespeare-hamlet.txt',
 'shakespeare-macbeth.txt',
 'whitman-leaves.txt']
```

Some of these books are very big, making the training excessively long. It's best for this exercise to limit ourselves to midium/small size books. In order to do so, we first compute the length of each book, then we subset those that are within 2000000 characters. This is an arbitrary value but for the sake of this exercise it serves our purpose. If we were to train all books from the Gutenberg dataset we would need access to a server a different type of code that deals with allocation of the memory more efficiently.

```python
# Calculate the length of each book
book_lengths = {book: len(nltk.corpus.gutenberg.raw(book)) for book in available_books}

# Print the size of each book
for book, length in book_lengths.items():
    print(f"{book}: {length} characters")
```

We set `npl.max_length` to `2000000` because some books in the Gutenberg libraries are very long. The `nlp.max_length` parameter ensures that they are processed without issues related to document length.

```python
nlp.max_length = 2000000 

```

Now we filter the books that are within our `nlp.max_length`:

```python
# filter books that are less than our max_length
filtered_books = [book for book, length in book_lengths.items() if length <= nlp.max_length]
filtered_books
```

We are ready to preprocess our books and train our word2vec model.

First, preprocess the books. We use an ad-hoc function to tokenize, lowercase, remove stop words and punctuation:

```python
# Function to read and preprocess texts using spaCy
def read_input(book_ids):
    """This method reads the input book IDs and preprocesses the text"""
    
    logging.info("Reading and preprocessing books...this may take a while")
    
    for book_id in book_ids:
        logging.info(f"Reading book {book_id}")
        raw_text = nltk.corpus.gutenberg.raw(book_id)
        doc = nlp(raw_text)
        
        for sentence in doc.sents:
            # Tokenize, lowercase, remove stop words and punctuation
            tokens = [token.text.lower() for token in sentence if not token.is_stop and not token.is_punct]
            if tokens:
                yield tokens
                
```

Then we run it over our dataset:

```python
# Read and preprocess the texts from the selected books
documents = list(read_input(filtered_books))
logging.info("Done reading and preprocessing data")
```

We initialise the word2vec model:

```python
model = gensim.models.Word2Vec(documents, vector_size=50, window=10, min_count=2, workers=4)
```

Note that: 

- we set `vector_size` to 50. This means that each word will be represented by a 50-dimensional vector in the embedding space. Higher dimensions can capture more semantic nuances but require more computational resources. 

- In addition, we set `window` to 10. This setting ensures that the model consider up to 10 words to the left and 10 words to the right of the target word for context. Larger windows can capture broader context but might introduce noise. 

- We ignore all words with total frequency lower than a predifined threshold by setting `min_count` to 2. This helps to remove infrequent words that may not provide useful information and could potentially introduce noise.

- We set `workers` to 4 to use 4 parallel threads for training. More workers can speed up training on multicore machines.

Now that we are all set, we start training on the polished dataset:

```python
model.train(documents,total_examples=len(documents),epochs=10)
```

We can then explore the embedding space as we did for the `GloVE` model.



:::::::::::::::: challenge

- Try exploring your pre-trained model by re-computing the famous analogy of the king and outputting the first 5 words. Do you find any difference in performance?

- If you ask for the top 10 words most similar to `King` what's the output? 

- How do you explain differences in performance, if any?


:::::: solution

- Reproducing the famous analogy of the `King - man + woman ~= queen`:

```python
model.wv.most_similar(positive=['king', 'woman'], negative=['man'], topn=5)
```

Output:

```python
[('tranquo', 0.6235862970352173),
 ('weyard', 0.6232213377952576),
 ('sicily', 0.6137406229972839),
 ('allemaine', 0.6106457710266113),
 ('bridewell', 0.597455620765686)]
```

- If you ask for the top 10 words most similar to `King` what's the output? 

```python
# the word most similar to king?
w1 = "king"
model.wv.most_similar(w1, topn=10)
```

Output:

```python
[('allemaine', 0.6935099363327026),
 ('saul', 0.6925671696662903),
 ('david', 0.6918837428092957),
 ('tranquo', 0.6515917181968689),
 ('robber', 0.6457288265228271),
 ('lords', 0.6413857936859131),
 ('sourse', 0.6361883878707886),
 ('commons', 0.6356653571128845),
 ('liued', 0.6342921853065491),
 ('queene', 0.6335513591766357)]
```

- Difference in performance is due to (1) different trained corpus and (2) different size of the corpus. 
::::::
::::::::::::::::

::::::::::::::::::::::::::::::::::::: keypoints

- We can both train or load a pre-trained word2vec model

- Embeddings of a trained model will reflect the statistics of the input dataset

- Loading a (big) pre-trained word2vec model allows us to get embeddings that better reflect the syntactic and semantic relationship among (pairs of) words. Using one or the other will depend on your research question.
:::::::::::::::::::::::::::::::::::::
