---
title: "Episode 4: BERT and Transformers"
teaching: 10 
exercises: 0
---
:::::::::::::::::::::::::::::::::::::::::::::::: questions

- What is a Language Model?
- What are Transformers?
- What is BERT and how does it work?
- How can I use BERT as a text classifier?
- How should I evaluate my classifiers? 

:::::::::::::::::::::::::::::::::::::::::::::::: 

:::::::::::::::::::::::::::::::::::::::::::::::: objectives

After following this lesson, learners will be able to:

- Understand how a Transformer works and recognize their different use cases.
- Use pre-trained transformers language models (e.g. BERT) to classify texts.
- Use a pre-trained transformer Named Entity Recognizer.
- Understand assumptions and basic evaluation for NLP outputs.

::::::::::::::::::::::::::::::::::::::::::::::::

# Language Model

In the previous lesson we learned how Word2Vec can be used to represent words as vectors. Having these representations allows us to apply operations directly on the vectors that have a direct mapping to some syntactic and semantic properties of words; such as the cases of analogies or finding synonyms. The main drawback of Word2Vec is that each word is represented in isolation, and unfortunately that is not how language works. Words get their meanings based on the specific context in which they are used (take for example polysemy, the cases where the same word can have very different meanings depending on the context); therefore, we would like to have richer vector representations of words that also integrate context into account in order to obtain more powerful representations. 

One way to obtain rich word representations is via Language Modelling. In general, this is the task of calculating the probability of a word based on the known neighboring words. Obtaining training data for this task is very cheap, as all we need is millions of words written in existing texts. One of the most common ways of modelling language is by training a neural network that predicts the next word based on the previous ones.

In 2019, the BERT language model was introduced using a novel architecture called Transformer (2017), which allowed precisely to integrate words' context into representations. To understand BERT, we will first look at what a transformer is and we will then directly use some code to make use of BERT.

# Tranformers

Every text can be seen as a sequence of sentences and likewise each sentence can be seen as a sequence of tokens (we use the term _token_ instead of _word_ because it is more general: tokens can be words, punctuation symbols, numbers, or even sub-words). Traditionally Recurrent Neural Networks (RNNs; and later their fancy version, LSTMs) were used to tackle token and sentence classification problems to account for the interdependencies inherent to sequences of symbols (i.e. sentences). RNNs were in theory powerful enough to capture these dependencies, something that is very valuable when dealing with language, but in practice they were resource consuming (both in training time and computational resources) and also the longer the sequences got, the harder it was to capture long-distance dependencies succesfully.

The Transformer is a neural network architecture proposed by Google researchers [in 2017](https://arxiv.org/pdf/1706.03762) to address these and other limitations of RNNs and LSTMs. In their paper, *Attention is all you Need*, they tackled especifically the problem of Machine Translation (MT), which in NLP terms is stated as: how to generate a sentence (sequence of words) in target language B given a sentence in source language A? In order to translate, first one neural network needs to _encode_ the meaning of the source language A into vector representations, and then a second neural network needs to _decode_ that representation into tokens that are understandable in language B. Therefore translation is modeling language B _conditioned_ on what language A originally said.

![The general architecture for the Transformer](fig/trans1.png)

As seen in the picture, the original Transformer is an Encoder-Decoder network that tackles translation. We first need a token embedder which converts the string of words into a sequence of vectors that the Transformer network can process. The first component, the __Encoder__, is optimized for creating rich representations of the source sequence (in this case an English sentence) while the second one, the __Decoder__ is a generative network that is conditioned on the encoded representation and, with the help of the attention mechanism, generates the most likely token in the target sequence (in this case Dutch words) based on both the tokens generated so far and the full initial English context. 


Now that we understand the general architecture of Transformers, let's see how it was adapted for the case of BERT.

# BERT

[BERT](https://aclanthology.org/N19-1423.pdf) is an acronym that stands for **B**idirectional **E**ncoder **R**epresentations from **T**ransformers. The name describes it all: the idea is to use the power of the Encoder component of the Transformer architecture to create powerful (vectorized) token representations that preserve the contextual meaning of the whole input segment. The BERT vector representations of each token take into account both the left context (what comes before the word) and the right context (what comes after the word). Another advantage of the transformer Encoder is that it is parallelizable, which made it posible for the first time to train these networks on millions of datapoints, dramatically improving model generalization. 

::: callout
## Pretraining BERT
To obtain the BERT vector representations the Encoder is pre-trained with two different tasks:
- **Masked Language Model:** for each sentence, mask one token at a time and predict which token is missing based on the context from both sides. A training input example would be "Maria [MASK] Groningen" and the model should predict the word "loves".
- **Next Sentence Prediction:** the Encoder gets a linear binary classifier on top, which is trained to decide for each pair of sequences A and B, if sequence A precedes sequence B in a text. For the sentence pair: "Maria loves Groningen." and "This is a city in the Netherlands." the output of the classifier is "True" and for the pair "Maria loves Groningen." and "It was a tasty cake." the output should be "false" as there is no obvious continuation between the two sentences.

Already the second pre-training task gives us an idea of the power of BERT: after it has been pretrained on hundreds of thoudands of texts, one can plug-in a classifier on top and re-use the *linguistic* knowledge previously acquired to fine-tune it for a specific task, without needing to learn the weights of the whole network from scratch all over again. In the next sections we will describe the components of BERT and show how to use it. This model and hundreds of related transformer-based pre-trained encoders can also be found on [Hugging Face](https://huggingface.co/google-bert/bert-base-cased).
:::


# BERT as a Language Model

All of the following code is based on the HugingFace's _transformers_ python library. We can install it with:

```sh
pip install transformers
```

As mentioned before, the main pre-training task of BERT is Language Modeling. We can therefore directly use BERT as a predictor for word completion:


```python
from transformers import pipeline

def pretty_print_outputs(sentences, model_outputs):
    for i, model_out in enumerate(model_outputs):
        print("\n=====\t",sentences[i])
        for label_scores in model_out:
            print(label_scores)


nlp = pipeline(task='fill-mask', model='bert-base-cased', tokenizer='bert-base-cased')
sentences = ['Paris is the [MASK] of France', 'I want to eat a cold [MASK] this afternoon', 'Maria [MASK] Groningen']
model_outputs = nlp(sentences, top_k=5)
pretty_print_outputs(sentences, model_outputs)
```

We use the `pipeline` function to call the specific model we want to use. In this case `bert-base-cased` refers to the vanilla BERT English model. Once we declared a pipelne, we can feed it with sentences that contain one masked token at a time (beware that BERT can only predict one word at a time, since that was its training scheme). We request the pipelne to provide us with the top 5 most likely suggestions to complete the sentences. 

In the outputs for the first example it shows correctly that the missing token in the first sentence is _capital_, the second example is a bit more ambiguous, but the model at least uses the context to correctly predict a series of items that can be eaten (unfortunately, none of its suggestions sound very tasty); finally, the third example gives almost no useful context so the model plays it safe and only suggests prepositions or punctuation. This already shows some of the weaknesses of the approach.

# BERT Architecture

Now that we used the BERT language model component we can dive into the architecture of BERT to understand it better.

As in any basic NLP pipeline, the first step is to pre-process the raw text so it is ready to be fed into the Transformer. Tokenization in BERT does not happen at the word-level but rather splits texts into what they call WordPieces (the reason for this decision is complex, but in short, researchers found that splitting *human words* into *subtokens* exploits better the character sub-sequences inside words and helps the model converge faster). A word then sometimes is decomposed into one or several (sub) tokens.

1. **Tokenizer:** splits text into tokens that the model recognizes
2. **Embedder:** converts each token into a fixed-sized vector that represents it. These vectors are the actual input for the Encoder.
3. **Encoder** several neural layers that model the token-level interactions of the input sequence to enhance meaning representation
5. **Output Layer:** the final encoder layer contains arguably the best token-level representations that encode syntactic and semantic properties of each token, but this time each vector is contextualized within the specific sequence.
6. *OPTIONAL* **Classifier Layer:** an additional classifier can be connected on top of the BERT token vectors which are used as features for performing a downstream task. This can be used to classify at the text level, for example sentiment analysis of a sentence, or at the token-level, for example Named Entity Recognition.

![BERT Architecture](fig/bert3.png)

We will next see the case of combining BERT with a classifier on top.

# BERT for Text Classification

The task of text classification is assigning a label to a whole sequence of tokens, for example a sentence. With the parameter `task="text_classification"` the `pipeline()` function will load the base model and automatically add a linear layer with a softmax on top. This layer can be fine-tuned with our own labeled data or we can also directly load the fully pre-trained text classification models that are already available in HuggingFace.

![BERT as an Emotion Classifier](fig/bert4.png)

Let's see the example of an emotion classifier based on `RoBERTa` model. This model was fine-tuned in the Go emotions [dataset](https://huggingface.co/datasets/google-research-datasets/go_emotions) which is annotated data taken from English Reddit. The fine-tuned model is called [roberta-base-go_emotions](https://huggingface.co/SamLowe/roberta-base-go_emotions). This model takes a sentence as input and ouputs a probability distribution over 28 possible emotions that are conveyed in the text. For example:

```python

classifier = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=3)

sentences = ["I am not having a great day", "Maria loves Groningen"]
model_outputs = classifier(sentences)

pretty_print_outputs(sentences, model_outputs)
```

This code outputs the Top-3 emotions that each of the two sentences convey. In this case, the first sentence evokes (in order of likelihood) *dissapointment*, *sadness* and *annoyance*; whereas the second sentence evokes *love*, *neutral* and *approval*. Note however that the likelihood of each prediction decreases dramatically below the top choice, so perhaps this specific classifier is only useful for the top emotion.

::: callout
Because the _classifier_ is a very small neural network, it can be quickly trained to choose between the classes for your custom classification problem. As mentioned before, this classifier is just a one-layer neural layer with a softmax that assigns a score that can be translated to the probability over the label classes (in this case 1 of the 28 emotions) given the input hidden state provided by BERT, which _encodes_ the meaning of the entire sequence in it.  
:::

![BERT as an Emotion Classifier](fig/bert4b.png)


# Understanding the Tokenizer and Embedder


We should look now more in detail at how the tokenizer is working. To do this we can load the pre-trained tokenizer and model and play a bit with them. We can feed a sentence into the tokenizer to observe how it outputs a sequence of vectors (also called a *tensor*: by convention, a vector is a sequence of scalar numbers, a matrix is a 2-dimensional sequence and a tensor is a N-dimensional sequence of numbers), each one of them representing a wordPiece:

```python

# Load model and tokenizer
from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = BertModel.from_pretrained("bert-base-cased")

# Feed text into the tokenizer 
text = "Maria's passion for music is clearly heard in every note and every enchanting melody."
encoded_input = tokenizer(text, return_tensors='pt')
token_ids = list(encoded_input.input_ids[0].detach().numpy())
print(token_ids)

```

This shows a list of 21 token IDs (the ID identifies each token in the embedding layer of the transformer). To see the actual string representation of 21 tokens we can convert the IDs like this:

```python

string_tokens = tokenizer.convert_ids_to_tokens(token_ids)
print(string_tokens)

```

We can see that most "words" were converted into a single token, however *enchanting* was splitted into three sub-tokens: `'en', '##chan', '##ting'` the hashtags indicate wether a sub-token was part of a bigger word or not, this is useful to recover the human-readable strings later. The `[CLS]` token was added at a beginning and is intended to represent the meaning of the whole sequence, likewise the `[SEP]` token was added to indicate that it is where the sentence ends.

The next step is to give the sequence of tokens to the Encoder which processes it through the transformer layers and outputs a sequence of dense vectors, each one of them representing one contextualized token piece in the embedded space. The most basic approach is to only retrieve the last layer of the encoder sequence, since it arguably holds the highest-level global information of the sequence.

```python
output = model(**encoded_input)
print(output.last_hidden_state.shape)
print(output.last_hidden_state[0])
```

Once again we can look at the shape of the output and see that it holds a tensor of dimensions `[1, 21, 768]`. The **first dimension is the batch size** (to process several sequences in parallel), the **second dimension is the number of tokens** in the sequences and the **third dimension is always 768** with BERT since those were the dimensions chosen by the creators of the model. 

When we print the vectors we only see a lot of fine-tuned weights which are not very informative in their own, but the full-vectors are meaningful within the embedding space, which emulates some aspects of linguistic meaning. In the case of wanting to obtain a single vector for *enchanting*, you can average the three vectors that belong to the token pieces that ultimately from that word.

We can use the same method to encode another sentence with the word *note* to see how BERT actually handles polysemy thanks to the representation of each word being contextualized instead of isolated.

```python
# Search for the index of 'note' and obtain its vector from the sequence
note_index_1 = string_tokens.index('note')
note_vector_1 = output.last_hidden_state[0][note_index_1].detach().numpy()
note_token_id_1 = token_ids[note_index_1]

# Encode and then take the 'note' token from the second sentence
note_text_2 = "I could not buy milk in the supermarket because the bank note I wanted to use was fake."
encoded_note_2 = tokenizer(note_text_2, return_tensors='pt')
token_ids = list(encoded_note_2.input_ids[0].detach().numpy())
string_tokens_2 = tokenizer.convert_ids_to_tokens(token_ids)

note_index_2 = string_tokens_2.index('note')
note_vector_2 = model(**encoded_note_2).last_hidden_state[0][note_index_2].detach().numpy()
note_token_id_2 = token_ids[note_index_2]

print(note_index_1, note_token_id_1, string_tokens)
print(note_vector_1[:5])
print(note_index_2, note_token_id_2, string_tokens_2)
print(note_vector_2[:5])
```

We can compute the cosine similarity of the word *note* in the first sentence and the word *note* in the second sentence to confirm that they are indeed two different representations, even when in both cases they have the same token-id and they are the 12th token of the sentence:

```python
from sklearn.metrics.pairwise import cosine_similarity

vector1 = np.array(note_vector_1).reshape(1, -1)
vector2 = np.array(note_vector_2).reshape(1, -1)

similarity = cosine_similarity(vector1, vector2)
print(f"Cosine Similarity 'note' vs 'note': {similarity[0][0]}")
```

# Understanding the Attention Mechanism

The original attention mechanism is a component in between the Encoder and the Decoder that helps the model to _align_ the important information from the input sequence in order to generate a more accurate token in the output sequence:

![The Attention Mechanism](fig/trans3.png)

In the example above, the attention puts more weight in the input _etudiant_, so the decoder uses that information to _know_ that is should generate _student_. Note that if the decoder based it's next work probability just based in the sequence "I am a ..." it could basically generate any word and still sound natural, but it is thanks to the attention mechanism that it preserves the meaning of the input sequence.

Attention is a neural layer, therefore it can also be plugged-in within the Encoder, this is called self-attention since the mechanism will look at the interactions between the input sequence and the input sequence. This is how BERT uses (self-) attention, which is very useful to capture longer-range word dependencies such as correference, where, for example, a pronoun can be linked to the noun it refers to previously in the same sentence. See the following example:
![The Attention Mechanism](fig/trans5.png)

There are two sentences, in each one the pronoun "it" refers to a different noun, "animal" or "street", and this is completely depending on the sentence context. Thanks to the self-attention BERT relates the pronoun to its relevant correferent.

# BERT for Token Classification

Just as we plugged in a trainable text classifier layer, we can add a token-level classifier that assigns a class to each of the tokens encoded by a transformer. A specific example of this task is Named Entity Recognition.

## Named Entity Recognition

Named Entity Recognition (NER) is the task of recognizing mentions of real-world entities inside a text. The concept of entity includes proper names that unequivocally identify a unique individual (PER), place (LOC), organization (ORG), or object (MISC). Depending on the domain, the concept has been expanded to recognize other unique (and more conveptual) entities such as DATE, MONEY, WORK_OF_ART, etcetera. In terms of NLP, this boils down to classifying each token into a series of labels (PER, LOC, ORG, O). Since a single entity can be expressed with multiple words (e.g. New York) the usual notation used for labeling the text is IOB (**I**nner **O**ut **B**eginnig of entity) notations which identifies the limits of each entity tokens. For example:

![BERT as an NER Classifier](fig/bert5.png)

This is a typical sequence classification problem where an imput sequence must be fully mapped into an output sequence of labels with unique constraints (for example, there can't be an inner I-PER label before a beginning B-PER label). Since the labels of the tokens are context dependent a language model with attention mechanism such as BERT is very beneficial for a task like NER.

Because this is one of the core tasks in NLP, there are dozens of pre-trained NER classifiers in HuggingFace that you can use right away. Since the task is similar to any other task, that is, just add a classification layer on top of a pre-trained transformer you can still use the `pipeline()` function to run the model in your custom data, in this case with `task="ner"`. For example:

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

ner_classifier = pipeline("ner", model=model, tokenizer=tokenizer)
example = "My name is Wolfgang Schmid and I live in Berlin"

ner_results = ner_classifier(example)
for nr in ner_results:
    print(nr)
```

This prints all the entity labels that were found in the provided text. Note that the labels are assigned at the token level (wordPieces), and there is also an indication of the substring in the original sentence where you can recover these entities. You can assume all the rest of the tokens qwere labeled as no-entity, that is `"O"`. To recover the full-word entities you can initialize the pipeline with `aggregation_strategy="simple"`:

```python
ner_classifier = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
example = "My name is Wolfgang Schmid and I live in Berlin"

ner_results = ner_classifier(example)
for nr in ner_results:
    print(nr)
```

As you can see, the entities now are given at the Span Leven instead of the Token Level (that is, multiword entities are assigned a single entity label).

The next very important step is to evaluate how does the pre-trained model actually performs in **your dataset**. This is important since the fine-tuned model could be overfitted to other custom benchmarks that do not share the characteristics of your dataset.

## Using a Pre-trained Model on LitBank

Now that you know how to use the NER tagger you can apply it to a custom dataset. We will use the [LitBank](https://github.com/dbamman/litbank) corpus, an annotated dataset of 100 works of English-language fiction to support tasks in natural language processing and the computational humanities. Specifically they have human annotations of entities on these books. We can measure how good is this pre-trained classifier by making the model predict the entities inside the text and them compare the outputs with the humam annotations. The NER portion of the dataset we will use is the tabulated data from [here](https://github.com/dbamman/litbank/tree/master/entities/tsv) and one example looks like this:

| Index 	| Token    	| IOB-1 	| IOB-2 	| IOB-3 	| IOB-4 	|
|-------	|----------	|-------	|-------	|-------	|-------	|
| 1     	| CHAPTER  	| O     	| O     	| O     	| O     	|
| 2     	| I        	| O     	| O     	| O     	| O     	|
| 3     	| In       	| O     	| O     	| O     	| O     	|
| 4     	| Chancery 	| B-FAC 	| O     	| O     	| O     	|
| 5     	| London   	| B-GPE 	| O     	| O     	| O     	|
| 6     	| .        	| O     	| O     	| O     	| O     	|

It contains the information of 4 annotators, this is very useful interannotator agreement, a technique in computational linguistics for validating the correctness and consistency of the dataset. Yes! Humans are wrong too all the time when labeling! For simplicity, we will assume we only have the information from annotator 1  and take that as our ground truth. 

The format of the dataset resembles the conll format, a widely used format in computational linguistics for token-based annotations. Another important aspect to observe is that they have other labels for entities. The pre-trained model we chose only labels PER, LOC, ORG and MISC. We can translate FAC and GPE to LOC label as they are only more fine-grained occurrences of locations which our model should recognize as such. To read the data we can use the following function:

```python

def quick_conll_reader(filepath):
    all_sentences, all_labels = [], []
    sent_txt, sent_lbl = [], []
    gold_label_column = 1
    label_translator = {
        "B-FAC": "B-LOC",
        "I-FAC": "I-LOC",
        "B-GPE": "B-LOC",
        "I-GPE": "I-LOC",
        "B-VEH": "O",
        "I-VEH": "O"
    }
    with open(filepath) as f:
        for line in f.readlines():
            row = line.strip().split("\t")
            if len(row) > 1:
                sent_txt.append(row[0])
                label = row[gold_label_column]
                if label in label_translator:
                    sent_lbl.append(label_translator[label])
                else:
                    sent_lbl.append(label)
            else:
                all_sentences.append(" ".join(sent_txt))
                all_labels.append(sent_lbl)
                sent_txt, sent_lbl = [], []
    return all_sentences, all_labels


sentences, gold_labels = quick_conll_reader("1023_bleak_house_brat.tsv")
print(sentences[0].split(' '))
print(gold_labels[0])
```

This code processes the *Bleak House* book and extracts a list of tokenized sentences (as strings) and a list of IOB Labels corresponding to each token in the sentence. You can see the first sentence and its corresponding list of *gold labels* on this example. Next, we load the NER pre-trained model again and process the sentences to obtain model predictions. The problem here is that the model predictions are lists of dictionaries and we need to post-process them so they are also on IOB-format. We use the get_iob_labels() function to do this conversion. 

```python

def token_to_spans(tokens):
    token2spans = {}
    char_start = 0
    for i, tok in enumerate(tokens):
        tok_end = char_start + len(tok)
        token2spans[i] = (char_start, tok_end)
        char_start = tok_end + 1
    return token2spans


def get_iob_labels(tokenized_sentence, entities):
    # Initialize all labels empty
    iob_labels = ['O'] * len(tokenized_sentence)
    # Get Token <-> Chars Mapping
    tok2spans = token_to_spans(tokenized_sentence)
    start2tok = {v[0]:k for k, v in tok2spans.items()}
    end2tok = {v[1]:k for k, v in tok2spans.items()}
    # Iterate over each entity to populate labels
    for entity in entities:
        label = entity['entity_group']
        if label == "MISC":  # Design choice: Do NOT count MISC entities!
            continue
        token_start = start2tok.get(entity['start'])
        token_end = end2tok.get(entity['end'])
        
        if token_start is not None:
            iob_labels[token_start] = f'B-{label}'
            if token_end is not None:
                for i in range(token_start+1, token_end+1):
                    iob_labels[i] = f'I-{label}'
    
    return iob_labels
```

And we finally apply the model to the sentences that we previously read:

```python

ner_classifier = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

ner_results = ner_classifier(sentences)
model_predictions = []
for i, sentence_ner in enumerate(ner_results):
    print(f"\n===== SENTENCE {i+1} =====")
    print('Tokens:', sentences[i].split())
    print('GOLD:', gold_labels[i])
    # Get the IOB labels for the tokenized sentence
    tokenized_sentence = sentences[i].split()
    predicted_iob_labels = get_iob_labels(tokenized_sentence, sentence_ner)
    model_predictions.append(predicted_iob_labels)
    print('MODEL:', predicted_iob_labels)
    for nr in sentence_ner:
        print(f'\t{nr}')

```

For each model prediction we are printing the sentence tokens, the IOB gold labels and the IOB predicitons. Now that the data is in this shape we can perform evaluation.

##  Model Evaluation

To evaluate a token classification model the basic metrics are Precision (of all the labeled words, how many of them are correct), Recall (of the expected gold entities, howe many of them were recognized by the model) and F1 (a harmonic mean of precision and recall that leverages both metrics and shows a unified score of performance). The three metrics range from 0 to 1, also sometimes niromalized from 0 to 100. A score of 0 means everything is wrong, and 100 means everything is correct. Many English NER benchmarks nowadays report scores above the 90 F1 scores, but this does not say anything about the performance in your own dataset.

To perform evaluation in your data

```python

from seqeval.metrics import classification_report
print(classification_report(gold_labels, model_predictions))

```

Since we took a classifier that was not trained for the book domain, the performance is quite poor. The solution in this case is to use another of the great characteristics of BERT: fine-tuning for domain adaptation. It is possible to train your own classifier with relatively small data (given that a lot of linguistic knowledge was already provided during the language modeling pre-training). In the following section we will see how to train your own NER model and use it for predictions.