---
title: "Episode 1"
teaching: 10
exercises: 2
---

:::::: questions 
- What is natural language processing (NLP)?
- Why is it important to learn about NLP?
- What are some classic tasks associated with NLP?
::::::

:::::: objectives
- Recognise the importance and benefits of learning about NLP
- Identify and describe classic tasks and challenges in NLP 
- Explore practical applications of natural language processing in industry and research
::::::

## Introducing NLP

### What is NLP?
Natural language processing (NLP) aims to develop methods for solving practical problems involving language.
In literature, it is common to see a crossover with the field of Computational Linguistics (CL), whereby the focus
is on employing computational methods to understand properties of human language. In recent years, the field of NLP
has undergone a dramatic shift, both in terms of methodology and of the applications supported thanks to the advances
in computer hardware. This made several other crossovers with related fields possible, among which: Computer science,
artificial intelligence, machine learning, statistics, psycholinguistics, neuroscience and many others.

### Why do we care?
The past decade's breakthroughs have resulted in NLP being increasingly used in a range of diverse domains such as retail,
healthcare, finance, law, marketing and many more. At the same time, an increased interest in Digital Humanities and Social
Sciences textual datasets within the NLP community has prompted the development of computational techniques to analyse and model
humanities and social sciences data. To date, a number of successful results have been achieved, in the area of handwriting recognition [cit],
computational stylistics [cit], historical corpora [cit], semantic change detection [cit] and sentiment analysis [in social media, cit, in politics, cit], to name a few examples.

With so many contributions and such impressive advances of recent years, it is an exciting time to start bringing NLP techniques
in our own work. 

:::::::::::: challenge 
## NLP in the real world

Name three to five products that you use on a daily basis and that rely on NLP techniques. To solve this exercise you can get 
some help from the web.


:::::: solution
These are some of the most popular NLP-based products that we use on a daily basis:

- Voice-based assistants (e.g., Alexa, Siri, Cortana)
- Machine translation (e.g., Google translate, Amazon translate)
- Search engines (e.g., Google, Bing, DuckDuckGo)
- Keyboard autocompletion on smartphones
- Spam filtering
- Spell and grammar checking apps
::::::
::::::::::::

## What is NLP typically good at?

Here's a collection of fundamental tasks in NLP:

- Text classification
- Information extraction 
- NER (named entity recognition)
- Next word prediction
- Text summarization
- Question answering
- Topic modeling
- Machine translation
- Conversational agent

In this lesson we are going to see the NER and topic modeling tasks in detail, and learn how to develop solutions that work for these particular use cases.
Specifically, our goal in this lesson will be to identify characters and locations in novels, and see what are the most relevant topics in these books.
However, it is useful to have an understanding of the other tasks and its challenges.

### Text classification

This task requires assigning a label category to a text or a document based on its content. This task is used in spam filtering and sentiment analysis.

### Information extraction

With this term we refer to a collection of techniques for extracting relevant information from the text or a document. 
Useful to discover cause-effects links and populate databases.

### Named Entity Recognition (NER)

The task of detecting names, dates and organisations. 

### Next word prediction

This task involves predicting what the next word in a sentence will be based on the history of previous words.
Speech recognition, spelling correction, handwriting recognition all run an implementation of this task.

### Text summarization 

Create short summaries of longer documents while retaining the core content. 

### Question answering 

Task of building a system that answer questions posed in natural (i.e., human) language.

### Topic modeling 

Task of discovering topical structure in documents.

### Machine translation

TAsk of converting a piece of text from one language to another. 
