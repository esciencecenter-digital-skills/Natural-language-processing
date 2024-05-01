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
Natural language processing (NLP) is an area of research and application that focuses on making natural (i.e., human) language accessible to computers so that they can be used to perform useful tasks (Chowdhury & Chowdhury, 2023). Research in NLP is highly interdisciplinary, drawing on concepts from computer science, linguistics, logic, mathematics, psychology, etc. In the past decade, NLP has evolved significantly with advances in technology to the point that it has become embedded in our daily lives: automatic language translation or chatGPT are only some examples. This evolution has enhanced its applications and expanded its interaction with fields like artificial intelligence, machine learning, reaching practically any other research field.

### Why do we care?
The past decade's breakthroughs have resulted in NLP being increasingly used in a range of diverse domains such as retail (e.g., customer service chatbots), healthcare (e.g., AI-assisted hearing devices), finance (e.g., anomaly detection in monetary transactions), law (e.g., legal research), and many more. These applications are possible because NLP researchers developed (and constantly do so) tools and techniques to make computers understand and manipulate language effectively.

With so many contributions and such impressive advances of recent years, it is an exciting time to start bringing NLP techniques in your own work. Thanks to dedicated python libraries, these tools are now more accessible. They offer modularity, allowing you to integrate them easily in your code, and scalability, i.e., capable of processing vast amounts of text efficiently. 

These tools are easily accessible via dedicated python libraries that allow for modularity (i.e., you can build upon those in your code) and scalability (i.e., you can process vast amount of text) without necessarily being an advanced python programmer. Whether dealing with text or audio, NLP tools provide a means to handle and interpret language data to meet specific needs and objectives. Even those without advanced programming skills can leverage these tools to address problems in social sciences, humanities, or any field where language plays a crucial role. In a nutshell, NLP opens up possibilities, making sophisticated techniques accessible to a broad audience. 

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

In this lesson we are going to see the NER and topic modeling tasks in detail, and learn how to develop solutions that work for these particular use cases. Specifically, our goal in this lesson will be to identify characters and locations in novels, and determine what are the most relevant topics in these books. However, it is useful to have an understanding of the other tasks and its challenges.

### Text classification

The goal of text classification is to assign a label category to a text or a document based on its content. This task is for example used in spam filtering - is this email spam or not - and sentiment analysis; is this text positive or negative.

### Information extraction

With this term we refer to a collection of techniques for extracting relevant information from the text or a document and finding relationships between those. This task is useful to discover cause-effects links and populate databases. For instance, finding and classifying relations among entities mentioned in a text (e.g., X is the child of Y) or geospatial relations (e.g., Amsterdam is north of Bruxelles)

### Named Entity Recognition (NER)

The task of detecting names, dates, language names, events, work of arts, countries, organisations, and many more.

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

The task of translating a piece of text from one language to another. 
