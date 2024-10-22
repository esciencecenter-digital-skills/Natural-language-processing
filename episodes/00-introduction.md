---
title: "Introduction"
teaching: 30
exercises: 10
---

:::::: questions
- What is NLP?
- What will be covered in this lesson?
- Which datasets will be used?
::::::

:::::: objectives
- Identify the target audience
- Identify the learning goals of the lesson
- Get data that will be used in the course
::::::

### What is NLP?
Natural language processing (NLP) is an area of research and application that focuses on making natural (i.e., human) language accessible to computers so that they can be used to perform useful tasks. Research in NLP is highly interdisciplinary, drawing on concepts from computer science, linguistics, logic, mathematics, psychology, etc. In the past decade, NLP has evolved significantly with advances in technology, especially in the field of deep learning, to the point that it has become embedded in our daily lives: automatic language translation or chatGPT are only some examples. 

:::::::::::: challenge 
## NLP in the real world
Many of the concepts and problems we will tackle in this NLP course have a tangible impact in your daily life. Let's see how.

Your task is to name three to five tools/products that you use on a daily basis and that you think leverage NLP techniques. To solve this exercise you can get some help from the web.

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

The exercise above tells us that a great deal of NLP techniques is embedded in our daily life. What is that is working under the hood of these technologies?

## What will we be covering in this lesson?

This lesson provides a high-level introduction to NLP with particular emphasis on applications in the humanities and the social sciences. We will focus on solving particular problems over the lesson. Some problems deal with capturing the meaning of a word (e.g., `head`) and how this changes over time and context (e.g., `top part of the body` vs `leaders of others`), others with identifying key entities in text (such as people, places, companies, dates and more) in literary texts and labeling each one of them with the right category name. These problems are examples of useful applications in your own research, however, they also offer a window in the latest NLP advancements that are now embedded in our daily life.

After following this lesson, learners will be able to:

- Explain and differentiate what are the core topics in NLP
- Identify what kinds of tasks NLP techniques excel at, and what are their limitations
- Structure a typical NLP pipeline
- Extract vector representations of individual words, visualise and manipulate them
- Applying a machine learning algorithm to textual data to extract and categorise names of entities (e.gs., places, people)
- Using natural language to produce a desired response from a large language model (LLM), i.e. prompt engineering

## Datasets
We will be using the following material.

For the episode 01: preprocessing and word embeddings (Word2Vec):

- An excerpt from a Dutch journal of your choice dated from the 1950 up until 1989. This can be freely downloaded as a `txt` from [Delpher](https://www.delpher.nl/nl/kranten)
- Word2Vec models trained on 6 national Dutch newspaper data spanning a time period from 1950 to 1989 (Wevers, M., 2019). These models are available on [Zenodo](https://zenodo.org/records/3237380).

-
-

For the episode 03: LLM and prompt engineering
- 
-

Ensure that you have downloaded the following material before diving into the course. 

:::::: keypoints 
- This lesson on Natural language processing in Python is for researchers working in the field of Humanities and/or Social Sciences
- This lesson is an introduction to NLP and aims at implementing first practical NLP applications 
::::::

