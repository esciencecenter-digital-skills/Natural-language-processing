---
title: "Welcome"
teaching: 5
exercises: 0
---

:::::: questions
- Who is this lesson for?
- What will be covered in this lesson?
::::::

:::::: objectives
- Identify the target audience
- Identify the learning goals of the lesson
::::::

# Welcome
This course covers core concepts of Natural Language Processing (or NLP) and it is designed to equip researchers in the humanities and social sciences with the foundational skills and knowledge needed to carry over text-based research projects. NLP refers to a set of techniques involving the application of statistical methods to understand natural (i.e, human) language for the sake of solving real-world tasks.

## What will we be covering in this lesson?

This lesson provides a high-level introduction to NLP with particular emphasis on applications in the humanities and the social sciences. We will focus on solving particular problems over the lesson. Some problems deal with capturing the meaning of a word (e.g., `head`) and how this changes over time and context (e.g., `top part of the body` vs `leaders of others`), others with identifying key entities in text (such as people, places, companies, dates and more) in literary texts and labeling each one of them with the right category name. These problems are examples of useful applications in your own research, however, they also offer a window in the latest NLP advancements that are now embedded in our daily life.

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

In this lesson, we will provide you with the building blocks to know which problems these tools solve, and how, and what are the foundational theoretical advances that made them possible. We will do so by working on small-scale datasets and models that can easily run on our laptop. 

After following this lesson, learners will be able to:

- Explain and differentiate what are the core topics in NLP
- Identify what kinds of tasks NLP techniques excel at, and what are their limitations
- Structure a typical NLP pipeline
- Extract vector representations of individual words, visualise it and manipulate it
- Applying a machine learning algorithm to textual data to extract and categorise names of entities (e.gs., places, people)
- Using natural language to produce a desired response from a large language model (LLM), i.e. prompt engineering
- Other?

## Software packages required
The lesson is coded entirely in Python. We are going to use Jupyter notebooks throughout the lesson and the following packages:

- spacy
- gensim
- transformers
- [to be updated at the end of development]

## Dataset
In this lesson, we'll use N books from the [Project Gutenberg](https://www.gutenberg.org/). We will use their Plain Text UTF-8 version.

- The Adventures of Sherlock Holmes by Arthur Conan Doyle - [Full text](https://www.gutenberg.org/cache/epub/1661/pg1661.txt) - [Wikipedia](https://en.wikipedia.org/wiki/The_Adventures_of_Sherlock_Holmes)
- The Count of Monte Cristo by Alexandre Dumas - [Full text](https://www.gutenberg.org/cache/epub/1184/pg1184.txt) - [Wikipedia](https://en.wikipedia.org/wiki/The_Count_of_Monte_Cristo)


:::::: keypoints 
- This lesson on Natural language processing in Python is for researchers working in the field of Humanities and/or Social Sciences
- This lesson is an introduction to NLP and aims at implementing first practical NLP applications from scratch 
::::::

