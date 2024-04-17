---
title: "Welcome"
teaching: 5
exercises: 0
---

:::::::::::::::::::::::::::::::::::::: questions

- Who is this lesson for?
- What will be covered in this lesson?

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Identify the target audience
- Identify the learning goals of the lesson

::::::::::::::::::::::::::::::::::::::::::::::::

# Welcome
This is a hands-on introduction to Natural Language Processing (or NLP). NLP refers to a set of techniques involving the application of statistical methods, 
with or without insights from linguistics, to understand natural (i.e, human) language for the sake of solving real-world tasks.

This course is designed to equip researchers in the humanities and social sciences with the foundational
skills needed to carry over text-based research projects. 

## What will we be covering in this lesson?

This lesson provides a high-level introduction to NLP with particular emphasis on applications in the humanities and the social
sciences. We will focus on solving a particular problem over the lesson, that is how to identify key info in text (such as people,
places, companies, dates and more) and labeling each one of them with the right category name. Towards the end of the lesson,
we will cover also other types of applications (such as topic modelling, and text generation).

After following this lesson, learners will be able to:

- Explain and differentiate what are the core topics in NLP
- Identify what kinds of tasks NLP techniques excel at, and what are their limitations
- Structure a typical NLP pipeline
- Extract vector representations of individual words, visualise and manipulate it
- Applying a machine learning algorithm to textual data to extract and categorise names of entities (e.gs., places, people)
- Apply popular tools and libraries used to solve other tasks in NLP (such as topic modelling, and text generation)

## Software packages required
The lesson is coded entirely in Python. We are going to use Jupyter notebooks throughout the lesson and the following packages:

- spacy
- stanza
- gensim
- transformers

## Dataset
In this lesson, we'll use N books from the [Project Gutenberg](https://www.gutenberg.org/). We will use their Plain Text UTF-8 version.

- The Adventures of Sherlock Holmes by Arthur Conan Doyle - [Full text](https://www.gutenberg.org/cache/epub/1661/pg1661.txt) - [Wikipedia](https://en.wikipedia.org/wiki/The_Adventures_of_Sherlock_Holmes)
- The Count of Monte Cristo by Alexandre Dumas - [Full text](https://www.gutenberg.org/cache/epub/1184/pg1184.txt) - [Wikipedia](https://en.wikipedia.org/wiki/The_Count_of_Monte_Cristo)


::::::::::::::::::::::::::::::::::::: keypoints 
- A lesson on Natural language processing in Python is for researchers working in the field of Humanities and/or Social Sciences
- This lesson is an introduction to NLP and aims at implementing first practical NLP applications from scratch 