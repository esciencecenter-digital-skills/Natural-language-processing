---
title: "Introduction"
teaching: 30
exercises: 10
---

:::::: questions
- What is NLP?
- What are real-world applications of NLP (Natural Language Processing)?
- Which problems NLP solves best?
- What is language from a NLP perspective?

::::::

:::::: objectives
- Identify the target audience
- Identify the learning goals of the lesson
- Get data that will be used in the course
::::::

### What is NLP?
Natural language processing (NLP) is an area of research and application that focuses on making natural (i.e., human) language accessible to computers so that they can be used to perform useful tasks. Research in NLP is highly interdisciplinary, drawing on concepts from computer science, linguistics, logic, mathematics, psychology, etc. In the past decade, NLP has evolved significantly with advances in technology, especially in the field of deep learning, to the point that it has become embedded in our daily lives. 

Let's start by looking at some popular applications you use in everyday life that have some form of NLP component. 

:::::::::::: challenge 
## NLP in the real world

Name three to five tools/products that you use on a daily basis and that you think leverage NLP techniques. To solve this exercise you can get some help from the web.

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

The exercise above tells us that a great deal of NLP techniques is embedded in our daily life. Indeed NLP is an important component in a wide range of software applications that we use in our daily lives.

## Core applications

- Email providers use NLP in several ways: automatically detect and filter-out spam emails, classify important emails (e.g., Priority inbox), recognise dates and events to automatically add them to your calendar and suggesting you phrases while you're typing

- Voice-based assistants use NLP to recognise speech, interpret requests (e.g., "Set alarm for 8 AM tomorrow") and perform it accurately, translate spoken language in real time and store individual preferences and history to tailor responses based on the past activities with the user

- Search engines use NLP to interpret the meaning behind user queries (e.g., "What's the best restaurant near me?"), pull and highlight key information directly from a webpage to answer your query and personalise results based on user history

### Other types of applications:

- Customer care services use NLP to summarise and understand user reviews to provide actionable insights to their companies

- Spelling and grammar correction tools use NLP to highlight typos or errors and suggest the most valid alternative

- The [Historical Archives of the European Parliament](https://historicalarchives.europarl.europa.eu/en/sites/historicalarchive/home/cultural-heritage-collections/news/ai-dashboard.html) have used NLP to instantly search, retrieve and understand decades of legislative documents and parliamentary proceedings in multiple languages

## NLP tasks

- Language modeling: Given a sequence of words, the model predicts the next word. For example, in the sentence "The cat is on the _____", the model might predict "mat" based on the context. This task is useful for building solutions that require speech and optical character recognition (even handwriting), language translation and spelling correction

- Text classification: Given a set of items (e.g., emails), assign a label (e.g., spam/not-spam). It is the task of assigning predefined categories or labels to a given text. Text classification is extremely popular in NLP applications, from spam filtering to movies ratings based on reviews.

- Information extraction: This is the task of extracting relevant information from the text. "Eva Viviani, a Research Software Engineer at the eScience Center, attended the 17th Conference of the European Chapter of ACL on May 2nd, 2023". Person: Eva Viviani, Job title: RSE, Event: 17th Conference of the European Chapter of ACL, Date: May 2nd, 2023, etc. 

- Information retrieval: This is the task of finding relevant information or documents from a large collection of unstructured data based on user's query, e.g., "What's the best restaurant near me?".

- Conversational agent (also known as ChatBot): Building a system that interacts with a user via natural language, e.g., "What's the weather today, Siri?". These agents are widely used to improve user experience in customer service, personal assistance and many other domains.

- Topic modelling: Automatically identify abstract "topics" that occur in a set of documents, where each topic is represented as a cluster of words that frequently appear together. This task is used in a variety of domains, from literature to bioinformatics as a common text-mining tool.

## Natural vs Artificial Language

[Language](https://en.wikipedia.org/wiki/Language) is a structured system of communication that consists of grammar and vocabulary. Within this definition, we refer to Human language as [*Natural* language](https://en.wikipedia.org/wiki/Natural_language), as opposed to *artificial* languages which includes, for instance, [programming languages](https://en.wikipedia.org/wiki/Programming_language) or [formal languages](https://en.wikipedia.org/wiki/Formal_language).

It is within these definitions that the field of NLP is born, primarily interested in converting the building blocks of human/natural language into something that a machine can process.

The image below shows you the building blocks of language and a few NLP applications that leverage this type of information.

![building blocks of language]('https://github.com/esciencecenter-digital-skills/Natural-language-processing/tree/episodes/fig/intro.pdf)

It's important to remember that each one of the building blocks of the human language carry vast amount of information, and the field of NLP focuses exactly making this information available to computers.

## NLP challenges

Natural language is highly creative and often ambiguous, among many other complex traits. These characteristics make NLP a particularly challenging field to work in. 

### Ambiguity

Ambiguity refers to the characteristics of the human language to refer to multiple meanings. Sisambiguation is possible only considering the context. For instance, the sentence "The ball is in your court" means that it's now someone else's turn to take action, i.e., it's their responsibility to move next. To actually grasp better its meaning, we must use it in a context like this: "Now that you've got all the info, the ball is in your court". 

:::::: keypoints 

::::::

