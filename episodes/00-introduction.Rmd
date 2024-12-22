---
title: "Introduction"
teaching: 50
exercises: 10
---

:::::: questions
- What is NLP?
- What are real-world applications of NLP?
- Which problems NLP solves best?
- What is language from a NLP perspective?
- How does NLP relates to Deep Learning and Machine Learning?
::::::

:::::: objectives
- Define Natural Language Processing
- Detailing classic NLP tasks and applications in practice
- Describe the theoretical perspectives that the field of NLP draws upon, including linguistics (syntax, semantics, and pragmatics), Deep Learning and Machine Learning
::::::

## What is NLP?
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

### Other types of applications

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

Why does NLP have "natural" in its name? A [Language](https://en.wikipedia.org/wiki/Language) is a structured system of communication that consists of grammar and vocabulary. Within this definition, in NLP we refer to human language as [*Natural*](https://en.wikipedia.org/wiki/Natural_language) to contrast it to *artificial* languages which are [formal languages](https://en.wikipedia.org/wiki/Formal_language). The reason for this is that many experts believe that *natural* languages have emerged *naturally* tens of thousands of years ago and have evolved ever since. Formal languages, on the other hand, are languages that have been engineered by humans and have rigid and explicitly defined rules.

To understand this perspective, let's consider for instance Python or R. These are [programming languages](https://en.wikipedia.org/wiki/Programming_language) that have explicit, clear grammatical and syntactic rules. This means that within the realm of those programming languages, there is no room for ambiguity, otherwise your code would allow for different behaviours depending on the situation, or the machine. This is not the case for human languages. 

### Ambiguity
Natural language is highly creative and often ambiguous, among many other complex traits. A sentence of the type "I saw a bat" may mean many things depending on who is hearing/saying it, where  and when it is pronounced. The disambiguation of meaning is usually a by-product of the context in which sentences are pronounced and the historic accumulation of interactions which are transmitted across generations (think for instance to idioms -- these are usually meaningless phrases that acquire meaning only if situated within their historical and societal context). These characteristics make NLP a particularly challenging field to work in. 

We cannot expect a machine to process human language and simply understand it as it is. We need a systematic, scientific approach to deal with it. It's within this premise that the field of NLP is born, primarily interested in converting the building blocks of human/natural language into something that a machine can understand. We'll see what does this mean in the next episode.

The image below shows you the building blocks of language and a few NLP applications that leverage this type of information.

![Diagram showing building blocks of language](fig/intro.pdf)

Each building block of human language carries a large amount of information, which we process quickly and effortlessly. Some of this information is still being studied by scientists because it’s unclear how to measure it, whether the human brain uses it at all to aid understanding, and, if so, to what extent. A lot of research effort is spent on this problem in academia, and it's important to keep in mind that we are still far from solving it.

How do we make language then understandable for machines? How do we expose and exploit the statistical information within the human language? The field of NLP focuses exactly on these challenges. The ultimate goal is to make this information available to computers, so that they can use it to understand language as closely as possible to the way we (humans) do. 

### Discreteness
NLP is a subfield of Artificial Intelligence that intersects with [Deep Learning](https://carpentries-incubator.github.io/deep-learning-intro/1-introduction.html) and more broadly with [Machine Learning](https://en.wikipedia.org/wiki/Machine_learning). There are many concepts in NLP that indeed draw upon those fields. For instance the task of categorising text as positive or negative, is a classification [problem](https://carpentries-incubator.github.io/deep-learning-intro/2-keras.html#formulateoutline-the-problem-penguin-classification) that has been formulated and solved also in the Deep Learning realm. What's the difference then between classifying which species a penguin belongs to (based on their pictures) and understand the difference between "cat" and "sat"?

If you take an image of a penguin and change a pixel it will still be recognised as the same penguin as before. This tiny change has resulted in a small change that did not affect the whole picture. If you change one letter of a word, however, as in cat vs sat, then even if the difference for the computer is a single bit, the two things in the human language are two separate, *discrete* concepts. They just happen (for historical reasons or just by chance) to have similar spellings. 

The reason why NLP is a distinct field is that, unlike images and sounds (which are typically handled in Deep Learning and are continuous data), words are discrete units. This characteristic of human language demands a completely different approach because, while computers excel at processing continuous variables, they struggle with the discrete nature of language. In the next episode, we’ll explore how a solution to this challenge has only recently been developed.


:::::: keypoints 
- NLP is embedded in numerous daily-use products
- Key tasks include language modeling, text classification, information extraction, information retrieval, conversational agents, and topic modeling, each supporting various real-world applications.
- NLP is a subfield of Artificial Intelligence (AI) that deals with approaches to process, understand and generate natural language
- Deep learning has significantly advanced NLP, but the challenge remains in processing the discrete and ambiguous nature of language
- The ultimate goal of NLP is to enable machines to understand and process language as humans do, but challenges in measuring and interpreting linguistic information still exist.
::::::

