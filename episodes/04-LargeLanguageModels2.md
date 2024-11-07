---
title: "Episode 4: Using large language models"
teaching: 
exercises: 
---

:::::: questions 
- What is a large language model
...
::::::

:::::: objectives
At the end of this episodes, the learners will:
- 
::::::

## This episode
- what are we going to discuss in this episode
- which challenges will we tackle
general description

## What are Large Language Models?
- wat is it
- What are LLMs good at and what not
-- startup simple chat? (code along)
- how are they different from other NLP techniques
-- challenge: something with having them try out stuff that does not work? (code)
- how do they work? (how is it that you can chat with the model) -> something (very globally) on the architecture?

## Examples of existing LLMs
### A zoo of Large Language Models
The era of Large Language Models started with the release of the model called BERT, that we discussed in the previous episode. The techniques used in that model started a movement of the creating of many new models. There are a number of big companies that keep improving on their models and releasing new ones by the XXX. The most famous one that we all probably heard of is ChatGPT, developed by OepnAI. The first version of ChatGPT was released in XXX. Since then, Google released XXX updated versions. A downside of ChatGPT is that the model is not open source, not fully available for free, and most of all that it is unclear what it was trained on. 

Other models: UPDATE
- Llama, now at version 3.2 - Meta
- Mixtral - Mistral
- Gemini - Google
- Claude - Anthropic
- Phi - Microsoft
- Grok - xAI

### Which one to chose when
- how do you use an LLM such that you get the best results?

### Prompt engineering
We are now going to start with prompt engineering, When you provide input to an LLM, such as asking a question to ChatGPT, this is called prompting a model - you are sending a prompt to the models, which will trigger the LLM to generate an answer.

Starting Ollama

```
Ollama serve
```

```
ollama run Llama3.1:7b
```




## Hands on
Collect the titles of the articles.
Get a one-sentence description of the articles
Classify the articles: e.g. politics, economics, sports, culture...
Compare the writing style

## Pitfalls, limitations, caveats, privacy


:::::::::::: challenge 



:::::: solution

::::::
::::::::::::

## What is NLP typically good at?

