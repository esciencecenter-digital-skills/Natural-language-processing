---
title: "Episode 4: Using large language models"
teaching: 
exercises: 
---

:::::: questions 
- What is a Large Language Model (LLM)?
- How do LLMs differ from traditional NLP models?
- What is the Transformer architecture, and why is it important for LLMs?
- How does prompt engineering influence LLM outputs?
- What are some real-world applications of LLMs?
- How can LLMs generate, classify, and summarize text?

...
::::::

:::::: objectives
After following this lesson, learners will be able to:
- Understand what Large Language Models (LLMs) are and their role in NLP
- Explain the Transformer architecture and why it is foundational for LLMs
- Use prompt engineering to generate high-quality text responses
- Apply LLMs to real-world tasks like news article analysis
- Explore pre-trained LLMs and experiment with custom tasks
- Set up a chatbot and a simple RAG
- Understand the impact of LLMs in modern AI and language processing

::::::

# WHAT IS THE DIFFERENCE BETWEEN BERT AND GPT? WHY CAN BERT NOT GENERATE?

## This episode
Large Language Models (LLMs) are a hot and a big topic these days, and are continuously in the news. Everybody heard of ChatGPT, many tried it out for a variety of purposes, or even incorporated these tools in their daily work. But what are these models exactly, how do they work 'under the hood', and how can you make use of them in the best possible way?

In this episode, we will:
- Explore LLMs, which represent a significant advancement in natural language processing (NLP). We will begin by defining what LLMs are and touching on their foundational architecture, particularly the Transformer model, which allows the LLM to understand and generate human-like text. 
- Through practical examples, you will discover how to work with thse models for specific tasks with content generation.
- We'll also discuss real-world applications of LLMs, such as analyzing news articles and generating insights from large amounts of text data. 

This episode aims to equip you with both theoretical knowledge and practical skills, preparing you to harness the power of LLMs in your own projects and applications.

## What are Large Language Models?
Large language models (LLMs) represent a significant advancement in artificial intelligence, designed to process and interpret large-scale natural language data to generate responses to user queries. By being trained on extensive datasets through advanced machine learning algorithms, these models learn the intricate patterns, structures and nuances of human language. This enables them to produce coherent and natural-sounding language outputs across various inputs. As a result, large language models are becoming increasingly essential in a range of tasks such as text generation, text summarisation, rewriting, question answering, and language translation.

The emergence of *ChatGPT*, powered by OpenAI's advanced LLMs, has brought these capabilities into the mainstream. With ChatGPT, users interact through natural language, enabling seamless conversations and performing complex tasks across various sectors, like customer service, education, and content creation. Models like GPT-4, BERT, and LLaMA are also used across various applications; from chatbots, virtual assistants, text analysis, translation, summarization, and more.

Notably, the success of ChatGPT and other LLM-driven applications highlights their versatility and potential to transform how we interact with digital systems. These models continue to push the boundaries of what's possible in human-computer interaction, offering a glimpse into a future where machines and humans communicate more naturally and effectively.

### Timeline
The journey of large language models over the past decade highlights the rapid evolution and growing impact of AI in language processing:

- **2012**: Significant advancements in deep learning architectures and access to larger datasets lead to the creation of the GPT (Generative Pre-trained Transformer) framework, marking a pivotal point in AI development.

- **2018**: Google unveils BERT (Bidirectional Encoder Representations from Transformers), introducing a groundbreaking architecture that sets the stage for the next generation of large language models.

- **2020**: OpenAI releases GPT-3, which, with its remarkable 175 billion parameters, establishes a new standard for performance in language-related tasks.

- **2022**: The launch of ChatGPT marks a turning point by making GPT-3 and similar models easily accessible to the public via a web interface, significantly raising public awareness and interest in LLMs and generative AI.

- **2023**: Open-source LLMs begin to deliver impressive outcomes with the introduction of models like Dolly 2.0, LLaMA, Alpaca, and Vicuna. Meanwhile, the release of GPT-4 pushes the envelope by setting new records for both parameter scale and overall performance.

### Starting up a simple chat 
Let's explore how these LLMs work by doing a simple chat demonstration.

Imagine asking a computer a question and getting a human-like response. An LLM, like GPT-4, can generate responses that feel natural, informative, and tailored to your question by analyzing the context and meaning of your input. In the demonstration below, you'll see how an LLM responds to a prompt and how it builds upon previous information to maintain the conversation.

### Example Chat

- *User*: "What's the tallest mountain in the world?"

- *LLM*: "The tallest mountain in the world is Mount Everest, which stands at 8,848 meters (29,029 feet)."

- *User*: "How long does it take to climb it?"

- *LLM*: "Climbing Mount Everest typically takes around two months, due to the acclimatization process needed to adjust to high altitudes. Expeditions often start from Base Camp in Nepal, where climbers spend several weeks preparing."

- *User*: "Could someone with no climbing experience try it?"

- *LLM*: "Climbing Mount Everest is extremely challenging and requires significant training and experience due to the harsh conditions, high altitudes, and technical aspects of the climb. Most climbers spend years training and gaining experience on other mountains before attempting Everest."

This example shows how LLMs can maintain context across multiple questions. Notice how the model connects information across each question, keeping the conversation relevant and informed.

### Challenge 1: Try chatting yourself with an LLM
If you’d like to try this yourself, here’s a small exercise:

- Open an LLM interface (such as ChatGPT or a similar service).
- Think of a question on a topic you’re interested in, like a famous historical event, a scientific question, or even advice on writing.
- Notice how the model responds to your follow-up questions, using previous context to build its responses.

This quick chat example introduces you to the conversational abilities of LLMs, demonstrating their ability to respond contextually, provide coherent answers, and follow up on earlier parts of the conversation.

## How are LLMs different from other NLP techniques?
LLMs stand apart from traditional NLP techniques due to their scale, versatility, and reliance on deep learning models, especially Transformers. Here’s a breakdown of how they differ:

### Scale of training data and parameters
- LLMs: Trained on massive datasets (trillions of words) and use billions of parameters, allowing them to learn language patterns, facts, and relationships at an unprecedented depth.
- Traditional NLP: Techniques like rule-based systems or smaller machine learning models rely on much smaller datasets, often requiring domain-specific training for each task (e.g., sentiment analysis or named entity recognition).

### Model architecture
- LLMs: Use the Transformer architecture, particularly self-attention, to analyze relationships between words regardless of position. This allows them to capture long-range dependencies and context better than traditional models.
- Traditional NLP: Often use simpler models like bag-of-words, TF-IDF (term frequency-inverse document frequency), or shallow neural networks, which treat words independently or consider only local context, missing the complex, global relationships.

### Generalization vs. task-specific models
- LLMs: Can be applied across a wide range of tasks—summarization, translation, question answering, etc.—without the need for separate models for each. Fine-tuning makes them even more adaptable to specific needs.
- Traditional NLP: Typically requires developing or training a separate model for each task. For example, separate models for sentiment analysis, translation, and entity recognition.

### Learning from unlabeled data
- LLMs: Leverage unsupervised or self-supervised learning during pretraining, enabling them to learn language patterns from raw text without human-labeled data.
- Traditional NLP: Often relies on labeled data for training (e.g., labeled sentiment or part-of-speech tags), which can be costly and time-consuming to create at scale.

### Context and language nuance
- LLMs: Excel at understanding context, tone, and nuance, due to their ability to weigh word relationships dynamically. This enables better handling of idioms, sarcasm, and ambiguous phrases.
- Traditional NLP: Struggles with complex language nuances, often producing more rigid or literal interpretations. Contextual understanding is limited, especially for longer sentences or paragraphs.

### Adaptability and fine-tuning
- LLMs: Easily adaptable to new tasks or domains with fine-tuning, making them versatile across different applications.
- Traditional NLP: Less flexible, often requiring retraining from scratch or heavy feature engineering to adapt to new domains or tasks.

In short, LLMs represent a leap forward by combining scale, flexibility, and deep learning power, allowing for more accurate, nuanced, and versatile language processing compared to traditional NLP techniques.

## What LLMs are good at
- *Language generation*: Creating coherent and contextually appropriate text, making them ideal for creative writing, chatbots, and automated responses.
- *Summarization and translation*: Quickly summarizing articles, books, and translating text between languages with reasonable accuracy.
- *Information retrieval and answering questions*: LLMs can recall and apply general knowledge from their training data to answer questions, though they don’t actually “know” facts.
- *Sentiment and text classification*: LLMs can classify text for tasks like sentiment analysis, spam detection, and topic categorization.

## What LLMs struggle with
- *Fact-based accuracy*: Since LLMs don’t “know” facts, they may generate incorrect or outdated information and are prone to hallucinations (making up facts).
- *Understanding context over long passages*: LLMs can struggle with context over very long texts and may lose track of earlier details, affecting coherence.
- *Mathematical reasoning and logic*: Though improving, LLMs often find complex problem-solving and detailed logical reasoning challenging without direct guidance.
- *Ethical and sensitive issues*: LLMs may produce biased or offensive text based on biases present in the training data, making content moderation necessary in sensitive applications.

## How do LLMs work?
So, how is it that you can chat with a model and receive responses that seem almost human? The answer lies in the architecture and training of Large Language Models (LLMs), which are powered by advanced neural networks that understand, generate, and even translate human language with surprising accuracy.

At the core of LLMs lies a framework known as the **transformer**; a concept already encountered in the previous episode. Transformers allow these models to process vast amounts of text and learn the structure and nuances of language. This setup enables LLMs not only to answer questions but also to predict, complete, and even generate coherent text based on the patterns they've learned.

LLMs are trained on large text datasets and later fine-tuned on specific tasks, which helps them adapt to a wide range of applications, from conversation to text classification. The result? A model that can chat, summarize, translate, and much more—all by leveraging these core mechanisms. LLM's rely on the following key concepts:

- **Transformers and self-attention**: The transformer architecture, especially the self-attention mechanism, is at the heart of LLMs. Self-attention enables these models to understand the importance of each word in relation to others in a sequence, regardless of their position.
- **Pretraining and fine-tuning**: LLMs are first pre-trained on large text datasets using tasks like predicting the next word in a sentence, learning language patterns. They are then fine-tuned on specific tasks (e.g., translation, summarization) to enhance performance for targeted applications.
- **Generative vs. discriminative models**: LMs can be applied to both generative tasks (e.g., text generation) and discriminative tasks (e.g., classification).

Below is a simplified diagram of the transformer architecture, which highlights the role of self-attention in understanding relationships between words:

[#TODO: ADD A NICE FIGURE. CAN WE USE AN EXISITNG IMAGE FROM SOMEWHERE]

Let’s see how this actually works with a hands-on example. We’ll use Hugging Face’s library to load a pretrained model and visualize the self-attention scores for a sentence.

### How self-attention enables language models to generate responses

In practice, this attention mechanism helps LLMs produce coherent responses by establishing relationships between words as each new token is generated. Here’s how it works: 

- *Understanding word relationships*. Self-attention enables the model to weigh the importance of each word in a sentence, no matter where it appears, to make sense of the sentence as a whole.

- *Predicting next words based on context*. With these relationships mapped out, the model can predict the next word in a sequence. For example, in “The fox,” self-attention allows the model to anticipate that “jumps” or “runs” might come next rather than something unrelated like “table.”

- *Structuring responses*. As each word is generated, the model assesses how each new token impacts the entire sentence, ensuring that responses are relevant, logically sound, and grammatically correct. This ability to “structure” language is why LLMs can produce responses that are contextually meaningful and well-organized.
  
By leveraging self-attention to build meaningful relationships across tokens, transformers power LLMs to generate responses that feel relevant, accurate, and human-like.

## A zoo of Large Language Models
The era of Large Language Models started with the release of the model called BERT, created by Google, that was discussed in the previous episode. The techniques used in that model started a movement of the creating of many new models. There are a number of big companies that keep improving on their models and releasing new ones rapidly. One of the most well-known is GPT, the model developed by OpenAI and that is used for ChatGPT. The first version of GPT was released in 2018. Since then various versions have been released, each improved in performance by using more parameters and a larger training data set.  While the first GPT model was open source, the recent versions are not. This means that the model architecture, number of parameters, and used training data is undisclosed. There are however many more competing models, some of which are more transparent of even fully open source. Llama is currently one of the best-performing open-source models

Other models:
- ChatGPT - OpenAI
- Llama  - Meta 
- Mistral / Mixtral - Mistral AI
- Gemini - Google
- Claude - Anthropic - (fouded by former OpenAI employees)
- Grok - xAI

Training a large language model is extremely resource intensive; while you can train for example a simple classifier to identify if a sentence is positive or negative yourselve, this is not possible for training a LLM. For example, llama published Llama 3.1 405B; this is model that has 405 billion paramters, and that was trained on 15 trillion tokens, taking 31 million GPU hours (H100 gpus), and emmittion almost 9000 tons of CO_2 (for the training process only).

## Which one to chose when
With so many available models the question arises which model you should use when? One thing to consider here is whether you want to use an open source model or not. But another important aspect is that it depends on the task at hand. There are various leaderboards that track which tasks specific models are good at, based on widely used benchmarks. Also, which language are you using? Most models are fully trained on English, not many models are trained on Dutch text. So if you are using Dutch texts, you may want to look for a model that is trained on Dutch. Additionally, some LLMs are multimodal models,meaning they can process various forms of input; text, images, timeseries, audio, videos and so on.

### Building a chatbot
 It is time to start working with LLM models. We are not going to train our own LLM, but use Meta's open source Llama model to set up a chatbot. The chat model can be used for prompt engineering - When you provide input to an LLM, such as asking a question to ChatGPT, this is called prompting a model - you are sending a prompt to the models, which will trigger the LLM to generate an answer. Prompt engineering 

Starting Ollama
Ollama will be used to run a chosen LLM. Ollama is a platform that allows users to run various LLM locally on your own computer. This is different from for example using chatgpt, where you log in and use the online api. ChatGPT collects the input you are providing and uses this to their own benefit. Running an LLM locally using Ollama thus preserves your privacy. It also allows you to customize a model, by setting certain parameters, or even by finetuning a model.

To start ollama:
```
ollama serve
```

Next, download the large language model to be used. In this case use the smallest open source llama model, which is llama3.1:8b. Here 3.1 is the version of the model and 8b stands for the number of paramters that the model has. 
```
!ollama pull llama3.1:8b
```
In general, a bigger version of the same model (such as Llama3.1:70b) is better in accuracy, but since it is larger it takes more resources to run and can hence be too much for a laptop.

Import the packages that will be used:
```python
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
```

Create a model instance. Here, `model` defines the LLM to be used, which is set to the model just downloaded, and `temperature` sets the randomness of the mode, using the value zero ensures that repeating a question will give the same model output (answer).

```
llm = ChatOllama(model=llama3.1:8b, temperature=0)
```

Now that the model is set up, it can be prompted - ask it a question.

```python
question = "When was the moon landing?"
chatresult = llm.invoke([HumanMessage(content=question)])
print(chatresult.content)
```

:::::::::::: challenge 

Play around with the chat bot by changing the questions.
- How is the quality of the answers? 
- Is it able to answer general questions, and very specific questions?
- Which limitations can you identify?
- How could you get better answers?

:::::: solution

::::::

::::::::::::


### Use context
To improve on what to expect the LLM to return, it is also possible to provide it with some context. For example, add:
```python
context = "You are a highschool history teacher trying to explain societal impact of historic events."
messages = [
    SystemMessage(content=context),
    HumanMessage(content=question),
]
```

```python
chatresult = llm.invoke(messages)
print(chatresult.content)
```

The benefit here is that your answer will be phrased in a way that fits your context, without having to specify this for every question.

### Use chat history
With this chatbot the LLM can be prompted to generate output based on the provided input and context. However, what is not possible in this state, is to ask followup questions. This can be useful to refine the output that it generates. So lets implement this as well.

```python
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
```
First define a storage distionary and a configurable, so that the history of a certain conversation can be saved based on a session_id.
```python
store = {}
config = {"configurable": {"session_id": "nlp_workshop"}}
```

Then define a function that takes in this session_id and saves it in the storage.

```python
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]
```

```python
with_message_history = RunnableWithMessageHistory(llm, get_session_history)
```

```python
# Followup question
response_history = with_message_history.invoke(
    messages,
    config=config,
)

print(response_history.content)
```

```python
# Followup
response = with_message_history.invoke(
    [HumanMessage(content="Shorten the answer to 20 words")],
    config=config,
)

print(response.content)
```

```python
# Followup instruction
response = with_message_history.invoke(
    [HumanMessage(content="Translate the answer to Dutch")],
    config=config,
)

print(response.content)
```

### Retrieval Augmented Generation - Build a RAG
A chatbot tends to give quite generalised answers. A more specific chatbot can be made by building a Retrieval Argumented Generation agent. This is an information that you youself provide with a knowledge base: a large number of documents. When prompted with a questions, the agent first retrieves relevant sections of the data that is in the knowledge base, and then generates and answer based on that data. In this way you can build an agent with very specific knowledge.

The simplest form of a rag consists of two parts, a retriever and a generator. The retriever part will collect data from the provided data, so first a knowledge base has to be created for the retriever.

To generate text in the RAG the trained Llama model will be used, which works well for English text. Because this model was not trained on Dutch text, the RAG will work better for an English knowledge base.

Three newpaper pages will be used for the example RAG, these are pages from a Curacoa newspaper. This is a Dutch newspaper with an additional page in English. The text versions of the newspapers can be downloaded to only get these specific English pages. Save them in a folder called "rag_data" for further processing:
- [page1](https://www.delpher.nl/nl/kranten/view?query=the+moon&coll=ddd&identifier=ddd:010460545:mpeg21:p012&resultsidentifier=ddd:010460545:mpeg21:a0134&rowid=4)
- [page2](https://www.delpher.nl/nl/kranten/view?query=moon+landing&coll=ddd&page=1&facets%5Bspatial%5D%5B%5D=Nederlandse+Antillen&identifier=ddd:010460616:mpeg21:a0146&resultsidentifier=ddd:010460616:mpeg21:a0146&rowid=1)
- [page3](https://www.delpher.nl/nl/kranten/view?query=moon+landing&coll=ddd&page=1&facets%5Bspatial%5D%5B%5D=Nederlandse+Antillen&identifier=ddd:010460520:mpeg21:a0167&resultsidentifier=ddd:010460520:mpeg21:a0167&rowid=7)

#### The knowledge base - a vector store
Language models all work with vectors - embedded text. Instead of saving text, a the data has to be stored in embedded versions in a vector store, where the retriever can shop around for the relevant text.

There a number of packages to be used in this section to build the RAG.
```python
import os
from IPython.display import Image, display
from typing_extensions import List, TypedDict

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from langchain_nomic.embeddings import NomicEmbeddings
```

Define the large language model to be used to generate an answer based on provided context:
```python
llm = ChatOllama(model="llama3.1:8b", temperature=0)
```

Define the embeddings model, this is the model to convert our knowledge base texts into vector embeddings and will be used for the retrieval part of the RAG:
```python
embeddings=NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local")
```

::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: instructor
For more on Nomic embeddings see: https://python.langchain.com/api_reference/nomic/embeddings/langchain_nomic.embeddings.NomicEmbeddings.html

using inference_model="local" uses (Embed4All)[https://docs.gpt4all.io/old/gpt4all_python_embedding.html]


::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

In the text files, the articles are split by '---'. This information can be used to store the individual articles into a list. Store the filename of the articles in a list as well, so that one can find easily in from which file a text snippet was taken.

```python
dir = "./rag_data"
articles = []
metadata = []

# Iterate over files and add individual articles and corresponding filenames to lists
for file in os.listdir(dir):
    file_path = os.path.join(dir, file)
    with open(file_path, "r") as f:
        content = f.read().split('---')
        articles.extend(content)
        metadata.extend([file_path] * len(content))
```

The generator will in the end provide an answer based on the text snippet that is retrieved from the knowledge base. If the fragment is very long, it may contain a lot of irrelevant information, which will blur the generated answer. Therefor it is better to split the data into smaller parts, so that the retriever can collect very specific pieces of text to generate an answer from. It is usefull to keep some overlap between the splits, so that information does not get lost because of for example splits in the middle of a sentence.

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=50
)

documents = text_splitter.create_documents(articles, metadatas=[{'filename': file} for file in files])

print(documents)
```

This text splitter splits text based on the defined character chunk size, but also takes into account spaces and newlines to slit in "smart" chunks, so the chunks will not be exactely of length 500.

Finally, convert each text split into a vector, and save all vectors in a vector store. The text is converted into embeddings using the earlier defined embeddings model.

```python
vectorstore = InMemoryVectorStore.from_texts(
    [doc.page_content for doc in documents],
    embedding=embeddings,
)
```

#### Setting up the retriever and generator
Define the structure of a dictionary with the keys `question`, `context`, and `answer`.

```python
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str
```

Define the retriever function of the RAG. It taked in the question and does a similarity search in the created vectorstore and returns the text snippets that were found to be similar. The similarity search converts the question into an embeddings vector and uses the cosine similarity to determine the similarity between the question and snippets. It then returns the top 4 snippets with the highest cosine similarity score. The snippets are returned in the original text form, i.e. the retrieved vectors are transformed back into text.

```python
def retrieve(state: State):
    "Retrieve documents that are similar to the question."
    retrieved_docs = vectorstore.similarity_search(state["question"], k=4)
    return {"context": retrieved_docs}
```

Define the generator function of the RAG. In this function a prompt is defined for the RAG using the context and question. The large language model (the Llama model, defined above) is then invoked with this question and generates an answer for the provided prompt, which is returned as the answer key of the dictionary.

```python
def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    
    rag_prompt = """You are an assistant for question-answering tasks.
    Here is the context to use to answer the question:
    {context}
    Think carefully about the above context.
    Now, review the user question:
    {question}
    Provide an answer to this questions using only the above context.
    Use 10 sentences maximum and keep the answer concise.
    Answer:"""

    rag_prompt_formatted = rag_prompt.format(context=docs_content, question=State["question"])
    
    generate = llm.invoke([HumanMessage(content=rag_prompt_formatted)])
    return {"answer": generate.content}
```

#### Build the workflow
The retriever and generator are combined into a workflow graph. The workflow is defined as a StateGraph that uses the dictionary structure (with the keys `question`, `context`, and `answer`) defined above. The retriever and generator are added as nodes, and the two are connected via the edge. The retrieve is set as the start point of the workflow, and finally the graph is compiled into an executable.
```
workflow = StateGraph(State)
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)
workflow.add_edge("retrieve", "generate")
workflow.set_entry_point("retrieve")

graph = workflow.compile()
```
```python
display(Image(graph.get_graph().draw_mermaid_png()))
```

![workflow](../workflow.png)

That's it! The RAG can now be asked questions. Let's see what it can tell about the moon landing:

```python
response = graph.invoke({"question": "Who landed on the Moon?"})
print(response["answer"])
```

This is quite a specific answer. It can be seen why by looking at the text snippets that were used:
```
print(response["context"])
```

While a general chatbot uses all the information in the material that it was trained on, the RAG only uses the information that was stored in the vectorstore to generate the answer.

:::::::::::: challenge 
Try generating more answers with the RAG based on other questions, perhaps also looking at the newspaper texts that are used. What stands out?

:::::: solution
For example:
- The RAG returns in some cases that no answer can be generated on the context it was provided with
- For some questions, the LLM returns that it cannot provide an answer because of safety precausions that are inherent to the LLM used, such as information about violent acts.
::::::

::::::::::::

This is the simpelest form of a RAG, with a retriever and a generator. However, one can make the RAG more complex by adding more components to the workflow.



## Pitfalls, limitations, caveats, privacy




:::::::::::: challenge 



:::::: solution

::::::

::::::::::::

## Key points to remember

- *Transformer models power LLMs*: The Transformer architecture and its self-attention mechanism allow LLMs to handle long-range dependencies in text effectively.
- *LLMs excel at multiple tasks*: From text generation to classification and summarization, LLMs like Llama are versatile and perform well on various NLP tasks.
- *Prompt engineering is crucial*: Designing effective prompts can significantly improve the quality of outputs from LLMs, especially for creative tasks like text generation and translation.
- *Real-world use cases*: LLMs can be applied to real-world problems like news classification, summarization, and headline generation, improving efficiency in content management and delivery.


