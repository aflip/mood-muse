# MoodMuse

[![License](https://img.shields.io/badge/License-BSD_2--Clause-orange.svg)](https://opensource.org/licenses/BSD-2-Clause)

An app for discovering poetry using embedding-based semantic retrieval


[What is semantic search and why do we want it](https://anandphilip.com/what-is-semantic-search-and-why-do-we-want-it/)
 


## Overview

The app happened because I wanted to understand semantic search.I figured out the basics using the wikiscience dataset, but figured i'd make something that would be fun to use myself and maybe share with friends. So I decided to make something that helps me find better poetry.

## Data


~16000 poems in english scraped from poetryfoundation.org

## Demo


[MoodMuse: Demo app](https://starry-eyed-geometry.anvil.app/)

## Modelling notes

Used the [MTEB leaderboard ](https://huggingface.co/spaces/mteb/leaderboard) and the models listed [in the sentence transformers documentation](https://www.sbert.net/docs/pretrained-models/msmarco-v3.html) and tested about 10-15 different models.

Contrary to intuition, larger language models didn't necessarily have better embeddings. This worked out great because the larger models also take much longer to embed and create much larger embeddings. 

Embedding-as-service platforms like openAI are fast, but those embeddings were not great. The larger models tend to have much vaguer connection to the query than is ideal. Some vagueness is good, too much isn't. And embedding large swaths of text and holding it in a vector db somewhere is much tougher with these services.

The models that are trained for assymmetric retrieval were inferior to the ones trained on symmetric search. This too is counter-intuitive. [`all-mpnet-base-v2`](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) was the best sentence-tranformer model, although [`BAAI/bge-base-en`](https://huggingface.co/BAAI/bge-base-en) and [`thenlper/gte-base`](https://huggingface.co/thenlper/gte-base) were also good.

The main problem with these models is that the `max_seq_length` is generally much smaller than the text that needs to be embedded. This makes for great representation of the first 300-500 or so characters and then no representation of the rest of the text. To solve this, I tried out [chunking the text and max-pooling the results](https://github.com/simonw/llm-sentence-transformers/issues/8#issuecomment-1732618592) which definitely improved the results but I wanted more.

Further search lead to [`jinaai/jina-embeddings-v2-base-en`](https://huggingface.co/jinaai/jina-embeddings-v2-base-en). This embedding model was the best performing. These guys have figured out a way to ingest upto `8192` tokens using [ALiBi](https://arxiv.org/abs/2108.12409). They have a [fine-tuning library that looks very interesting](https://github.com/jina-ai/finetuner) and seem like a good alternative to the openai/anthropics of the world. 

Sentence-Transformers recommends using a reranking model, and I tried them out, and while they do marginally improve the results, the improvements were not enough to justify the extra work. 


## Indexing and retrieval

Following the [guide at pinecone](https://www.pinecone.io/learn/series/faiss/) and [ANN benchmarks](https://ann-benchmarks.com/), I tried out [Neighborhood Graph and Tree (NGT)](https://github.com/yahoojapan/NGT), [FAISS](https://github.com/facebookresearch/faiss) and [HNSW](https://github.com/nmslib/hnswlib) extensively on multiple datasets. I found that on smaller datasets, NGT and FAISS work the best, and on larger datasets the difference between the three is negligible. This could be because I didn't try out large enough datasets. The differences are small and some hyperparameter tuning could improve things. I implemented NGT in the app because I like Japan and I don't like Facebook. 


