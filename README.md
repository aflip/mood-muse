# MoodMuse

[![License](https://img.shields.io/badge/License-BSD_2--Clause-orange.svg)](https://opensource.org/licenses/BSD-2-Clause)

An app for discovering poetry using embedding-based semantic retrieval


[What is semantic search and why do we want it](https://anandphilip.com/what-is-semantic-search-and-why-do-we-want-it/)
 
## Demo


[MoodMuse: Demo app](https://starry-eyed-geometry.anvil.app/)

## Features

- Open-ended discovery of poetry based on emotions, themes, objects or settings.
- Efficient Approximate Nearest Neighbors (ANN) search using NGT

## Overview

The app happened because I wanted to understand semantic search. I figured out the basics using the [`millawell/wikipedia_field_of_science`](https://huggingface.co/datasets/millawell/wikipedia_field_of_science) dataset, but wanted to make something that would be fun to use myself and maybe share with friends. So I decided to make something that helps me find better poetry.

## Data

~16000 english poems scraped from poetryfoundation.org

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

## Tech stack/Process


1. Embed corpus on `jina-embeddings-v2-base-en`
2. Index embedding using NGT
3. Embed query using the same model
4. Search NGT index using query embedding, retrieving based on cosine similarity
5. Look up top results in a pandas dataframe that has the text of the poems (don't judge me, it's just 50MB and a db is too much work)
6. Serve the top 5 hits using an [Anvil app](https://anvil.works/)


## Resources

The app takes great inspiration from the excellent Vicki Boykis, who, around the same time as when I began puttering around with semantic search, was doing the same and shared her findings in great detail. Her app for [finding books by vibes - Viberary](https://viberary.pizza/) is excellent and her[ research on this subject](https://github.com/veekaybee/viberary) was a major source of information. 

Pinecone has a great online [book on NLP for semantic search](https://www.pinecone.io/learn/series/nlp/) 

[Sentence-transformers docuemntation](https://www.sbert.net/) and [github repo](https://github.com/UKPLab/sentence-transformers/tree/master/examples) are filled with great instructions and examples on how to train, embed, retreieve etc. This site was open all the time for the last few months. 

## Interesting papers

Mengzhao Wang, Xiaoliang Xu, Qiang Yue, and Yuxiang Wang. 2021. [A comprehensive survey and experimental comparison of graph-based approximate nearest neighbor search](https://arxiv.org/abs/2101.12631). Proc. VLDB Endow. 14, 11 (July 2021), 1964–1978. https://doi.org/10.14778/3476249.3476255 

[Pretrained Transformers for Text Ranking: BERT and Beyond](https://aclanthology.org/2021.naacl-tutorials.1) (Yates et al., NAACL 2021)


