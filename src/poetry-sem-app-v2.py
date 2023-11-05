""" 
Anvil-App-Server App for doing semantic search using sentence embeddings

"""


from transformers import AutoModel
import ngtpy
import anvil.server
import pickle 


anvil.server.connect("YOURSECRETHERE")

# Load the data

with open('../data/poem-corpus.pkl', 'rb') as fIn:
    corpus= pickle.load(fIn)

# Load the NGT Index, created earlier

corpus_ix_name = "../indices/ngt_index_all-poems-AC-AN-jina-v2-base-en"
corpus_index = ngtpy.Index(bytes(j_corpus_ix_name, encoding="utf8"))
 

# Load the model on to the cpu
# Needs Accelerate pip install accelerate
# Im using Jina AI's embedding model which permits a long sequence length
# https://huggingface.co/jinaai/jina-embeddings-v2-base-en

model = AutoModel.from_pretrained(
    "jinaai/jina-embeddings-v2-base-en", trust_remote_code=True, device_map="cpu"
)


@anvil.server.callable
def query_embedding(user_query: str) -> str:

    """
    Retrieves query embeddings and returns relevant poems.
    
    Args:
        user_query (str): The user's query.
    
    Returns:
        str: Poems relevant to the query.
    """
    query = str(user_query).strip()

    # Check if user_query is null.
    if not query:
        return "Not a valid query"


    # Encode the query using the same model that was used to make the corpus embedding

    query_embedding = model.encode(query)

    # embedding search produces a tuple (ix, score) I am extracting just the Ix locations 
    # from this to look up in the corpus.  
    # The size parameter decides how many top k results are returned. 
    # between 5 and 10 generally returns good results for this dataset

    ids = [t[0] for t in corpus_index.search(query_embedding, size=5)]
    
    
    # Saving the user query and results to file

    with open("user_queries.txt", "a", encoding="utf-8",) as f:
        f.write(query + " " + "{}".format(ids) + "\n")

    print("query was: ", query, "\n")
    return print_top_hits(ids)



def print_top_hits(hit_list: list) -> str:
    """
    Prints the top poem hits based on the provided list of IDs.

    Args:
        hit_list (list): List of poem IDs.

    Returns:
        str: Concatenated poems.
    """
    poems=[]

    
    for hit in hit_list[:6]:
        hit_poem= corpus[hit]
        poems.extend(hit_poem)
    return "".join(poems)
