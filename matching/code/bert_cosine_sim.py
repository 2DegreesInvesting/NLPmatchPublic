import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def cos_sim_for_condition(doc_embeddings_a, doc_embeddings_b, condition):
    """
    Returns DataFrame of pairwise cosine similarities
    
    Calculates pairwise cosine similarities of all entries in doc_embeddings_a
    against doc_embeddings_b and returns them as a dataframe. The dataframe also
    holds the name of the condition.

    Attributes:
        doc_embeddings_a (list): List of embeddings.
        doc_embeddings_b (list): List of embeddings.
        condition(str): Name of condition.

    Output:
        res (pd.DataFrame): Dataframe holding cosine similarities and condition.

    """

    idx_a = []
    idx_b = []
    sim = []
   
    for i in range(len(doc_embeddings_a)):
        out = cosine_similarity(
            [doc_embeddings_a[i]],
            doc_embeddings_b[:]
        )
     
    sims_for_i = out.flatten().tolist()
    idx_a_for_i = [i] * len(sims_for_i)
    idx_b_for_i = list(range(len(sims_for_i)))
 
    idx_a += idx_a_for_i
    idx_b += idx_b_for_i
    sim += sims_for_i
 
    col_name = "cos_sim_" + condition     
    res = pd.DataFrame({"clustered": idx_a, "lca_act_name": idx_b, col_name :sim})
    return res
