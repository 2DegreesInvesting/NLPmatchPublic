import bert_cosine_sim
from sentence_transformers import SentenceTransformer

model_b_mini = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def get_cos_sim_bert_mini(list_a, list_b):
    """
    Returns DataFrame of pairwise cosine similarities for bert_mini approach

    Attributes:
    list_a (list): List of strings.
    list_b (list): List of strings.

    Output:
    bert_mini_df (pd.DataFrame): Dataframe holding cosine similarities and condition.
    
    """
    bert_mini_embeddings_a = model_b_mini.encode(list_a)
    bert_mini_embeddings_b = model_b_mini.encode(list_b)
    bert_mini_df = bert_cosine_sim.cos_sim_for_condition(bert_mini_embeddings_a, bert_mini_embeddings_b, "bert_mini")
    return(bert_mini_df)
