from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

import pandas as pd


model_b_mini = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


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
        out = cosine_similarity([doc_embeddings_a[i]], doc_embeddings_b[:])

    sims_for_i = out.flatten().tolist()
    idx_a_for_i = [i] * len(sims_for_i)
    idx_b_for_i = list(range(len(sims_for_i)))

    idx_a += idx_a_for_i
    idx_b += idx_b_for_i
    sim += sims_for_i

    col_name = "cos_sim_" + condition
    res = pd.DataFrame({"clustered": idx_a, "lca_act_name": idx_b, col_name: sim})
    return res


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
    bert_mini_df = cos_sim_for_condition(
        bert_mini_embeddings_a, bert_mini_embeddings_b, "bert_mini"
    )
    return bert_mini_df


def selecting_best_activity(input):
    """
     Calculates pairwise cosine similarities of all entries in europages products
    against ecoinvent_activities and returns them as a dataframe.

    Attributes:
        input (pd.DataFrame): Dataframe of matches for which best activity needs to be chosen.

    Returns:
        best_matches (pd.DataFrame): Dataframe holding cosine similarities.

    """

    selected_product = list(set(input.iloc[:, 0].tolist()))
    candidate_list = list(set(input.iloc[:, 1].tolist()))
    main_act = list(set(input.iloc[:, 2].tolist()))

    matches_df = get_cos_sim_bert_mini(list_a=selected_product, list_b=candidate_list)
    matches_df["clustered"] = selected_product * len(candidate_list)
    matches_df["lca_act_name"] = candidate_list
    matches_df["lca_act_type"] = main_act * len(candidate_list)

    best_matches = matches_df.sort_values(
        by=["cos_sim_bert_mini"], ascending=False
    ).head(1)
    best_matches = best_matches[["clustered", "lca_act_name", "lca_act_type"]].head(1)

    return best_matches
