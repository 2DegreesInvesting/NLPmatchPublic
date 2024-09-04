import pandas as pd
import os
from haystack.nodes import DensePassageRetriever
from haystack.document_stores import InMemoryDocumentStore
import datetime

def get_data():
    """
        This function reads in the manually preprocessed products from europages
        and ecoinvent_complete dataset which has not been filtered on geography.
    Attributes:
        None
    Output:
        euro (pd.DataFrame): Dataframe containing all preprocessed europages products.
        eco (pd.DataFrame): eDataframe containing all ecoinvent products.
    """

    # euro = pd.read_csv('matching/data/input/20240603_EP_compare.csv', encoding = 'latin-1')
    euro = pd.read_csv('_00_production/data/input/product_categorisation.csv', sep = ';', encoding = 'latin-1')
    eco = pd.read_csv('matching/data/input/ecoinvent_v310_incomplete_filtered.csv').rename(columns = {'Reference Product Name': 'ecoinvent_prod'})[['Activity UUID_Product UUID', 'ecoinvent_prod']].drop_duplicates().reset_index(drop=True)
    return euro, eco


def get_retriever(save_dir, document_store, eco):
    """
        This function load the pre-trained Dense Passage Retriever model from the directory,
        writes the ecoinvent products to the document store and turns them into embeddings.

    Attributes:
        save_dir (str): The directory where the trained DPR model is stored
        document_store (Haystack database): Database where the ecoinvent products are stored.
        eco (pd.DataFrame): A dataframe containing all ecoinvent products

    Output:
        retriever (retriever): Retrieves the Ecoinvent product embeddings by sweeping through the document store
    """

    retriever = DensePassageRetriever.load(load_dir=save_dir, document_store = document_store)
    # documents = [{"content": row["ecoinvent_prod"], "meta": {"id": str(row["Product UUID"])}} for _, row in eco.iterrows()]
    documents = [{"content": row["ecoinvent_prod"], "meta": {"id": str(row["Activity UUID_Product UUID"])}} for _, row in eco.iterrows()]
    document_store.write_documents(documents)
    retriever.embed_documents(document_store)
    document_store.update_embeddings(retriever)

    return retriever


def get_similarities(euro, retriever):
    """
        This function loads the ecoinvent embeddings from the DPR model, and calculates a similarity score
        for each of the europages products, where similarity scores are rounded off to 10 decimals
        to assure uniqueness.

    Attributes:
        euro (pd.DataFrame): Dataframe containing all preprocessed europages products.
        retriever (retriever): Retrieves the Ecoinvent product embeddings by sweeping through the document store

    Output:
        top_5_model_model_sims (pd.DataFrame): Dataframe containing the top 5 Ecoinvent products
        for each europages product as calculated by the DPR model

    """

    top_5_model_sims = pd.DataFrame(columns=["clustered", "ecoinvent_prod", "model_similarity"])

    for query in euro['clustered']:
        print("Getting the top 5 candidates for " + str(query))
        ecoinvent_prods = []
        sim_scores = []
        clustered = [query] * 10
        retrieved_results = retriever.retrieve(query, top_k = 10)

        for i in range(len(retrieved_results)):
            document = retrieved_results[i]                                                                                                                                
            content = document.content
            ecoinvent_prods.append(content)
            score = round(document.score, 10)
            sim_scores.append(score)

        result_df = pd.DataFrame({"clustered": clustered, "ecoinvent_prod": ecoinvent_prods, "model_similarity": sim_scores})
        top_5_model_sims = pd.concat([top_5_model_sims,result_df])

    return top_5_model_sims


def main():
    """
        The main function loads the document store, sets the path requirements, and loads all functions to calculate
        the top 5 similarities, which are written to a csv file.

    Attributes:
        None

    Output:
        top_5_model_model_sims (csv): Dataframe containing the top 5 Ecoinvent products
        for each europages product as calculated by the DPR model.

    """

    print("Executing DPR similarity")
    save_dir = "matching/model/dpr"
    document_store = InMemoryDocumentStore(embedding_dim=768)
    current_date = datetime.datetime.now().strftime("%Y%m%d")
    output_folder = 'matching/data/intermediate'
    file_name = f"{current_date}_similarities.csv"
    output_file_path = os.path.join(output_folder, file_name)

    euro, eco = get_data()
    print("Got the data")
    retriever = get_retriever(save_dir, document_store, eco)
    print("Got the retriever")
    top_5_model_sims = get_similarities(euro, retriever)

    top_5_model_sims.to_csv(output_file_path, index=False)

if __name__ == "__main__":
    main()

