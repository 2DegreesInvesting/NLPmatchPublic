import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
import json
from haystack.nodes import DensePassageRetriever
from haystack.document_stores import InMemoryDocumentStore

import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s: %(message)s"
)
logging.getLogger("haystack").setLevel(logging.INFO)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


doc_dir = ""
train_filename = "matching/data/input/train_data.json"
dev_filename = "matching/data/input/val_data.json"
test_filename = "matching/data/input/test_data.json"
save_dir = "matching/model/dpr"

query_model = "facebook/dpr-question_encoder-single-nq-base"
passage_model = "facebook/dpr-ctx_encoder-single-nq-base"

def group_answers(data):

    """
    This function creates, for each europages product, a list of ecoinvent products for which
    it has a positive or negative match.

    Attributes:
        data (pd.DataFrame): Dataframe containing either positive or negative matches

    Output:
        new_data_train (pd.DataFrame): Input dataframe, including list of ecoinvent products.

    """

    # Grouping by product and combining input products as separate strings
    grouped_answers = data.groupby('clustered')['ecoinvent_prod'].apply(list).reset_index()
    grouped_answers = grouped_answers.rename(columns={"ecoinvent_prod": "answers"})
    new_data_train = pd.merge(data, grouped_answers, on="clustered")

    return new_data_train


def transform_group(data):

    """
        This function creates positive, negative, and hard negative columns which are required as input 
        format in training the Dense Passage Retrieval model.

    Attributes:
        data (pd.Dataframe): Training data for the DPR model.

    Output:
        grouped_train_data (pd.DataFrame): TRaining data for DPR model, including sentiment columns.

    """

    data['positive'] = np.where(data['manual_certainty'] == 'yes', data['ecoinvent_prod'], np.nan)
    data['negative'] = np.where((data['manual_certainty'] == 'no') & data['hard no'].isna(), data['ecoinvent_prod'], np.nan)
    data['hard_negative'] = np.where(data['hard no'] == True, data['ecoinvent_prod'], np.nan)
    grouped_train_data = data.copy()
    # grouped_train_data = data.groupby('clustered')['positive'].apply(list).reset_index()

    return grouped_train_data


def define_pos_neg(data):

    """
        This function creates lists of positive, negative, and hard_negative ecoinvent matches for
        each of the europages products.

    Attributes:
        data (pd.DataFrame): Dataframe containing matches between europages and ecoinvent

    Output:
        grouped_data_new (pd.DataFrame): Dataframe containing lists of ecoinvent products per sentiment
        per europages product.

    """

    grouped_data = data.groupby(['clustered'])
    grouped_entries = []
    for group, group_df in grouped_data:
        # Get the entries from the three columns for the current group
        entries1 = group_df['positive'].dropna().tolist()
        entries2 = group_df['negative'].dropna().tolist()
        entries3 = group_df['hard_negative'].dropna().tolist()
        # entries4 = group_df['answers'].iloc[0]
        entries4 = entries1

        if not entries1 or all(e == '' or e == ',' for e in entries1):
            entries1 = np.nan
        if not entries2 or all(e == '' or e == ',' for e in entries2):
            entries2 = np.nan
        if not entries3 or all(e == '' or e == ',' for e in entries3):
            entries3 = np.nan

        # Append the grouped entries to the list
        grouped_entries.append((group, entries4, entries1, entries2, entries3))
        grouped_data_new = pd.DataFrame(grouped_entries).replace(np.nan, '', regex=True)
        grouped_data_new.columns = ['question', 'answers', 'positive', 'negative', 'hard_negative']

    return grouped_data_new


def create_dictionary(data):

    """
        This function creates a list of dictionaries for the training data of the DPR model,
        including formats for each dictionary required by the haystack DPR package

    Attributes:
        data (pd.DataFrame): Dataframe containing training data for the DPR model.

    Output:
        data_dict_list (list of dicts): List of dictionaries with training data for the DPR model.
    
    """

    # Create dictionary for one each data point (row)
    data_dict_list = []

    for item, row in data.iterrows():
        pos_list = []
        neg_list = []
        hard_neg_list = []

        for pos_element in row['positive']:
            if pd.isna(pos_element):
                continue
            pos_ctxs = {'title': '', 'text': pos_element, 'score': 0, 'title_score': 0, 'passage_id': ''}
            pos_list.append(pos_ctxs)
        for neg_element in row['negative']:
            if pd.isna(neg_element):
                continue
            neg_ctxs = {'title': '', 'text': neg_element, 'score': 0, 'title_score': 0, 'passage_id': ''}
            neg_list.append(neg_ctxs)
        for hard_neg_element in row['hard_negative']:
            if pd.isna(hard_neg_element):
                continue
            hard_neg_ctxs = {'title': '', 'text': hard_neg_element, 'score': 0, 'title_score': 0, 'passage_id': ''}
            hard_neg_list.append(hard_neg_ctxs)

        data_dict = {
            'dataset': "data_train",
            'question': row['question'][0],
            'answers': row['answers'],
            'positive_ctxs': pos_list,
            'negative_ctxs': neg_list,
            'hard_negative_ctxs': hard_neg_list
        }
        data_dict_list.append(data_dict)

    return data_dict_list


def data_split(data_dict_train, test_data):

    """
        This function splits the training data dictionary into a train and validation set, 
        and writes the json files to the specified path.

    Attributes:
        data_dict_train (list of dicts): list of dictionaries with training data
        test_data (list_of_dicts): list of dictionaries with test data

    Output:
        train_data (list of dicts): list of dictionaries with training data
        val_data (list_of_dicts): list of dictionaries with validation data
        test_data (list of dicts) list of dictionaries with test data    
    
    """

    train_data, val_data = train_test_split(data_dict_train, test_size=0.4, random_state=None)
    file_path_model = 'matching/data/input/model_data.json'
    file_path_train = 'matching/data/input/train_data.json'
    file_path_val = 'matching/data/input/val_data.json'
    file_path_test = 'matching/data/input/test_data.json'

    with open(file_path_model, 'w') as json_file:
        json.dump(data_dict_train, json_file, indent=4)

    with open(file_path_train, 'w') as json_file:
        json.dump(train_data, json_file, indent=4)

    with open(file_path_val, 'w') as json_file:
        json.dump(val_data, json_file, indent=4)

    with open(file_path_test, 'w') as json_file:
        json.dump(test_data, json_file, indent=4)

    return train_data, val_data, test_data



#Start training our model and save it when it is finished


data_train = pd.read_csv('matching/data/input/20240603_training_data.csv')[['clustered', 'ecoinvent_prod', 'manual_certainty', 'hard no']]
data_test = pd.read_csv('matching/data/input/1210_general_testdata.csv', index_col = [0])

grouped_answers_train = group_answers(data_train)
grouped_train_data = transform_group(grouped_answers_train)
grouped_data_train = define_pos_neg(grouped_train_data)
data_dict_train = create_dictionary(grouped_data_train)

grouped_answers_test = group_answers(data_test)
grouped_test_data = transform_group(grouped_answers_test)
grouped_data_test = define_pos_neg(grouped_test_data)
data_dict_test = create_dictionary(grouped_data_test)

train_data, val_data, test_data = data_split(data_dict_train, data_dict_test)

retriever = DensePassageRetriever(
    document_store=InMemoryDocumentStore(),
    query_embedding_model=query_model,
    passage_embedding_model=passage_model,
    max_seq_len_query=64,
    max_seq_len_passage=256
)

retriever.train(
    data_dir=doc_dir,
    train_filename=train_filename,
    dev_filename=dev_filename,
    test_filename=test_filename,
    n_epochs=4,
    batch_size=16,
    grad_acc_steps=1,
    save_dir=save_dir,
    evaluate_every=100,
    embed_title=False,
    num_positives=1,
    num_hard_negatives=0,
)


