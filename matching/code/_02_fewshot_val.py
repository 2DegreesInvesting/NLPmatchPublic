import openai
import pandas as pd

from langchain.prompts import FewShotPromptTemplate
import numpy as np
import os
import datetime

from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate

def get_data(input_file_name, file_path):
    """
        This function reads in the dataset containing top 5 similarities from the DPR model,
        as well as the training data used for few-shot validation of GPT-3.5 and the prompts used.

    Attributes:
        input_file_name (pd.DataFrame): Dataframe containing top 5 product similarities
        file_path (str): path to the folder containing the prompt instructions for few-shot validation.

    Output:
        train (pd.DataFrame): Dataframe containing manually assessed product combinations
        data (pd.DataFrame): Dataframe containing top 5 similarities from the DPR model.
        prompt_prefix (str): Text file containing the prompt prefix for GPT.
    """

    data = pd.read_csv(input_file_name)
    train = pd.read_csv('matching/data/input/20240603_training_data_full.csv', index_col = [0])[['clustered', 'ecoinvent_prod', 'manual_match_certainty']].rename(columns = {'manual_match_certainty':'Result'})
    
    with open(file_path, 'r') as file:
    # Read the entire contents of the file into a string
       prompt_prefix = file.read()
    return train, data, prompt_prefix
    
# Function to generate prompt output using OpanAI's GPT-3.5-turbo-instruct engine
def prompt_output(base_prompt, table):
    """
        This function reads in the dataset containing top 5 similarities from the DPR model,
        as well as the training data used for few-shot validation of GPT-3.5 and the prompts used.

    Attributes:
        input_file_name (pd.DataFrame): Dataframe containing top 5 product similarities
        file_path (str): path to the folder containing the prompt instructions for few-shot validation.

    Output:
        train (pd.DataFrame): Dataframe containing manually assessed product combinations
        data (pd.DataFrame): Dataframe containing top 5 similarities from the DPR model.
        prompt_prefix (str): Text file containing the prompt prefix for GPT.
    """

    lst = []
    for europages, ecoinvent in zip(list(table.clustered), list(table.ecoinvent_prod)):
        prompt = base_prompt.format(clustered = europages, ecoinvent_prod = ecoinvent)

        response = openai.Completion.create(
            engine="gpt-3.5-turbo-instruct",
            prompt=prompt,
            temperature=0,
            max_tokens=10
        )
        lst.append({"clustered": europages, "ecoinvent_prod": ecoinvent, "output": response.choices[0].text.strip()})
        
    return table.merge(pd.DataFrame(lst), how = 'left', on = ['clustered', 'ecoinvent_prod'])


# Function to create example for FewShotPromptTemplate
def create_examples(train, prompt_prefix):

    """
        This function splits the prompt that is used in the GPT API in several building blocks,
        that are required for few-shot prompting.

    Attributes:
        train (pd.DataFrame): The dataframe used as examples for few-shot prompting
        prompt_prefix (str): Text file containing instructions for GPT-3.5.

    Output:
        examples (dict): Dictionary containing examples for few-shot prompting
        example_prompt (PromptTemplate): Describing the template of examples giving to GPT.
        prefix (str): Text file containing instructions for the matching validation.
        suffix (str): Text file containing product-pairs to be validated by GPT.
    """

    examples = train.to_dict('records')

    example_template = """
    clustered: {clustered}
    ecoinvent_prod: {ecoinvent_prod}
    output: {Result}
    """

    example_prompt = PromptTemplate(
        input_variables=["clustered", "ecoinvent_prod", "Result"],
        template=example_template
    )

    prefix = prompt_prefix
    suffix = """
    clustered: {clustered}
    ecoinvent_prod: {ecoinvent_prod}
    output:
    """
    return examples, example_prompt, prefix, suffix


# Function to initialize OpenAI Embeddings
def get_embedder():
    """
        This function imports the embedding model used for few-shot prompting.

    Attributes:
        None

    Output:
        embedder (OpenAIEmbeddings): Model used to embed examples for few-shot prompting
    """

    embedder = OpenAIEmbeddings(model='text-embedding-ada-002',
                                deployment='text-embedding-ada-002',
                                chunk_size=256)
    return embedder

# Function to define SemanticSimilarityExampleSelector
def define_example_selector(examples, embedder):
   
    """
        This function splits the prompt that is used in the GPT API in several building blocks,
        that are required for few-shot prompting.

    Attributes:
        examples (dict): Dictionary containing examples for few-shot prompting
        embedder (OpenAIEmbeddings): Model used to embed examples for few-shot prompting

    Output:
        example_selector (SementicSimilarityExampleSelector): Package used to search for the semantically best examples.
    """
   
    example_selector = SemanticSimilarityExampleSelector.from_examples(
        examples, embedder, FAISS, k=10
    )
    return example_selector


# Function to define FewShotPromptTemplate
def define_prompt_template(example_selector, example_prompt, prefix, suffix):
   
    """
        This function collects all building blocks for the few-shot prompt template and creates the prompt.

    Attributes:
        example_selector (SemanticSimilarityExampleSelector): Package used to search for the semantically best examples.
        example_prompt (PromptTemplate): Describing the template of examples giving to GPT.
        prefix (str): Text file containing instructions for the matching validation.
        suffix (str): Text file containing product-pairs to be validated by GPT.

    Output:
        few_shot_prompt_template (FewShotPromptTemplate): Package used to construct the few-shot prompt.
    """

    few_shot_prompt_template = FewShotPromptTemplate(
        example_selector = example_selector,
        example_prompt = example_prompt,
        prefix = prefix,
        suffix = suffix,
        input_variables=["clustered", "ecoinvent_prod"],
    )
    return few_shot_prompt_template

# Function to set the output directory
def set_output_directory(intermediate_folder):

    """
        This function creates the folder to store batches of the few-shot validation approach.

    Attributes:
        intermediate_folder (str): path to store the batches

    Output:
        None
    """

    cur_time = datetime.datetime.now()
    cur_time_form = cur_time.strftime("%Y.%m.%d.%H.%M.%S")
    batch_path = intermediate_folder + '\directory_batch\\' + str(cur_time_form)
    os.makedirs(str(batch_path))

    return

# Function to get batch output
def get_batch_output(data_list, output_file_path, few_shot_prompt_template):

    """
        This function loops over batches of the top 5 similarities dataframe, reads whether the specific batch has already
        been assessed based in a previous run, connects with the GPT API to assess the specific batch and writes the results
        to an output CSV file.

    Attributes:
        data_list (list): list of dataframes to be assess by GPT in batches
        output_file_path (str): path for the output files to be stored
        few_shot_prompt_template (FewShotPromptTemplate): template containing the few-shot prompt.

    Output:
        calc_total (csv): output file containing all the assessed batches of top 5 similarities by GPT-3.5 turbo.
    """

    # batch_path = set_output_directory(intermediate_folder)
    for idx, batch in enumerate(data_list):
        calc_total = pd.read_csv(output_file_path)
        min_row = min(batch["row_number"])
        max_row = max(batch["row_number"])
        print("\n\n")
        print("Starting batch number " + str(idx) + " with row_numbers " + str(min_row) + " to " + str(max_row) + ".")
        
        if max_row in calc_total['row_number'].values:
            print("Rows " + str(min_row) + " to " + str(max_row) + " already done.")
            continue
        else:
            fewshot = prompt_output(base_prompt=few_shot_prompt_template, table = batch)
            # temp_file_name = str(idx) + ".csv"
            # fewshot.to_csv(os.path.join(batch_path, temp_file_name))
    
            calc_total = pd.concat([calc_total, fewshot])
            calc_total.to_csv(output_file_path, index=False)

    return calc_total

# Main function to collect all results
def main():

    """
        This function creates all input and output paths for the required files, imports the API key,
        imports the data, splits the data into a list, creates the prompt, and validates the batches by GPT.

    Attributes:
        None

    Output:
        None
    """

    print("Executing fewshot validation")
    # openai.api_key = 
    os.environ['OPENAI_API_KEY'] = openai.api_key

    input_folder = 'matching/data/input'
    prompt_file_path = os.path.join(input_folder, "prompt_prefix_new.txt")
    current_date = datetime.datetime.now().strftime("%Y%m%d")
    intermediate_folder = 'matching/data/intermediate'
    file_name = f"{current_date}_fewshot_validation.csv"
    input_file_name = "20240604_similarities.csv"
    input_file_path = os.path.join(intermediate_folder, input_file_name)
    output_file_path = os.path.join(intermediate_folder, file_name)

    # empty = pd.DataFrame(columns = ['clustered', 'ecoinvent_prod', 'model_similarity', 'output', 'row_number'])
    # empty.to_csv(output_file_path, index=False)

    train, df_val, prompt_prefix = get_data(input_file_path, prompt_file_path)

    df_val["row_number"] = range(len(df_val))
    data_list = np.array_split(df_val, 100) 
    examples, example_prompt, prefix, suffix = create_examples(train, prompt_prefix)

    embedder = get_embedder()
    example_selector = define_example_selector(examples, embedder)
    few_shot_prompt_template = define_prompt_template(example_selector = example_selector, example_prompt = example_prompt, prefix=prefix, suffix=suffix)
    get_batch_output(data_list, output_file_path, intermediate_folder, few_shot_prompt_template)


if __name__ == "__main__":
    main()
