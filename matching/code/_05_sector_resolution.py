import pandas as pd
import numpy as np
import openai
import os
import datetime
import time


def import_csv_files(file_path):
    '''
        This function imports all the relevant CSV files, for which the best tilt_subsector has to be found

    Attributes:
        file_path (path): A path to the folder where input data is stored

    Output:
        ep_companies (pd.DataFrame): Dataframe containing all company information scraped from europages
        matched_full (pd.DataFrame): Mapper with products that have succeeded the matching and filtering process
        ep_sector_mapper (pd.DataFrame): Dataframe mapping europages sectors to tilt sectors.

    '''

    ep_companies = pd.read_csv(f'{file_path}/input/ep_companies.csv')[['companies_id', 'sector', 'subsector', 'country', 'main_activity', 'clustered']].drop_duplicates().reset_index(drop=True)
    ep_sector_mapper = pd.read_csv(f'{file_path}/input/EP_tilt_sector_mapper.csv')
    matched_full = pd.read_csv(f'{file_path}/output/20240625_mapper_ep_ei.csv')

    return ep_companies, matched_full, ep_sector_mapper



def distinguish_sectors(ep_companies, matched_full, ep_sector_mapper):
    '''
        This function distinguishes the products which on one hand have 1 tilt_subsector, 
        and on the other hand have multiple tilt_subsectors for which we have to choose the best one.

    Attributes:
        ep_companies (pd.DataFrame): Dataframe containing all company information scraped from europages
        matched_full (pd.DataFrame): Mapper with products that have succeeded the matching and filtering process
        ep_sector_mapper (pd.DataFrame): Dataframe mapping europages sectors to tilt sectors.
    
    Output:
        one_sec_df (pd.DataFrame): Dataframe containing europages products for which only a single sector exists in europages
        more_sec_df_combined (pd.DataFrame): Dataframe containing europages products for which multiple sectors exist in europages
        
    '''

    #Select only the list of europages products that are not in the dataset of products that are matched to ecoinvent.
    no_match_head = ep_companies[~ep_companies['clustered'].isin(matched_full['ep_clustered'])]

    #Join with the sector mapper to get the tilt_subsector
    no_match_tilt_sector = no_match_head.merge(ep_sector_mapper, how = 'left', left_on = ['sector', 'subsector'], right_on = ['ep_sector', 'ep_subsector'])[['main_activity', 'clustered', 'tilt_subsector']].drop_duplicates().reset_index(drop=True)

    #Count the number of unique tilt_subsectors
    n_sectors_df = no_match_tilt_sector.groupby(['main_activity','clustered'])["tilt_subsector"].nunique().reset_index().sort_values("tilt_subsector", ascending=False)
    n_all_products = len(no_match_tilt_sector[['main_activity', "clustered"]].drop_duplicates())

    #Put the products with only 1 tilt_subsector in a dataframe
    one_sec_list = n_sectors_df[n_sectors_df["tilt_subsector"] == 1][['main_activity', 'clustered']].drop_duplicates()
    one_sec_df = no_match_tilt_sector.merge(one_sec_list, how = 'inner', on = ['main_activity', 'clustered'])
    n_row_one_sec_df = len(one_sec_df[['main_activity', 'clustered']].drop_duplicates())

    #Filter such that the remaining products are products with multiple sectors.
    more_sec_df = no_match_tilt_sector.merge(one_sec_list, how = 'left', indicator=True)
    more_sec_df = more_sec_df[more_sec_df['_merge'] == 'left_only'].drop(columns = '_merge')
    more_sec_df_combined = more_sec_df.groupby(['main_activity','clustered'])['tilt_subsector'].apply(list).reset_index(name='tilt_subsector_candidates')

    print("Out of " + str(n_all_products) + " main_activity / product combinations " + str(n_row_one_sec_df) + " are associated to only 1 sector.")

    return one_sec_df, more_sec_df_combined

# def prompt_output(base_prompt, table):
#     '''
#         This function calls for the GPT API to send our prompt to GPT and ask for the best tilt_subsector in return.

#     Attributes:
#         base_prompt (txt): Text file containing GPT instructions on selecting the best tilt_subsector.
#         table (pd.DataFrame): Table with multiple sector per europages product, for which the best needs to be chosen.

#     Output:
#         pd.DataFrame(lst) (pd.DataFrame): Dataframe containing chosen tilt_subsector for each combination 
#         of europages product and main_activity.
#     '''
#     lst = []
#     for main_activity, clustered, tilt_subsector_candidates in zip(list(table.main_activity), list(table.clustered), list(table.tilt_subsector_candidates)):
#         prompt = base_prompt.format(main_activity = main_activity, clustered = clustered, tilt_subsector_candidates = tilt_subsector_candidates)

#         response = openai.Completion.create(
#             engine="gpt-3.5-turbo-instruct",
#             prompt=prompt,
#             temperature=0,
#             max_tokens=10
#         )
#         lst.append({"main_activity": main_activity, "clustered": clustered, "tilt_subsector": response.choices[0].text.strip().replace("'", "")})

#     return pd.DataFrame(lst)

def prompt_output(base_prompt, table, max_requests_per_minute=90000):
    '''
        This function calls for the GPT API to send our prompt to GPT and ask for the best tilt_subsector in return.

    Attributes:
        base_prompt (txt): Text file containing GPT instructions on selecting the best tilt_subsector.
        table (pd.DataFrame): Table with multiple sector per europages product, for which the best needs to be chosen.

    Output:
        pd.DataFrame(lst) (pd.DataFrame): Dataframe containing chosen tilt_subsector for each combination 
        of europages product and main_activity.
    '''
    lst = []
    request_count = 0
    start_time = time.time()
    
    for main_activity, clustered, tilt_subsector_candidates in zip(list(table.main_activity), list(table.clustered), list(table.tilt_subsector_candidates)):
        # Check if we've hit the request limit for this minute
        if request_count >= max_requests_per_minute:
            elapsed_time = time.time() - start_time
            sleep_time = 60 - elapsed_time
            if sleep_time > 0:
                print(f"Rate limit of {max_requests_per_minute} requests per minute reached. Sleeping for {sleep_time:.2f} seconds.")
                time.sleep(sleep_time)
            # Reset for the next minute
            request_count = 0
            start_time = time.time()

        prompt = base_prompt.format(main_activity=main_activity, clustered=clustered, tilt_subsector_candidates=tilt_subsector_candidates)

        try:
            response = openai.Completion.create(
                engine="gpt-3.5-turbo-instruct",
                prompt=prompt,
                temperature=0,
                max_tokens=10
            )
            lst.append({"main_activity": main_activity, "clustered": clustered, "tilt_subsector": response.choices[0].text.strip().replace("'", "")})
            request_count += 1

        except openai.error.RateLimitError:
            print("Rate limit exceeded. Retrying after 60 seconds...")
            time.sleep(60)
            # Reset the timer and the request count after sleeping
            start_time = time.time()
            request_count = 0
            # Retry the API request
            response = openai.Completion.create(
                engine="gpt-3.5-turbo-instruct",
                prompt=prompt,
                temperature=0,
                max_tokens=10
            )
            lst.append({"main_activity": main_activity, "clustered": clustered, "tilt_subsector": response.choices[0].text.strip().replace("'", "")})
            request_count += 1

    return pd.DataFrame(lst)


def get_prompt(file_path, file_name):
    '''
        This function reads in the prompt for sector_resolving.

    Attributes:
        file_path (string): path to the folder with the prompt
        file_name (string): name of the prompt file

    Output:
        prompt_prefix (string): prompt with instructions for GPT
    '''

    with open(os.path.join(file_path, file_name), 'r') as file:
        # Read the entire contents of the file into a string
        prompt_prefix = file.read()

    return prompt_prefix


def combine_prompts(prompt_prefix):

    '''
        This function prepares the prompt and rows to be validated by GPT

    Attributes:
        prompt_prefix (str): prompt with instructions for GPT

    Output:
        total_prompt (str): Prompt with instructions for GPT including rows to be assessed
    '''

    prefix = prompt_prefix
    suffix = """
    main_activity: {main_activity}
    product:{clustered}
    candidates_list: {tilt_subsector_candidates}
    tilt_subsector:
    """
    total_prompt = prefix + suffix

    return total_prompt



def main():
    '''
    This function runs all the functions above, and performs some final preparations for the indicatorbefore code.
    '''
    print("Executing sector resolution")
    # openai.api_key = 
    os.environ['OPENAI_API_KEY'] = openai.api_key
    current_date = datetime.datetime.now().strftime("%Y%m%d")
    input_file_path = "matching/data"
    output_folder = "matching/data/output"
    ep_companies, matched_full, ep_sector_mapper = import_csv_files(input_file_path)
    one_sector, multiple_sectors = distinguish_sectors(ep_companies, matched_full, ep_sector_mapper)
    file_name = 'input/sector_resolve_prompt.txt'
    prompt_prefix = get_prompt(input_file_path, file_name)
    prompt = combine_prompts(prompt_prefix)

    best_sector = prompt_output(prompt, multiple_sectors)


    #In some cases, the answer provided by GPT included "The tilt_subsector would be", which needs to be removed for the indicatorbefore part.
    best_sector['tilt_subsector'] = best_sector['tilt_subsector'].apply(lambda x: x.replace('The tilt_subsector would be ', ''))
    best_sector['tilt_subsector'] = best_sector['tilt_subsector'].apply(lambda x: x.replace('No_match', 'no_match'))

    output_file_name = f"{current_date}_sector_resolve.csv"
    output_file_path = os.path.join(output_folder, output_file_name)
    sector_resolved = pd.concat([one_sector, best_sector])
    sector_resolved.to_csv(output_file_path, index=False)
    print("Sector resolution done")
    

if __name__ == '__main__':
    main()