import pandas as pd
import os
import datetime
import bert_matching_full

def get_csv_files(input_folder, intermediate_file_path):

    """
        This function reads in the datasets containing europages preprocessed products, output from the GPT validation,
        companies data, geography mapping, main activity mapping, and ecoinvent data which has already been filtered on geography.

    Attributes:
        input_folder (str): path specifying the input dataset location
        intermediate_file_path (str): Path specifying location of intermediately created files

    Output:
        euro (pd.DataFrame): Table containing preprocessed europages products.
        match_data (pd.DataFrame): Table containing outputs from the GPT matching validation.
        ep_companies (pd.DataFrame): EP_companies file containing 
        geo_mapping (pd.DataFrame): Geography mapping table
        main_act_mapper (pd.DataFrame): Main activity mapping table.
        eco (pd.DataFrame). Ecoinvent product table, already filtered on used geographies.
    """

    euro = pd.read_csv(f'{input_folder}/product_categorisation.csv', sep = ';', encoding = 'latin-1')
    match_data = pd.read_csv(intermediate_file_path).drop_duplicates().reset_index(drop=True)
    ep_companies = pd.read_csv(f'{input_folder}/ep_companies.csv')[['companies_id', 'company_name', 'country_id', 'country','main_activity', 'clustered']].drop_duplicates().reset_index(drop=True)
    geo_mapping = pd.read_csv(f'{input_folder}/geography_mapper_v310.csv')[['country_id', 'lca_geo', 'priority']]
    main_act_mapper = pd.read_csv(f'{input_folder}/main_activity_mapper.csv')
    eco = pd.read_csv(f'{input_folder}/ecoinvent_v310_filtered.csv')[['Activity UUID & Product UUID', 'Activity UUID', 'Activity Name', 'Geography'
                                                                  ,'Special Activity Type','Product UUID', 'Reference Product Name']].rename(
                                                                      columns = {'Activity UUID & Product UUID': 'activity_uuid_product_uuid','Activity UUID': 'activity_uuid',
                                                                                  'Activity Name': 'lca_act_name','Geography': 'lca_geo', 'Special Activity Type': 'lca_act_type',
                                                                                    'Product UUID': 'product_uuid', 'Reference Product Name': 'ecoinvent_prod'})

    return euro, match_data, ep_companies, geo_mapping, main_act_mapper, eco

def get_output_mapping(map_data):

    '''
        This function takes the output of the GPT model, and turns the values of high. medium, low and no match into numerical values,
        such that the maximum value per europages product can be taken.

    Attributes:
        map_data (pd.DataFrame): Table with matching output from GPT validation.

    Output:
        map_data(pd.DataFrame): Table with matching output, including numeric categories of GPT validation scores.

    '''
    output_mapping = {
        'high': 3,
        'medium': 2,
        'low': 1,
        'no': 0
    }
    map_data['output_num'] = map_data['output'].map(output_mapping)

    return map_data


def filter_and_merge(input_data, ep_companies, main_act_mapper, eco, geo_mapping):

    '''
        This function takes the output of the GPT validation, and joins it with the geography mapper, main activity mapper,
        and EP_companies table, such that for each company the best product match is chosen.
    
    Attributes:
        input_data (pd.DataFrame): Table with matching output from GPT
        ep_companies (pd.DataFrame): Companies table from Europages companies, already joined and filtered on relevant attributes.
        main_act_mapper (pd.DataFrame): Main activity mapper
        eco (pd.DataFrame): ecoinvent dataset
        geo_mapping (pd.DataFrame): Geography mapping

    Output:
        highest_prio_all (pd.DataFrame): Table containing products matched to ecoinvent with GPT validation, matched to the 
        companies table, filtered on main_activity and prioritized on geography.
    '''

    #Take the subset of products that have at least a low match.
    matched = input_data[input_data['output_num'] != 0]

    #Make the values of geography lower case for mapping.
    eco['lca_geo'] = eco['lca_geo'].str.lower()

    #This piece takes the maximum match value per europages product, and maps it back to the dataframe such that only the maximum match value is taken.
    max_output = pd.DataFrame(matched.groupby('clustered')['output_num'].max()).reset_index()
    out_max_output = matched.merge(max_output, how = 'inner')

    #For europages products that have multiple ecoinvent products with the same maximum match certainty, the one with the highest DPR similarity is chosen.
    max_sim = pd.DataFrame(out_max_output.groupby('clustered')['model_similarity'].max().reset_index())
    out_max_output_max_sim = out_max_output.merge(max_sim, how = 'inner')

    #The resulting dataframe is then joined with the companies table, the main activity mapper, ecoinvent, and geography.
    merged_all = (
        ep_companies.merge(main_act_mapper, how='inner', on='main_activity')
        .merge(out_max_output_max_sim, how='inner', on='clustered')
        .merge(eco, how='inner', on=['ecoinvent_prod', 'lca_act_type'])
        .merge(geo_mapping, how='inner', on=['country_id', 'lca_geo'])
    )

    #For each company having multiple products, the highest priority (lowest number) is being chosen.
    highest_prio = pd.DataFrame(merged_all.groupby('companies_id')['priority'].min()).reset_index()
    highest_prio_all = merged_all.merge(highest_prio, how = 'inner')

    return highest_prio_all


def get_best_activity(data):

    '''
        This function splits the output of filter_and_merge() into europages product+ ecoinvent activity type combinations,
        as the possibility still exists that after filtering on geography and main activity, a single ecoinvent product can 
        be used in multiple activities. Therefore we use BERT to select the best activity for each of these combinations.

    Attributes:
        data (pd.DataFrame): Dataframe containing the filtered matches based on priority and main activity.

    Output:
        best_act_data (pd.DataFrame): Dataframe containing the filtered matches, for which the best matching activity has been chosen.

    '''
    
    split_dataframes = {}
    
    #Group by clustered and ecoinvent activity type, and select the best activity.
    best_activities = pd.DataFrame(columns = ['clustered', 'lca_act_name', 'lca_act_type'])
    groups = data[['clustered', 'lca_act_name', 'lca_act_type']].groupby(['clustered', 'lca_act_type'])

    for (value1, value2), group_df in groups:
        key = f"{value1}_{value2}"
        split_dataframes[key] = group_df
        best_activity = bert_matching_full.selecting_best_activity(group_df)
        best_activities = pd.concat([best_activities, best_activity])

    best_act_data = data.merge(best_activities, how = 'inner').reset_index(drop=True)
    return best_act_data


def create_indicatorbefore_format(input, euro):
    '''
        This function prepares the output for the indicatorbefore package, adds necessary columns and correct column names.

    Attributes:
        input (pd.DataFrame): Dataframe containing the unique best matches from get_best_activity()
        euro (pd.DataFrame): Europages products that have been preprocessed.

    Output: 
        output (pd.DataFrame): Dataframe containing output of matching process, prepared for indicatorBefore input.
    '''

    matched_full = input.merge(euro, how = 'inner', on = 'clustered')
    matched_full['group_var'] = matched_full['clustered'] + '-.-' + matched_full['country'] + '-.-' + matched_full['main_activity']
    matched_full['multi_match'] = False
    output = matched_full[['group_var', 'clustered_id', 'country', 'main_activity', 'clustered', 'activity_uuid_product_uuid', 'multi_match', 'output']].rename(
        columns = {'clustered_id': 'ep_id', 'clustered': 'ep_clustered', 'country': 'ep_country', 'main_activity': 'ep_main_act', 'output': 'completion'}
    ).drop_duplicates().reset_index(drop=True)

    return output


def main():

    '''
    This function executes the above functions in correct order, and sends the output to csv files.
    '''

    print("Executing matches full script")
    current_date = datetime.datetime.now().strftime("%Y%m%d")
    input_folder = 'matching/data/input'
    intermediate_folder = 'matching/data/intermediate'
    output_folder = 'matching/data/output'
    input_file_name = "20240611_finetune_validation_dpr.csv"
    file_name = f"{current_date}_mapper_ep_ei.csv"
    intermediate_file_path = os.path.join(intermediate_folder, input_file_name)
    output_file_path = os.path.join(output_folder, file_name)


    euro, input, ep_companies, geo_mapping, main_act_mapper, eco = get_csv_files(input_folder, intermediate_file_path)
    mapped_data = get_output_mapping(input)
    filtered_and_merged = filter_and_merge(mapped_data, ep_companies, main_act_mapper, eco, geo_mapping)
    best_act_data = get_best_activity(filtered_and_merged)
    indicatorbefore_input = create_indicatorbefore_format(best_act_data, euro)
    indicatorbefore_input.to_csv(output_file_path, index=False)
    print("Matched full script done")


if __name__ == "__main__":
    main()