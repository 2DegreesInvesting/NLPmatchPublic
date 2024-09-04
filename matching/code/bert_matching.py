import pandas as pd
import bert_MiniLM

data = pd.DataFrame(columns = ['europages_prod', 'ecoinvent_prod', 'cos_sim_bert_mini']).to_csv('matching/data/intermediate/20240604_bert_candidates.csv', index=None)


euro = pd.read_csv('matching/data/input/20240603_EP_compare.csv').rename(columns = {'clustered': 'europages_prod'})
eco = pd.read_csv('_00_production/data/input/ecoinvent_complete.csv').rename(columns = {'Reference Product Name': 'ecoinvent_prod'})

def product_matching(selected, candidate):
    """Returns DataFrame of pairwise cosine similarities
    
    Calculates pairwise cosine similarities of all entries in europ_products
    against econ_products and returns them as a dataframe. 

    Parameters:
    selected (DataFrame): List of all product-products from europages data.
    candidate (list): List of all reference products from ecoinvent data.
    
    Returns:
    pd.DataFrame: Dataframe holding cosine similarities between products.

    """
    best_match_list = list()
    selected_products = selected["europages_prod"].tolist()
    candidate_prods_list = list(set(candidate["ecoinvent_prod"].tolist()))


    for prod in selected_products:
        data = pd.read_csv('matching/data/intermediate/20240604_bert_candidates.csv')
        column_check = data["europages_prod"].drop_duplicates().tolist()
        if prod in column_check:
            continue
        else:
            matches_df = bert_MiniLM.get_cos_sim_bert_mini(list_a = [prod], list_b = candidate_prods_list)
            matches_df["a"] = prod
            matches_df["b"] = candidate_prods_list

            best_matches = matches_df.sort_values(by = ["cos_sim_bert_mini"], ascending = False).rename(columns = {'a':'europages_prod', 'b':'ecoinvent_prod'}).head(5) 
            best_match_prod = best_matches["europages_prod"].head(1).item()
            best_match_val = best_matches["cos_sim_bert_mini"].head(1).item()
            index = selected_products.index(prod)

            print(str(index+1) + "/" + str(len(selected_products)) + " for '" + prod + "' the matching algorithm identified '" + best_match_prod + "' as best matching product with a match of " + str(best_match_val) + ".")
            
            data = pd.concat([data,best_matches])
            data.to_csv("matching/data/intermediate/20240604_bert_candidates.csv", index=None)

    return data


matching_output = product_matching(euro, eco)


