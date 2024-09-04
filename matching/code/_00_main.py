from _01_DPR_similarities import main as DPR_similarities_main
from _02_fewshot_val import main as fewshot_validation_main
from _03_matched_full import main as matched_full_main
# from _04_matched_no_main import main as matched_no_main_main
from _05_sector_resolution import main as sector_resolution_main


def run_scripts():
    """
        The function loads the different scripts to map Europages data to Ecoinvent data,
        providing csv files as output. 
    Attributes:
        None

    Scripts:
        DPR_similarities_main() -> .csv
            Returns a CSV file in the data/intermediate folder called *current_date*_similarities.csv,
            containing top 5 similarity scores for Europages and Ecoinvent products provided by the DPR model.

        fewshot_validation_main() -> .csv
            Returns a CSV file in the data/intermediate folder called *current_date*_fewshot_validation.csv,
            containing GPT validation scores for each of the top 5 candidates between Europages products and Ecoinvent products.
            
        matched_full_main() -> .csv
            Returns a CSV file in the data/output folder called *current_date*_mapper_ep_ei.csv,
            containing the best matches between Europages and Ecoinvent products, which are filtered on geography and main_activity.

        matched_no_main_main() -> .csv
            Returns a CSV file in the data/output folder called *current_data*_multi_match.csv,
            containing companies with a match to ecoinvent, but who don't have a main_activity in europages.    

        sector_resolution_main() -> .csv
            Returns a CSV file in the data/output folder called *current_date*_sector_resolve.csv,
            containing products without a match to ecoinvent but are assigned a tilt_subsector.
    """
 
    print("Starting running main")
    # DPR_similarities_main()
    fewshot_validation_main()
    matched_full_main()
    # # matched_no_main_main()
    # sector_resolution_main()
    print("Running main done")

if __name__ == "__main__":
    run_scripts()
