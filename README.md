# NLPmatchPublic

This is a repository showing code, data and methodology used to match Europages products with the best related Ecoinvent product.
There are 2 subfolders present, containing code and data:

Code:
1. _00_main.py: The main python script which ignites the process and runs all .py scripts in subsequent order.
2. _01_DPR_similarities.py: Reads the finetuned DPR model containing embeddings of products, and selects for each Europages product the top 5 candidates for best fitting Ecoinvent product based on a similarity score.
3. _02_fewshot_val.py: For each of the top 5 candidates for a Europages product, the Ecoinvent product matches are being validated by a finetuned GPT model, giving a positive or negative validation result.
4. _04_matched_full.py: After GPT validation, the matches are further filtered on Activity, Main Activity and Geography to make sure that the matches fit the best observation in Ecoinvent.
5. _05_sector_resolution.py: For companies of which no product was assigned en Ecoinvent product, the best fitting tilt_subsector still has to be found. This is done with a GPT prompt.

Data:
1. Input: This folder contains all datasets that serve as input for the matching process, but might also be used in other processes within the tilt pipeline.
2. Intermediate: Contains intermediate datasets that are created in the matching process, but are not part of the end result and are therefore stored in an intermediate folder.
3. Output: Contains output files that are result of the matching process, and are used in the next part of the tilt data pipeline.