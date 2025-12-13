
import pandas as pd
def get_text_data(x):
    return x['Titre'].astype(str) + " " + x['Publisher'].astype(str)
def get_numeric_data(x):
    return x[['oa_works', 'oa_cited', 'oa_found', 'cr_has_doi', 'Impact_Ratio']]
