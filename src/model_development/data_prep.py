# Functions for manipulating the datasets before the training phase

import pandas as pd

def transform_init_data(init_data: pd.DataFrame):
    '''
    Go through the steps to enforce the initially collected data to match column names etc. with the data that is specified to be used in training.

    Args:
        init_data: df from initial csv file

    Returns: 
        pd.DataFrame : enchanced df that can be joined with the generated data
    '''

    transformed_data = init_data
    transformed_data = transformed_data.rename(columns = {
            "User_ID" : "id", 
            "Gender" : "Sex"
        })

    # We want IDs to be unique after joining - we'll prefix those from this data
    # To be fair, these IDs are much higher than those observed in generated so this isn't a real risk.
    transformed_data["id"] = "in" + transformed_data["id"].astype(str)
    return transformed_data