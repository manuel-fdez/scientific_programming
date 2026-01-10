import pandas as pd

def aggregate(input_df, group_by_col) -> pd.DataFrame:
    """
    Aggregates the DataFrame by
    grouping variable using pandas groupby. 

    Inputs:
        - input_df (pd.DataFrame): The input DataFrame.
        - group_by_col (str): The column to group by.

    Outputs:
        - pd.DataFrame: The aggregated DataFrame.
    """
    # Select all columns except diagnosis and diagnosis_label
    cols = input_df.columns.difference(["diagnosis", "diagnosis_label"])
    aggregated_df = input_df[cols].groupby(group_by_col).mean().reset_index()
    return aggregated_df