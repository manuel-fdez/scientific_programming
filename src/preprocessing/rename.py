def rename_columns (df, dict_names):
    """
    Rename columns of a DataFrame according to a given dictionary.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    dict_names (dict): A dictionary mapping old column names to new column names.

    Returns:
    pd.DataFrame: The DataFrame with renamed columns.
    """
    return df.rename(columns=dict_names)

def parse_column(column_name):
    #Replace spaces for underscores
    result = column_name.replace(" ","_")
    return result