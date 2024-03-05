import pandas as pd
import datetime

dataRoot = "/data/_"
heritability = "/h40"

def remove_single_value_columns(df):
    """
    Removes columns from a pandas dataframe where every row of the column is the same single value.

    Args:
    df (pandas.DataFrame): input dataframe.

    Returns:
    pandas.DataFrame: output dataframe with columns removed.
    """
    num_single_value_cols = (df.nunique() == 1).sum()
    print(f"Number of columns with just one single value: {num_single_value_cols}")
    return df.loc[:, (df.nunique() != 1)]


df_snp_pheno = pd.read_parquet(dataRoot + heritability + '/snp_pheno_BPpos.parquet', engine='pyarrow')

filtered_df = remove_single_value_columns(df_snp_pheno)

filtered_df.to_parquet(dataRoot + heritability + "/snp_pheno_BPpos_noStaticSNP.parquet")

print("Finished at " + str(datetime.datetime.now()))