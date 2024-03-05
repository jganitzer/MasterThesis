import pandas as pd
import datetime

dataRoot = "/data/_"
heritability = "/h40"

def filter_minor_allele_freq(df):
    mafThres = 0.01
    # Exclude id and phenotype columns
    df2 = df.drop(columns=['id', 'phenotype'], axis=1)
    cols_to_filter = df2.columns
    # Calculate minor allele frequency for each column and both alleles
    maf_1 = df[cols_to_filter].apply(lambda col: ((col == 1).sum() + 2*(col == 2).sum()) / (2*len(col)), axis=0)
    maf_2 = df[cols_to_filter].apply(lambda col: ((col == 1).sum() + 2*(col == 0).sum()) / (2*len(col)), axis=0)
    # Filter columns with MAF < 1%
    filtered_cols = cols_to_filter[(maf_1 >= mafThres) & (maf_2 >= mafThres)]
    filtered_cols_list = ['id', 'phenotype'] + filtered_cols.tolist()
    filtered_df = df.loc[:, filtered_cols_list]
    # Print number of removed columns
    num_removed_cols = len(cols_to_filter) - len(filtered_cols) + 2
    print(f"Number of removed columns: {num_removed_cols}")
    return filtered_df


df_snp_pheno = pd.read_parquet(dataRoot + heritability + '/snp_pheno_BPpos.parquet', engine='pyarrow')

filtered_df = filter_minor_allele_freq(df_snp_pheno)

filtered_df.to_parquet(dataRoot + heritability + "/snp_pheno_BPpos_maf_01.parquet")

print("Finished at " + str(datetime.datetime.now()))