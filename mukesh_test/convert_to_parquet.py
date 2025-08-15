import pandas as pd


df = pd.read_csv("outputs_math_500-Q2.5-3B-Ins.csv")

# just make sure the df has list for predictions
if 'critique' in df.columns:
    df['critique'] = [[df.iloc[i]['critique']] for i in range(len(df))]


if 'final_response' in df.columns:
    df['final_response'] = [[df.iloc[i]['final_response']] for i in range(len(df))]

if 'initial_response' in df.columns:
    df['initial_response'] = [[df.iloc[i]['initial_response']] for i in range(len(df))]


df.to_parquet("outputs_no_rev_3b_ins.parquet")

