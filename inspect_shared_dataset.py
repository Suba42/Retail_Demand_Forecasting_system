import pandas as pd

df = pd.read_csv('data/cleaned/cleaned_shared_dataset.csv')
print('shape=', df.shape)
print('columns=', df.columns.tolist())
print(df.head(2).to_string())
