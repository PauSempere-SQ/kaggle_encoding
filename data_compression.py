#%%
import pandas as pd 

train = pd.read_csv(r'.\data\train.csv', index_col = "id")
test = pd.read_csv(r'.\data\test.csv', index_col = "id")

# %%
name_map = {"target":"label"}
train = train.rename(name_map, axis = 1)
train.head()

# %%
train.to_parquet(fname = r'.\data\train.parquet')
test.to_parquet(fname = r'.\data\test.parquet')

# %%
