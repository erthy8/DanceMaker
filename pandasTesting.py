import pandas as pd

df_dance = pd.read_hdf('./data/final-The Robot- Fortnite Emote/videodata.hdf5', key='videodata')
print(df_dance)
print(len(df_dance))