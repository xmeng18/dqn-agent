key = input('name of the data:')
lines = open(key+".csv", "r").read().splitlines()

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv(os.getcwd()+'/GOOG1718.csv')

SP =  pd.read_csv(os.getcwd()+'/^GSPC.csv')

SP = SP[['Date','Volume']]
SP.columns = ['Date','SP_Volume']



full = pd.concat([df.set_index('Date'),SP.set_index('Date')],axis=1, join='inner').reset_index()

full['Volume_ratio'] = full.Volume/full.SP_Volume

full['Vol'] = full.Close.rolling(7).std().bfill()



tostandard = np.array([full['Volume_ratio'],full['Vol']]).T



full[['Volume_ratio','Vol']] = StandardScaler().fit_transform(tostandard)

full.to_csv(key+'_full.csv',index=False)
