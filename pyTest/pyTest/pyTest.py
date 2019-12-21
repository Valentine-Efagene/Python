import numpy as np

import pandas as pd
df_data = pd.read_excel("mtn_monthly.xlsx")
data = df_data.values
x_train = np.ravel(data.T[0])

print(x_train)
