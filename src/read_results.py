# %%

import json
import numpy as np

path = '../results/result.json'

with open(path) as f:
  data = json.load(f)

# %%
data.keys()
# %%
analyse = 'Treppe'
relevant = ['dec']
joint = 'hip'
err = 'mse'

all_errs = []

for subjn in data.keys():
    subject = data[subjn][analyse]

    for rel in relevant:
        measures = subject[rel]

        for mearurement in measures:

            if joint in mearurement.keys():
                if rel == 'imu':
                    if mearurement[joint][err] < 10:
                        all_errs.append(mearurement[joint][err])
                else:
                    if mearurement[joint][err] < 60:
                        all_errs.append(mearurement[joint][err])

print(np.mean(all_errs))
print(np.std(all_errs))

# %%
all_errs
# %%
