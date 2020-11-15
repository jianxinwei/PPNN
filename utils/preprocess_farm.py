import pandas as pd
import os
import numpy as np

import ipdb

farm_path = "../data/farm/farm-ads-vect"

farm_file = open(farm_path)
lines = farm_file.readlines()
farm_file.close()
print(len(lines))

farm_vector = np.zeros([len(lines),55000])
farm_label = np.zeros(len(lines))
for l in range(len(lines)):
    line = lines[l].strip().split(' ')
    farm_label[l] = int(line[0])
    for i in range(1, len(line)):
        index = int(line[i].split(':')[0])
        farm_vector[l][index-1] = 1

farm_label[farm_label < 0] = 0.0
farm_label = farm_label.reshape(-1,1)

farm_data = np.hstack((farm_vector,farm_label))
farm_df = pd.DataFrame(farm_data)
# np.dump(os.path.join(os.path.split(farm_path)[0], 'new_farm_whole.csv'), sep=';', index=False)