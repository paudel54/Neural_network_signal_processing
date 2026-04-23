import numpy as np
import pickle
from tools.load_physionet_data import load_raw_data, load_raw_data_local, ptbxl_save

# load from local folder
rec_file = 'data_raw/ptb_xl/RECORDS'
local_dir = 'data_raw/ptb_xl'
data = load_raw_data_local(rec_file, local_dir, sr=500)
data = np.array(data)
# save each record separately as a numpy array
dir = 'data/ptb_xl_500hz'
ptbxl_save(data, dir)

# load MIT-BIH noise database locally
import pandas as pd
import wfdb
rec_file_noise = 'nstdb/noise-stress/RECORDS'
local_dir_noise = 'nstdb/noise-stress'
records_list_noise = pd.read_csv(rec_file_noise, sep='/n', header=None, names=['filename'], engine='python')
data_noise = []
for f in records_list_noise.filename:
    content, meta = wfdb.rdsamp(f"{local_dir_noise}/{f}")
    data_noise.append(content)

# save
pickle_out = open("data_noise.pickle", "wb")
pickle.dump(data_noise, pickle_out)
pickle_out.close()
