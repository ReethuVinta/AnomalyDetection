import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import pickle

cols = """
    duration,
protocol_type,
service,
flag,
src_bytes,
dst_bytes,
land,
wrong_fragment,
urgent,
hot,
num_failed_logins,
logged_in,
num_compromised,
root_shell,
su_attempted,
num_root,
num_file_creations,
num_shells,
num_access_files,
num_outbound_cmds,
is_host_login,
is_guest_login,
count,
srv_count,
serror_rate,
srv_serror_rate,
rerror_rate,
srv_rerror_rate,
same_srv_rate,
diff_srv_rate,
srv_diff_host_rate,
dst_host_count,
dst_host_srv_count,
dst_host_same_srv_rate,
dst_host_diff_srv_rate,
dst_host_same_src_port_rate,
dst_host_srv_diff_host_rate,
dst_host_serror_rate,
dst_host_srv_serror_rate,
dst_host_rerror_rate,
dst_host_srv_rerror_rate"""

cols = [c.strip() for c in cols.split(",") if c.strip()]
cols.append('target')

kdd = pd.read_csv("./data/kddcup.data",names=cols)

# attacks_type = {
# 'normal': 'normal',
# 'back': 'dos',
# 'buffer_overflow': 'u2r',
# 'ftp_write': 'r2l',
# 'guess_passwd': 'r2l',
# 'imap': 'r2l',
# 'ipsweep': 'probe',
# 'land': 'dos',
# 'loadmodule': 'u2r',
# 'multihop': 'r2l',
# 'neptune': 'dos',
# 'nmap': 'probe',
# 'perl': 'u2r',
# 'phf': 'r2l',
# 'pod': 'dos',
# 'portsweep': 'probe',
# 'rootkit': 'u2r',
# 'satan': 'probe',
# 'smurf': 'dos',
# 'spy': 'r2l',
# 'teardrop': 'dos',
# 'warezclient': 'r2l',
# 'warezmaster': 'r2l',
#     }

kdd.head(5)

kdd_std=kdd.std()
kdd_std=kdd_std.sort_values(ascending=True)

kdd.drop(["service","is_host_login","num_outbound_cmds"],axis=1,inplace=True)

encoder = preprocessing.LabelEncoder()
for c in kdd.columns:
    if str(kdd[c].dtype) == 'object': 
        kdd[c] = encoder.fit_transform(kdd[c])

scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
kdd[['dst_bytes','src_bytes']] = scaler.fit_transform(kdd[['dst_bytes','src_bytes']])

with open('./data/kdd_df.pth','wb') as f:
    pickle.dump(kdd,f) 