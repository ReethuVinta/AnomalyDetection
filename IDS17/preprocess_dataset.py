import pandas as pd
import numpy as np
import os 

os.makedirs('data',exist_ok=True)

# Loading csv files of IDS2017 dataset
df1=pd.read_csv("./data/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")
df2=pd.read_csv("./data/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv")
df3=pd.read_csv("./data/Friday-WorkingHours-Morning.pcap_ISCX.csv")
df4=pd.read_csv("./data/Monday-WorkingHours.pcap_ISCX.csv")
df5=pd.read_csv("./data/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv")
df6=pd.read_csv("./data/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv")
df7=pd.read_csv("./data/Tuesday-WorkingHours.pcap_ISCX.csv")
df8=pd.read_csv("./data/Wednesday-workingHours.pcap_ISCX.csv")


# Concatenation of dataframes
df = pd.concat([df1,df2])
del df1,df2
df = pd.concat([df,df3])
del df3
df = pd.concat([df,df4])
del df4
df = pd.concat([df,df5])
del df5
df = pd.concat([df,df6])
del df6
df = pd.concat([df,df7])
del df7
df = pd.concat([df,df8])
del df8


for i in df.columns:
    df = df[df[i] != "Infinity"]
    df = df[df[i] != np.nan]
    df = df[df[i] != ",,"]
df[['Flow Bytes/s', ' Flow Packets/s']] = df[['Flow Bytes/s', ' Flow Packets/s']].apply(pd.to_numeric) 


# Removing these columns as their value counts are zero
df.drop([' Bwd PSH Flags'], axis=1, inplace=True)
df.drop([' Bwd URG Flags'], axis=1, inplace=True)
df.drop(['Fwd Avg Bytes/Bulk'], axis=1, inplace=True)
df.drop([' Fwd Avg Packets/Bulk'], axis=1, inplace=True)
df.drop([' Fwd Avg Bulk Rate'], axis=1, inplace=True)
df.drop([' Bwd Avg Bytes/Bulk'], axis=1, inplace=True)
df.drop([' Bwd Avg Packets/Bulk'], axis=1, inplace=True)
df.drop(['Bwd Avg Bulk Rate'], axis=1, inplace=True)

# Replacing nans, infs with zero's
df.replace([np.inf, -np.inf, -np.nan, np.nan], 0, inplace=True) 

X = df.drop(' Label', 1)
New_y = pd.factorize(df[' Label'])[0] + 1

labels = np.reshape(New_y,(New_y.shape[0],1))
dataset = X.to_numpy()

print("Shape of dataset(X) is", dataset.shape)
print("Shape of labels(Y) is", labels.shape)


np.save('./data/dataset.npy',dataset)
np.save('./data/labels.npy',labels)