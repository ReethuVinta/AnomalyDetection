import pandas as pd
import numpy as np

df1=pd.read_csv("./Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")
df2=pd.read_csv("./Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv")
df3=pd.read_csv("./Friday-WorkingHours-Morning.pcap_ISCX.csv")
df4=pd.read_csv("./Monday-WorkingHours.pcap_ISCX.csv")
df5=pd.read_csv("./Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv")
df6=pd.read_csv("./Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv")
df7=pd.read_csv("./Tuesday-WorkingHours.pcap_ISCX.csv")
df8=pd.read_csv("./Wednesday-workingHours.pcap_ISCX.csv")



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

df.info()
df.head()
print(df.shape)

for i in df.columns:
    df = df[df[i] != "Infinity"]
    df = df[df[i] != np.nan]
    df = df[df[i] != ",,"]
df[['Flow Bytes/s', ' Flow Packets/s']] = df[['Flow Bytes/s', ' Flow Packets/s']].apply(pd.to_numeric) 


print(df[' Bwd PSH Flags'].value_counts())
print(df[' Bwd URG Flags'].value_counts())
print(df['Fwd Avg Bytes/Bulk'].value_counts())
print(df[' Fwd Avg Packets/Bulk'].value_counts())
print(df[' Fwd Avg Bulk Rate'].value_counts())
print(df[' Bwd Avg Bytes/Bulk'].value_counts())
print(df[' Bwd Avg Packets/Bulk'].value_counts())
print(df['Bwd Avg Bulk Rate'].value_counts())
    
#Date preprocesing
df.drop([' Bwd PSH Flags'], axis=1, inplace=True)
df.drop([' Bwd URG Flags'], axis=1, inplace=True)
df.drop(['Fwd Avg Bytes/Bulk'], axis=1, inplace=True)
df.drop([' Fwd Avg Packets/Bulk'], axis=1, inplace=True)
df.drop([' Fwd Avg Bulk Rate'], axis=1, inplace=True)
df.drop([' Bwd Avg Bytes/Bulk'], axis=1, inplace=True)
df.drop([' Bwd Avg Packets/Bulk'], axis=1, inplace=True)
df.drop(['Bwd Avg Bulk Rate'], axis=1, inplace=True)

df.info()
print(df.shape)

df.replace([np.inf, -np.inf, -np.nan, np.nan], 0, inplace=True) 

print(df[' Label'].value_counts())


X = df.drop(' Label', 1)
New_y = pd.factorize(df[' Label'])[0] + 1



print(X.shape)
print(New_y.shape)
labels=np.reshape(New_y,(New_y.shape[0],1))
dataset=X.to_numpy()

# a = np.max(dataset,axis=0)
# print(a.shape)
# for i in range(a.shape[0]):
#     print(i,a[i])

print(dataset.shape)
print(labels.shape)
np.save('./data/dataset',dataset)
np.save('./data/labels',labels)