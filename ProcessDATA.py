import numpy as np
import pandas as pd 
import os

fildir = os.getcwd()+'/data/train'
filenames = os.listdir(fildir)
filenames_txt = []
filenames_ann = []

for filename in filenames:
    endname = os.path.splitext(filename)[1]
    if endname == '.txt':
        filenames_txt.append(filename)
    elif endname == '.ann':
        filenames_ann.append(filename)

#print(filenames_txt)
charlis = []
with open('./data/train/0.txt','r',encoding='utf-8') as f:
    txtdata = f.read()
    txtdata = txtdata.replace('\u3000',' ')
    for i in txtdata:
        charlis.append(i)

lbt = pd.DataFrame(data=charlis,columns=['charlis'])
lbt['feature'] = 'O'


index = []
rowdata = []
with open('./data/train/0.ann','r',encoding='utf-8') as f:
    for line in f:
        line = line.split()
        index.append(line[0])
        rowdata.append(line[1:])

label = pd.DataFrame(data=rowdata,index=index,columns=['Type','LocStart','LocEnd','Value'])
label[['LocStart','LocEnd']] = label[['LocStart','LocEnd']].astype(int)
label['LocEnd'] = label['LocEnd']-1

feature = ['O' for i in range(len(lbt.index))]
len(lbt.index)
len(feature)
def labelfuc(df):
    lbt.at[df['LocStart'],'feature'] =  df['Type']+'_B'
    if df['LocEnd']-df['LocStart'] != 1:
        lbt.loc[df['LocStart']+1:df['LocEnd']-1,'feature'] = [df['Type']+'_I' for i in range(df['LocEnd']-df['LocStart']-1)]
    lbt.at[df['LocEnd'],'feature'] =  df['Type']+'_E'

label.apply(labelfuc,axis=1)

pd.set_option('max_rows', None)
print(lbt)
pd.set_option('max_rows',200)