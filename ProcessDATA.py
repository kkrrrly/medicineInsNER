import numpy as np
import pandas as pd 
import os

def get_train_data(window_size,batch):
    fildir = os.getcwd()+'/data/train/'
    filenames = os.listdir(fildir)
    filenames_txt = []
    filenames_ann = []
    
    #获取文件名
    for filename in filenames:
        endname = os.path.splitext(filename)[1]
        if endname == '.txt':
            filenames_txt.append(filename)
        elif endname == '.ann':
            filenames_ann.append(filename)

    #print(filenames_txt)
    interval = (1000 - window_size) // (batch-1)
    for times in range(batch):
        train_data = []
        for filenum in range(0+times*interval,window_size+times*interval):
            charlis = []
            with open(fildir+filenames_txt[filenum],'r',encoding='utf-8') as f:
                txtdata = f.read()
                txtdata = txtdata.replace('\u3000',' ')
                for i in txtdata:
                    charlis.append(i)

            lbt = pd.DataFrame(data=charlis,columns=['charlis'])
            lbt['feature'] = 'O'


            index = []
            rowdata = []
            with open(fildir+filenames_ann[filenum],'r',encoding='utf-8') as f:
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

            tupl = ([],[])
            for index,row in lbt.iterrows():

                if index == len(lbt.index)-1 and len(tupl[0]) != 0:
                    train_data.append(tupl)
                
                if row[0] == ' ':
                    continue
                if row[0] == '。':
                    tupl[0].append(row[0])
                    tupl[1].append(row[1])
                    train_data.append(tupl)
                    tupl = ([],[])
                else:
                    tupl[0].append(row[0])
                    tupl[1].append(row[1])

        yield  train_data
