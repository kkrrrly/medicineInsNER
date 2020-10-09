import numpy as np
import pandas as pd 
import os

def get_train_data(window_size,batch):
    fildir = os.path.join(os.getcwd(),'data','train')
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
            with open(os.path.join(fildir,filenames_txt[filenum]),'r',encoding='utf-8') as f:
                txtdata = f.read()
                txtdata = txtdata.replace('\u3000',' ')


            index = []
            type_data = []
            type_location = []
            with open(os.path.join(fildir,filenames_ann[filenum]),'r',encoding='utf-8') as f:
                for line in f:
                    line = line.split()
                    index.append(line[0])
                    type_data.append(line[1])
                    type_location.append(([int(line[2]),int(line[3])))

            #全文件标注
            tag_txt = ['O'] * len(txtdata)
            for index_num in range(len(type_data)):
                lenth = type_location[index_num][1] - type_location[index_num][0]
                if lenth == 1:
                    tag_txt[type_location[index_num][0]] = type_data[index_num] + '_S'
                elif lenth == 2:
                    tag_txt[type_location[index_num][0]] = type_data[index_num]+'_B'
                    tag_txt[type_location[index_num][1]-1] = type_data[index_num]+'_E'
                else:
                    tag_txt[type_location[index_num][0]] = type_data[index_num]+'_B'
                    tag_txt[type_location[index_num][0]+1:type_location[index_num][1]] = [type_data[index_num]+'_I']*(lenth-2)
                    tag_txt[type_location[index_num][1]-1] = type_data[index_num]+'_E'



            sentence_data,sentence_cutnum = split_txt(txtdata)
            tag_sentence = []
            for cutnum in sentence_cutnum:
               tag_sentence.append(tag_txt[cutnum[0]:cutnum[1]]) 
 
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

def split_txt(txtdata):
    sentence_cutnum = []
    split_data = []
    txtdata.replace('\u3000',' ')
    for i in range(len(txtdata)):
        if not txtdata[i] == ' ':
            if i == 0:
                sentence_start = 0
            elif txtdata[i-1] == ' ' or txtdata[i-1] == '。':
                sentence_start = i
            elif txtdata[i] == '。' or i == len(txtdata):
                sentence_end = i+1
                sentence_cutnum.append([sentence_start,sentence_end])
                split_data.append(txtdata[sentence_start:sentence_end])
            elif txtdata[i+1] ==' ':
                sentence_end = i+1
                sentence_cutnum.append([sentence_start,sentence_end])
                split_data.append(txtdata[sentence_start:sentence_end])
    #返回列表1 分好的文本 列表2 开头和结尾的切片(位置需要结尾数-1)
    return split_data,sentence_cutnum