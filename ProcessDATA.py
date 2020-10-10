import numpy as np 
import os

def get_train_data():
    fildir = os.path.join(os.getcwd(),'data','train')
    filenames = os.listdir(fildir)
    filenames_txt = []
    filenames_ann = []
    
    #获取文件名
    for filename in filenames:
        name = os.path.splitext(filename)
        if name[1] == '.txt':
            filenames_txt.append(name[0])
        elif name[1] == '.ann':
            filenames_ann.append(name[0])

    #print(filenames_txt)
    #interval = (1000 - window_size) // (batch-1)
    
    train_data = []
    for name in filenames_txt:
        charlis = []
        with open(os.path.join(fildir,'{}.txt'.format(name)),'r',encoding='utf-8') as f:
            txtdata = f.read()
            txtdata = txtdata.replace('\u3000',' ')
            print('Process {}.txt'.format(name))


        index = []#编号
        type_data = []#TAG类型
        type_location = []#TAG的位置
        with open(os.path.join(fildir,'{}.ann'.format(name)),'r',encoding='utf-8') as f:
            for line in f:
                line = line.split()
                index.append(line[0])
                type_data.append(line[1])
                type_location.append(((int(line[2]),int(line[3]))))

        #全文件标注
        tag_txt = ['O'] * len(txtdata)#此变量指向全文本标注的list
        #print(type_data)
        for index_num in range(len(type_data)):
            lenth = type_location[index_num][1] - type_location[index_num][0]
            if lenth == 1:
                tag_txt[type_location[index_num][0]] = type_data[index_num] + '_S'
            elif lenth == 2:
                tag_txt[type_location[index_num][0]] = type_data[index_num]+'_B'
                tag_txt[type_location[index_num][1]-1] = type_data[index_num]+'_E'
            else:
                tag_txt[type_location[index_num][0]] = type_data[index_num]+'_B'
                tag_txt[type_location[index_num][0]+1:type_location[index_num][1]-1] = [type_data[index_num]+'_I']*(lenth-2)
                tag_txt[type_location[index_num][1]-1] = type_data[index_num]+'_E'

        sentence_data,sentence_cutnum = split_txt(txtdata)
        #print(sentence_data)
        tag_sentence = []
        for cutnum in sentence_cutnum:
            tag_sentence.append(tag_txt[cutnum[0]:cutnum[1]]) 

        for i in range(len(sentence_data)):
            tupl_1 = list(sentence_data[i])
            tupl_2 = tag_sentence[i]
            train_data.append((tupl_1,tupl_2))

    #train_data 是一个list 每一个元素是([sentence.split],[tag]) 的tuple
    return train_data

def split_txt(txtdata):
    sentence_cutnum = []
    split_data = []
    txtdata.replace('\u3000',' ')
    for i in range(len(txtdata)):
        if not txtdata[i] == ' ':
            if i == 0:
                sentence_start = 0
            elif txtdata[i-1] == ' ' or txtdata[i-1] == '。' or txtdata[i-1] == '>':
                sentence_start = i
            elif txtdata[i] == '。' or i == len(txtdata)-1:
                sentence_end = i+1
                sentence_cutnum.append([sentence_start,sentence_end])
                split_data.append(txtdata[sentence_start:sentence_end])
            elif txtdata[i+1] ==' ':
                sentence_end = i+1
                sentence_cutnum.append([sentence_start,sentence_end])
                split_data.append(txtdata[sentence_start:sentence_end])
    #返回列表1 文本分成句子 列表2 开头和结尾的切片(位置需要结尾数-1)
    return split_data,sentence_cutnum
