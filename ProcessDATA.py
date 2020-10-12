import numpy as np 
import os
import re

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
    #对每个文件循环
    for name in filenames_txt:
        charlis = []
        with open(os.path.join(fildir,'{}.txt'.format(name)),'r',encoding='utf-8') as f:
            txtdata = f.read()
            txtdata = txtdata.replace('\u3000',' ')
            print('Process {}.txt'.format(name))


        index = []#编号
        type_named_entity = []#类型+命名体
        with open(os.path.join(fildir,'{}.ann'.format(name)),'r',encoding='utf-8') as f:
            for line in f:
                line = line.split()
                index.append(line[0])
                type_named_entity.append((line[1],line[-1]))
                #type_location.append(((int(line[2]),int(line[3]))))
                #named_entity.append(line[-1])
        
        #增强当前文本标注
        type_data,type_location = enhanced_annotation(txtdata,type_named_entity)
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
        #以相同的下标切割全文件标注的list
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

def enhanced_annotation(txtdata,type_named_entity):
    #(类型，命名体)
    type_named_entity = list(set(type_named_entity))
    print(type_named_entity)
    new_type_list = []
    type_location = []
    for n in range(len(type_named_entity)):
        for same in re.finditer(type_named_entity[n][1],txtdata):
            new_type_list.append(type_named_entity[n][0])
            type_location.append((same.start(),same.end()))


    #删除细类中重复的大类项
    end_list = []
    
    for n in range(len(type_location)):
        end_list.append(type_location[n][1])

    end_list_set = set(end_list)
    if not len(end_list_set) == len(end_list):
        for item in end_list_set:
            end_list.remove(item)
        for endnum in end_list:
            c = []
            for n in range(len(type_location)):
                if type_location[n][1] ==  endnum:
                    c.append((n,type_location[n]))
            
            #print(c)
            if c[0][1][1]-c[0][1][0] > c[1][1][1]-c[1][1][0]:
                type_location.remove(c[1][1])
                #print('删除',new_type_list[c[1][0]])
                new_type_list.pop(c[1][0])
            else:
                #print('删除',c[1][1])
                type_location.remove(c[0][1])
                #print('删除',new_type_list[c[1][0]])
                new_type_list.pop(c[0][0])

    
    

    return new_type_list,type_location