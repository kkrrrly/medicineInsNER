import os
with open('E:\\Code\\medicineInsNER\\data\\train\\0.ann','r',encoding='utf-8') as f:
    index = []
    type_data = []
    type_location = []
    for line in f:
        line = line.split()
        print(line)
        index.append(line[0])
        type_data.append(line[1])
        type_location.append((int(line[2]),int(line[3])))

with open('E:\\Code\\medicineInsNER\\data\\train\\0.txt','r',encoding='utf-8') as f:
    txtdata = f.read()
    txtdata = txtdata.replace('\u3000',' ')       

tag_txt = ['O'] * len(txtdata)
for index_num in range(len(type_data)):
    contrl_num = type_location[index_num][1] - type_location[index_num][0]
    if contrl_num == 1:
        tag_txt[type_location[index_num][0]] = type_data[index_num] + '_S'
    elif contrl_num == 2:
        tag_txt[type_location[index_num][0]] = type_data[index_num]+'_B'
        tag_txt[type_location[index_num][1]-1] = type_data[index_num]+'_E'
    else:
        tag_txt[type_location[index_num][0]] = type_data[index_num]+'_B'
        tag_txt[type_location[index_num][0]+1:type_location[index_num][1]] = [type_data[index_num]+'_I']*(contrl_num-2)
        tag_txt[type_location[index_num][1]-1] = type_data[index_num]+'_E'

print(tag_txt)