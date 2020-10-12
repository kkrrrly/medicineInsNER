import os
import ProcessDATA
with open('E:\\Code\\medicineInsNER\\data\\train\\799.txt','r',encoding='utf-8') as f:
    txtdata = f.read()
    txtdata = txtdata.replace('\u3000',' ')       

index = []#编号
type_named_entity = []#类型+命名体
with open('E:\\Code\\medicineInsNER\\data\\train\\799.ann','r',encoding='utf-8') as f:
    for line in f:
        line = line.split()
        index.append(line[0])
        type_named_entity.append((line[1],line[-1]))
#增强当前文本标注
type_data,type_location = ProcessDATA.enhanced_annotation(txtdata,type_named_entity)
a = ProcessDATA.split_txt(txtdata)
#print(a)
#print(type_data,type_location)


