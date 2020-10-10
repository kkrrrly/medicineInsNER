import os
import ProcessDATA
with open('E:\\Code\\medicineInsNER\\data\\train\\9.txt','r',encoding='utf-8') as f:
    txtdata = f.read()
    txtdata = txtdata.replace('\u3000',' ')       

a = ProcessDATA.split_txt(txtdata)
print(a)


