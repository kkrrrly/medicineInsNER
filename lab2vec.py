#13类实体，共39维向量
import copy
zero = [0 for i in range(39)]
group = []
for i in range(39):
    vct = copy.copy(zero)
    vct[i]=1
    group.append(vct)

lab2vct = {
        'DRUG_B':group[0],
        'DRUG_I':group[1],
        'DRUG_E':group[2],
        'DRUG_INGREDIENT_B':group[3],
        'DRUG_INGREDIENT_I':group[4],
        'DRUG_INGREDIENT_E':group[5],
        'DISEASE_B':group[6],
        'DISEASE_I':group[7],
        'DISEASE_E':group[8],
        'SYMPTOM_B':group[9],
        'SYMPTOM_I':group[10],
        'SYMPTOM_E':group[11],
        'SYNDROME_B':group[12],
        'SYNDROME_I':group[13],
        'SYNDROME_E':group[14],
        'DISEASE_GROUP_B':group[15],
        'DISEASE_GROUP_I':group[16],
        'DISEASE_GROUP_E':group[17],
        'FOOD_B':group[18],
        'FOOD_I':group[19],
        'FOOD_E':group[20],
        'FOOD_GROUP_B':group[21],
        'FOOD_GROUP_I':group[22],
        'FOOD_GROUP_E':group[23],
        'PERSON_GROUP_B':group[24],
        'PERSON_GROUP_I':group[25],
        'PERSON_GROUP_E':group[26],
        'DRUG_GROUP_B':group[27],
        'DRUG_GROUP_I':group[28],
        'DRUG_GROUP_E':group[29],
        'DRUG_DOSAGE_B':group[30],
        'DRUG_DOSAGE_I':group[31],
        'DRUG_DOSAGE_E':group[32],
        'DRUG_TASTE_B':group[33],
        'DRUG_TASTE_I':group[34],
        'DRUG_TASTE_E':group[35],
        'DRUG_EFFICACY_B':group[36],
        'DRUG_EFFICACY_I':group[37],
        'DRUG_EFFICACY_E':group[38],
        'O':zero 
}
print(lab2vct)