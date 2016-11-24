import xgboost as xgb
import csv
from scipy.sparse import csr_matrix

POSITION_ID = 0
POSITION_PRODUCT_ID = 1
POSITION_HELPFULNESS_NUMERATOR = 4
POSITION_HELPFULNESS_DENOMINATOR = 5
POSITION_PREDICTION = 6

#Leo archivo de entrenamiento
csv_file = open('../../data/train.csv',encoding='utf8')
train_file = csv.reader(csv_file)
rows = next(train_file)

train_data = []
prediction = []

print ("Leyendo archivo de entrenamiento...")
for row in train_file:
    train_data.append([int(row[POSITION_HELPFULNESS_NUMERATOR]),int(row[POSITION_HELPFULNESS_DENOMINATOR])])
    prediction.append(float(row[POSITION_PREDICTION]))
print("OK\n")
csv_file.close()
del train_file

#Leo archivo de test
csv_file = open('../../data/test.csv',encoding='utf8')
data_file = csv.reader(csv_file)
rows = next(data_file)

test_data = []
idd = []

print ("Leyendo archivo de test...")
for row in data_file:
	idd.append(row[POSITION_ID])
	test_data.append([int(row[POSITION_HELPFULNESS_NUMERATOR]),int(row[POSITION_HELPFULNESS_DENOMINATOR])])
print("OK\n")
csv_file.close()
del data_file

train_data = csr_matrix(train_data)
test_data = csr_matrix(test_data)

print ("Inicio de xgboost..")
dtrain = xgb.DMatrix( train_data, label=prediction)
dtest = xgb.DMatrix( test_data )
print("OK\n")

param = {'booster': 'dart',
         'max_depth': 5, 
         'learning_rate': 0.1,
         'objective': 'reg:linear', 
         'silent': True,
         'sample_type': 'uniform',
         'normalize_type': 'tree',
         'rate_drop': 0.1,
         'skip_drop': 0.5
         }
num_round = 500

print ("Comienzo a generar el modelo...")
bst = xgb.train(param, dtrain, num_round)
print("OK\n")

print ("Predicciones...")
preds = bst.predict(dtest, ntree_limit=num_round)
print("OK\n")

print ("Creando submission...")
csvsalida = open('../../data/subm.csv','w')
salida = csv.writer(csvsalida)
salida.writerow(['Id','Prediction'])

i = 0
for i in range(len(idd)):
	salida.writerow ([idd[i], preds[i]])
csvsalida.close()
print("OK\n")