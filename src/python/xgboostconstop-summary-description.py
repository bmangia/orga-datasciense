import xgboost as xgb
import csv
from scipy.sparse import csr_matrix
from sklearn import svm
from scipy.sparse import *
from nltk import word_tokenize
from nltk.corpus import stopwords
import math


POSITION_HELPFULNESS_NUMERATOR = 4
POSITION_HELPFULNESS_DENOMINATOR = 5

def purificar_texto(texto,stop):
    lista_de_palabras = []
    for palabra in texto:
        if not palabra.lower() in stop:
            lista_de_palabras.append(palabra.lower())
    return lista_de_palabras
            
    
def seteando_stopwords():
    negadores = set(('but','no','nor','not','very','don','should'))
    puntuacion  = set(('.',',',':',';','(',')','[',']','{','}','<','>','!','?','/','...','""',"''",'--','-'))
    stop = set(stopwords.words('english')) - negadores
    stop_completo = set()
    stop_completo.update(stop)
    stop_completo.update(puntuacion)
    
    return stop_completo


def generando_diccionario(diccionario,texto,indice):
    diccionario_aux = {}
    for palabra in texto:
            if palabra in diccionario and not( palabra in diccionario_aux):
                diccionario[palabra][1] = diccionario[palabra][1] + 1

            elif not(palabra in diccionario):
                diccionario[palabra] = [indice,1]
                indice += 1

            diccionario_aux[palabra] = 1

    return indice


def tf_idf(diccionario,informacion,extra_features):

    matriz_resumen = lil_matrix((len(informacion),len(diccionario)+2))
    i = 0


    for doc in informacion:
        for palabra in doc:
            if palabra in diccionario:  
                matriz_resumen[i,diccionario[palabra][0]] = matriz_resumen[i,diccionario[palabra][0]]+ math.log(len(informacion) /diccionario[palabra][1],2)

        matriz_resumen[i,len(diccionario)-1] = extra_features[i][0]
        matriz_resumen[i,len(diccionario)] = extra_features[i][1]
        i = i + 1

    return matriz_resumen


stop = seteando_stopwords()

#Leo archivo de entrenamiento
csv_file = open('../../data/train.csv',encoding='utf8')
train_file = csv.reader(csv_file)
rows = next(train_file)

train_data = []
clase = []
lista_de_palabras = []
diccionario = {}
i = 0
informacion =[]
extra_features = []

for row in train_file:
    lista_de_palabras = purificar_texto(word_tokenize(row[8]),stop)
    i = generando_diccionario(diccionario,lista_de_palabras,i)
    lista_de_palabras_2 = purificar_texto(word_tokenize(row[9]),stop)
    i = generando_diccionario(diccionario,lista_de_palabras_2,i)
    informacion.append(lista_de_palabras + lista_de_palabras_2)
    #agregado para features extras a las que teniamos
    extra_features.append([int(row[POSITION_HELPFULNESS_NUMERATOR]),int(row[POSITION_HELPFULNESS_DENOMINATOR])])
    clase.append(float(row[6]))


train_data= tf_idf(diccionario,informacion,extra_features)

csv_file.close()
del train_file

#Leo archivo de test
print ("Leyendo archivo de test...")
csv_file = open('../../data/test.csv',encoding='utf8')
data_file = csv.reader(csv_file)
rows = next(data_file)

test_data = []
idd = []
extra_features = []

for row in data_file:
    lista_de_palabras = purificar_texto(word_tokenize(row[7]),stop)
    lista_de_palabras_2 = purificar_texto(word_tokenize(row[8]),stop)
    test_data.append(lista_de_palabras + lista_de_palabras_2)
    #agregado para features extras a las que teniamos
    extra_features.append([int(row[POSITION_HELPFULNESS_NUMERATOR]),int(row[POSITION_HELPFULNESS_DENOMINATOR])])
    idd.append(row[0])
    
test_data = tf_idf(diccionario,test_data,extra_features)
print("OK\n")
csv_file.close()
del data_file

train_data = csr_matrix(train_data)
test_data = csr_matrix(test_data)

print ("Inicio de xgboost..")
dtrain = xgb.DMatrix( train_data, label=clase)
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
num_round = 200

print ("Comienzo a generar el modelo...")
bst = xgb.train(param, dtrain, num_round)
print("OK\n")

print ("Predicciones...")
preds = bst.predict(dtest, ntree_limit=num_round)
print("OK\n")

print ("Creando submission...")
csvsalida = open('../../data/subm-xgboost-description_summary_helpfulness-200iter.csv','w')
salida = csv.writer(csvsalida)
salida.writerow(['Id','Prediction'])

i = 0
for i in range(len(idd)):
    salida.writerow ([idd[i], preds[i]])
csvsalida.close()
print("OK\n")