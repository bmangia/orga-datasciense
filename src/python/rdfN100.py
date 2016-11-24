import pandas as pd
from scipy.sparse import *
from matplotlib import pyplot as plt
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import math
import csv

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


def tf_idf(diccionario,informacion):

    matriz_resumen = lil_matrix((len(informacion),len(diccionario)))
    i = 0


    for doc in informacion:
        for palabra in doc:
            if palabra in diccionario:	
                matriz_resumen[i,diccionario[palabra][0]] = matriz_resumen[i,diccionario[palabra][0]]+ math.log(len(informacion) /diccionario[palabra][1],2)

        i = i + 1

    return matriz_resumen

#####

archivo = csv.reader(open('./data/train.csv',encoding='utf8'))
stop = seteando_stopwords()


filas = next(archivo)
datos = []
clase = []
lista_de_palabras = []
diccionario = {}
i = 0
informacion =[]
print ("OK\n")

for filas in archivo:
    lista_de_palabras_summary = purificar_texto(words_tokenize(filas[8]),stop)
    i = generando_diccionario(diccionario,lista_de_palabras,i)
    lista_de_palabras_text = purificar_texto(word_tokenize(filas[9]),stop)
    i = generando_diccionario(diccionario,lista_de_palabras,i)
    informacion_train.append(lista_de_palabras_text + lista_de_palabras_summary)
    clase.append(float(filas[6]))


datos= tf_idf(diccionario,informacion_train) 	

print ("Entrena el RF")

forest = RandomForestClassifier(n_estimators = 10, max_features="sqrt") #probar con n_estimators=100 si se puede y oob_score=true
forest_description = forest.fit(datos, clase)
print ("OK\n")

archivo2 = csv.reader(open('./data/test.csv'))
csvsalida = open('submRF.csv','w')
salida = csv.writer(csvsalida)
salida.writerow(['Id','Prediction'])
filas2 = next(archivo2)

informacion2 = []
idd = []


for filas in archivo2:
	lista_de_palabras_summary = purificar_texto(word_tokenize(filas[7]),stop)
	lista_de_palabras_text = purificar_texto(word_tokenize(filas[8]),stop)
	informacion_test.append(lista_de_palabras_text + lista_de_palabras_summary)
	idd.append(filas[0])

target = tf_idf(diccionario,informacion_test)

print ("Calculando predicciones...")

result_description = forest.predict(target)

print ("OK\n")

prediccion = 0
for x in lista:
	prediccion = prediccion + x

print (prediccion/len(lista))
i = 0
for i in range(len(idd)):
	salida.writerow ([idd[i], lista[i]])
	
del archivo
del archivo2
del salida
csvsalida.close()
print ("OK\n")
