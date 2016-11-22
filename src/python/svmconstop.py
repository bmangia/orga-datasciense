from sklearn import svm
from scipy.sparse import *
from nltk import word_tokenize
from nltk.corpus import stopwords
import math

import csv

def seteando_stopwords():
	negadores = set(('but','no','nor','not','very','t','don'))
	puntuacion  = set(('.',',',':',';','(',')','[',']','{','}','<','>'))
	stop = set(stopwords.words('english')) - negadores
	stop_completo = set()
	for palabra in stop:
		stop_completo.update(set(palabra.upper()))

	stop_completo.update(stop)
	stop_completo.update(puntuacion)
	
	return stop_completo


def generando_diccionario(diccionario,texto,indice,stopwords):
    diccionario_aux ={}
    for palabra in texto:
        if not palabra  in stopwords:
            if palabra in diccionario and not( palabra in diccionario_aux):
                diccionario[palabra][1] = diccionario[palabra][1] + 1

            elif not(palabra in diccionario):
                diccionario[palabra] = [indice,1]
                indice += 1

            diccionario_aux[palabra] = 1

        else:
            texto.remove(palabra)

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


archivo = csv.reader(open('train3.csv',encoding='utf8'))
stop = seteando_stopwords()


filas = next(archivo)
datos = []
clase = []
diccionario = {}
i = 0
informacion =[]

for filas in archivo:
    #i = generando_diccionario(diccionario,word_tokenize(filas[8]),i,stop)
    #informacion.append(word_tokenize(filas[8]))
    i = generando_diccionario(diccionario,word_tokenize(filas[9]),i,stop)
    informacion.append(word_tokenize(filas[9]))
    clase.append(float(filas[6]))

datos= tf_idf(diccionario,informacion) 	

print ("empece a entrenar")
svc = svm.LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
     intercept_scaling=1, loss='squared_hinge', max_iter=1000,
     multi_class='ovr', penalty='l2', random_state=11, tol=0.0001,
     verbose=0).fit(datos, clase)

print("termine de entrenar")

archivo2 = csv.reader(open('test2.csv'))
csvsalida = open('subm.csv','w')
salida = csv.writer(csvsalida)
salida.writerow(['Id','Prediction'])
filas2 = next(archivo2)

informacion2 = []
idd = []


for filas in archivo2:
	#informacion2.append(word_tokenize(filas[7]))
	informacion2.append(word_tokenize(filas[8]))
	idd.append(filas[0])
	

target = tf_idf(diccionario,informacion2)

print("empece a predecir")
lista = svc.predict(target)
print("termine de predecir")
print (lista)
x = 0
for numero in lista:
	x = x + numero
print(x/len(lista))
i = 0
for i in range(len(idd)):
	salida.writerow ([idd[i], lista[i]])
	
del archivo
del archivo2
del salida
csvsalida.close()
