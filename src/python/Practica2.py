from sklearn import svm
from scipy.sparse import *
import numpy as np
import math

import csv

def generando_diccionario(diccionario,texto,indice):
    diccionario_aux ={}
    for palabra in texto:
        if palabra in diccionario and not( palabra in diccionario_aux):
            diccionario[palabra][1] = diccionario[palabra][1] + 1

        elif not(palabra in diccionario):
            diccionario[palabra] = [indice,1]
            indice += 1

        diccionario_aux[palabra] = 1


def tf_idf(diccionario,informacion):

    matriz_resumen = lil_matrix((len(informacion),len(diccionario)))
   # matriz_descripcion = lil_matrix((len(informacion),len(diccionario)))

  #  i = 0
 #   for tupla in informacion:
 #       for palabra in tupla[1]:
         	
 #           matriz_descripcion[i,diccionario[palabra][0]] = matriz_descripcion[i,diccionario[palabra][0]] + 1/len(tupla[1]) * math.log(len(informacion) /diccionario[palabra][1],2)
 #       i = i + 1
    i = 0


    for tupla in informacion:
        for palabra in tupla[0]:
            if palabra in diccionario:	
                matriz_resumen[i,diccionario[palabra][0]] = matriz_resumen[i,diccionario[palabra][0]] + 1/len(tupla[0]) * math.log(len(informacion) /diccionario[palabra][1],2)

        i = i + 1

    return matriz_resumen
archivo = csv.reader(open('train2.csv',encoding='utf8'))

filas = next(archivo)
datos = []
clase = []
diccionario = {}
i = 0
informacion =[]

for filas in archivo:
    generando_diccionario(diccionario,filas[8].split(),i)
    generando_diccionario(diccionario,filas[9].split(),i)
    informacion.append([filas[8].split(),filas[9].split()])
    clase.append(float(filas[6]))

datos= tf_idf(diccionario,informacion) 	


print ("empece a entrenar")
svc = svm.LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
     intercept_scaling=1, loss='squared_hinge', max_iter=100000,
     multi_class='ovr', penalty='l2', random_state=7, tol=0.0001,
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
	informacion2.append([filas[7].split(),filas[8].split()])
	idd.append(filas[0])
	

target = tf_idf(diccionario,informacion2)

print("empece a predecir")
lista = svc.predict(target)
print("termine de predecir")
print (lista)
"""
i = 0
for i in range(len(idd)):
	salida.writerow (idd[i], svc.predict(target))
	
del archivo
del archivo2
del salida
csvsalida.close()
"""