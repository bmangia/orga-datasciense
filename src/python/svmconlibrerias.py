from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.corpus import stopwords

import csv

def seteando_stopwords():
	negadores = set(('but','no','nor','not','very','don','should'))
	puntuacion  = set(('.',',',':',';','(',')','[',']','{','}','<','>','!','?','/','...','""',"''",'--','-'))
	stop = set(stopwords.words('english')) - negadores
	stop_completo = set()
	stop_completo.update(stop)
	stop_completo.update(puntuacion)
	
	return stop_completo

archivo = csv.reader(open('train.csv',encoding='utf8'))
stop = seteando_stopwords()


filas = next(archivo)
datos = []
clase = []
diccionario = {}
i = 0
informacion =[]

for filas in archivo:
    informacion.append(filas[8])
    #informacion.append(filas[9])
    clase.append(float(filas[6]))

datos = TfidfVectorizer(input=u'content',stop_words = stop, ngram_range=(1, 3)).fit_transform(informacion)

print ("empece a entrenar")
svc = svm.LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
     intercept_scaling=1, loss='squared_hinge', max_iter=1000,
     multi_class='ovr', penalty='l2', tol=0.0001,
     verbose=0).fit(datos, clase)

print("termine de entrenar")

archivo2 = csv.reader(open('test.csv'))
csvsalida = open('submr1.csv','w')
salida = csv.writer(csvsalida)
salida.writerow(['Id','Prediction'])
filas2 = next(archivo2)

informacion2 = []
idd = []


for filas in archivo2:
	informacion2.append(filas[7])
	#informacion2.append(filas[8])
	idd.append(filas[0])
	

target = TfidfVectorizer(input=u'content',stop_words = stop, ngram_range=(1, 3)).fit_transform(informacion)

print("empece a predecir")
lista = svc.predict(target)
print("termine de predecir")
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
