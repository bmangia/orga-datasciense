import csv

COEF_AR_1 = 0.4
COEF_AR_2 = 0.4
COEF_AR_3 = 0.2

archivo1 = csv.reader(open('subm.csv',encoding='utf8'))
archivo2 = csv.reader(open('submr2.csv',encoding='utf8'))
archivo3 = csv.reader(open('submr3.csv',encoding='utf8'))
csvsalida = open('average.csv','w')


archivo_salida = csv.writer(csvsalida)
archivo_salida.writerow(['Id','Prediction'])

filas1 = next(archivo1)
filas2 = next(archivo2)
filas3 = next(archivo3)

for filas1,filas2,filas3 in zip(archivo1,archivo2,archivo3):
	prediccion = COEF_AR_1 * float(filas1[1]) + COEF_AR_2 * float(filas2[1]) + COEF_AR_3 * float(filas3[1])
	archivo_salida.writerow([filas1[0],prediccion])

del archivo1
del archivo2
del archivo3
del archivo_salida
csvsalida.close()