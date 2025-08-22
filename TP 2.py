#%% ------------------------------------------------------------------------------------------
# Importamos librerias
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import tree
import pandas as pd
from itertools import combinations, islice
import random

#%% ------------------------------------------------------------------------------------------
# Importamos el dataset
carpeta = "C:/Users/franc/Documents/Facultad/Laboratorio de datos/TP - 2/"
numeros = pd.read_csv(carpeta+'TMNIST_Data.csv')
#%% ------------------------------------------------------------------------------------------
# Analisis exploratorio
numeros.info()
# No hay nulls
numeros.dropna
#%% ------------------------------------------------------------------------------------------
pixel = numeros.iloc[:, 2:]
# Plotear la imagen del numero 
img = np.array(pixel.iloc[1]).reshape((28,28)) 
plt.imshow(img, cmap='gray') 
plt.show() 
# Tenemos 29900 filas con imagenes de numeros, 
# 784 conforman la parte que corresponde a los colores de la imagen, 
# la columna labels que indica de que numero se trata,
# Los atributos mas relevantes son los que generan la imagen, labels podria ser usado para para clasificar las filas por numerom, el nombre de
# la fuente podria ser descartada, y quiza algunos valores que conforman el fondo de la imagen, donde el numero no esta presente

#%% ------------------------------------------------------------------------------------------
# 1a)

# Promedio de cada pixel
promedio_de_cada_pixel = pixel.mean() 
pixel = numeros.iloc[:, 2:]
# Plotear la imagen del numero 
img = np.array(promedio_de_cada_pixel).reshape((28,28)) 
plt.imshow(img, cmap='gray') 
plt.title('Intensidad promedio de los pixeles en todas las imagenes')
plt.show() 

# Desviacion estandar de los pixeles
desviaciones = pixel.std()
pixel = numeros.iloc[:, 2:]
img = np.array(desviaciones).reshape((28,28)) 
plt.imshow(img, cmap='gray') 
plt.title('Desviacion estandar de la intensidad de los pixeles en todas las imagenes')
plt.show() 

# varianza de cada pixel
varianzas = pixel.var()
pixel = numeros.iloc[:, 2:]
img = np.array(varianzas).reshape((28,28)) 
plt.imshow(img, cmap='gray') 
plt.title('Varianza de la intensidad de los pixeles de todas las imagenes')
plt.show() 


#%% ------------------------------------------------------------------------------------------
# Recortar cada imagen
def recortar_imagen(fila):
    img = np.array(fila[2:]).reshape((28,28))  
    return img[2:-2, 2:-2].flatten()  

# Aplicar la funcion a cada fila
img_recortadas = numeros.apply(recortar_imagen, axis=1)
#  nuevo DataFrame
img_recortadas = pd.DataFrame(img_recortadas.tolist())
# Agregar labels
img_recortadas['labels'] = numeros['labels']

# Imagen recortada 
img_recortada = np.array(img_recortadas.iloc[0, 1:]).reshape((24,24))  
plt.imshow(img_recortada, cmap='gray')
plt.show()

# Promedio de cada pixel
promedio_de_cada_pixel = img_recortadas.iloc[:, :-1].mean()  
img = np.array(promedio_de_cada_pixel).reshape((24, 24))  
plt.imshow(img, cmap='gray')
plt.title('Intensidad promedio de los pixeles en todas las imagenes')
plt.show()

# Desviacion estandar de los pixeles
desviaciones = img_recortadas.iloc[:, :-1].std()  
img = np.array(desviaciones).reshape((24, 24))  
plt.imshow(img, cmap='gray')
plt.title('Desviación estándar de la intensidad de los pixeles en todas las imagenes')
plt.show()

# Varianza de cada pixel
varianzas = img_recortadas.iloc[:, :-1].var()  
img = np.array(varianzas).reshape((24, 24))  
plt.imshow(img, cmap='gray')
plt.title('Varianza de la intensidad de los pixeles de todas las imagenes')
plt.show()
#%% ------------------------------------------------------------------------------------------
# 1b)

# Filtro las imágenes de los dígitos 1 y 3
uno_y_tres = img_recortadas[(img_recortadas['labels'] == 1 ) | (img_recortadas['labels'] == 3)].iloc[:, 1:]  

# Promedio de cada pixel
promedio_de_cada_pixel = uno_y_tres.mean()  
img = np.array(promedio_de_cada_pixel).reshape((24, 24))  
plt.imshow(img, cmap='gray')
plt.title('Intensidad promedio de los pixeles de las imagenes de 1 y 3')
plt.show()

# Desviacion estandar de los pixeles
desviaciones = uno_y_tres.std()  
img = np.array(desviaciones).reshape((24, 24))  
plt.imshow(img, cmap='gray')
plt.title('Desviación estándar de la intensidad de los pixeles de las imagenes de 1 y 3')
plt.show()

# Varianza de cada pixel
varianzas = uno_y_tres.var()  
img = np.array(varianzas).reshape((24, 24))  
plt.imshow(img, cmap='gray')
plt.title('Varianza de la intensidad de los pixeles de las imagenes de 1 y 3')
plt.show()

# Filtro las imágenes de los dígitos 3 y 8
tres_y_ocho = img_recortadas[(img_recortadas['labels'] == 3 ) | (img_recortadas['labels'] == 8)].iloc[:, 1:]  

# Promedio de cada pixel
promedio_de_cada_pixel = tres_y_ocho.mean()  
img = np.array(promedio_de_cada_pixel).reshape((24, 24))  
plt.imshow(img, cmap='gray')
plt.title('Intensidad promedio de los pixeles de las imagenes de 3 y 8')
plt.show()

# Desviacion estandar de los pixeles
desviaciones = tres_y_ocho.std()  
img = np.array(desviaciones).reshape((24, 24))  
plt.imshow(img, cmap='gray')
plt.title('Desviación estándar de la intensidad de los pixeles de las imagenes de 3 y 8')
plt.show()

# Varianza de cada pixel
varianzas = tres_y_ocho.var()  
img = np.array(varianzas).reshape((24, 24))  
plt.imshow(img, cmap='gray')
plt.title('Varianza de la intensidad de los pixeles de las imagenes de 3 y 8')
plt.show()
#%% ------------------------------------------------------------------------------------------
# 1c)
cero = img_recortadas[img_recortadas['labels'] == 0].iloc[:, 1:]  

# Promedio de cada pixel
promedio_de_cada_pixel = cero.mean()  
img = np.array(promedio_de_cada_pixel).reshape((24, 24))  
plt.imshow(img, cmap='gray')
plt.title('Intensidad promedio de los pixeles del numero cero')
plt.show()

# Desviacion estandar de los pixeles
desviaciones = cero.std()  
img = np.array(desviaciones).reshape((24, 24))  
plt.imshow(img, cmap='gray')
plt.title('Desviación estándar de la intensidad de los pixeles del numero cero')
plt.show()

# Varianza de cada pixel
varianzas = cero.var()  
img = np.array(varianzas).reshape((24, 24))  
plt.imshow(img, cmap='gray')
plt.title('Varianza de la intensidad de los pixeles del numero cero')
plt.show()
#%% ------------------------------------------------------------------------------------------
# 2a)
# Dataframe con las imagenes correspondientes a ceros y unos
cero_o_uno = img_recortadas[(img_recortadas['labels'] == 0 ) | (img_recortadas['labels'] == 1)]
cero_o_uno.info()
# Hay 5980 imagenes, de las cuales la mitad son ceros y la otra mitad uno, asi que podriamos decir que esta balanceado

#%% ------------------------------------------------------------------------------------------
# 2b)
# Separo el dataframe en datos de train y test
X = cero_o_uno.drop(columns=['labels'])  
y = cero_o_uno['labels']  
# Dividir en conjuntos, x intensidades, y etiquetas, 80%  para train
# 20% para para test, respetando la proprcion de mitad y mitad
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

#%% ------------------------------------------------------------------------------------------
# 2c)
# Función para graficar matrices de confusión con exactitud
def graficar_matriz_confusion(cm, exactitud, grupo, index):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=range(2), yticklabels=range(2))
    plt.title(f'Matriz de confusión {grupo}\nExactitud: {exactitud:.4f}')
    plt.xlabel('Predicción')
    plt.ylabel('Valor real')
    plt.tight_layout()
    plt.show()

# Almacenar resultados
exactitudes = []
grupos = []
matrices_confusion = []

# Registrar resultados y matrices de confusión
def registrar_resultados(grupo, exactitud, cm):
    grupos.append(grupo)
    exactitudes.append(exactitud)
    matrices_confusion.append(cm)

# Entrenamiento con tres atributos
conjuntos_tres_atributos = [
    [156, 438, 566],
    [141, 300, 500],
    [90, 516, 460]
]
for grupo in conjuntos_tres_atributos:
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train[grupo], y_train)
    Y_pred = model.predict(X_test[grupo])
    acc = metrics.accuracy_score(y_test, Y_pred)
    cm = confusion_matrix(y_test, Y_pred, labels=range(2))  
    registrar_resultados(f"{grupo}", acc, cm)

# Entrenamiento con dos atributos
conjuntos_dos_atributos = [
    [100, 230],
    [134, 75],
    [457, 243]
]
for grupo in conjuntos_dos_atributos:
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train[grupo], y_train)
    Y_pred = model.predict(X_test[grupo])
    acc = metrics.accuracy_score(y_test, Y_pred)
    cm = confusion_matrix(y_test, Y_pred, labels=range(2))  
    registrar_resultados(f"{grupo}", acc, cm)

# Entrenamiento con cuatro atributos
conjuntos_cuatro_atributos = [
    [100, 230, 22, 357],
    [547, 354, 46, 352],
    [257, 54, 546, 435]
]
for grupo in conjuntos_cuatro_atributos:
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train[grupo], y_train)
    Y_pred = model.predict(X_test[grupo])
    acc = metrics.accuracy_score(y_test, Y_pred)
    cm = confusion_matrix(y_test, Y_pred, labels=range(2))  
    registrar_resultados(f"{grupo}", acc, cm)

# Graficar exactitudes
plt.figure(figsize=(10, 6))
x = range(len(grupos))
colores = ['lightblue','lightblue','lightblue','deepskyblue','deepskyblue','deepskyblue','blue','blue','blue'] # colores para el gráfico de barras
barras = plt.bar(x, exactitudes, color=colores, edgecolor='black')
# Agrego los valores encima de cada barra
for bar in barras:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval, f'{round(yval,2)}', ha='center', va='bottom', fontsize=10)
plt.xticks(x, grupos, rotation=45, ha='right')
plt.xlabel('Grupos de atributos')
plt.ylabel('Exactitud')
plt.legend()
plt.tight_layout()
plt.show()

# Graficar las matrices de confusión con la exactitud en cada una
for index, cm in enumerate(matrices_confusion):
    graficar_matriz_confusion(cm, exactitudes[index], grupos[index], index)

#%% ------------------------------------------------------------------------------------------
# 2d)
Nrep = 1
valores_k = [3, 5, 7, 10]
valores_n = [3, 10, 50, 100]

# Inicialización de matrices para almacenar resultados de test y entrenamiento
resultados_test = np.zeros((len(valores_n), len(valores_k)))
resultados_train = np.zeros((len(valores_n), len(valores_k)))

# graficar la matriz de confusión
def graficar_matriz_confusion(cm, exactitud, n_atributos, k, rep):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(2), yticklabels=range(2))
    plt.title(f'Matriz de confusión para n_atributos={n_atributos}, k={k}, repetición {rep+1}\nExactitud: {exactitud:.4f}')
    plt.xlabel('Predicciones')
    plt.ylabel('Valor real')
    plt.tight_layout()
    plt.show()

# entrenar y evaluar el modelo
for i, n_atributos in enumerate(valores_n):
    for rep in range(Nrep):
        # Selección de atributos
        columnas_aleatorias = X.sample(n=n_atributos, axis=1, random_state=rep)
        X_selected = columnas_aleatorias.values
        X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=rep)
        
        for j, k in enumerate(valores_k):
            # Creación y ajuste del modelo KNN con k vecinos
            model = KNeighborsClassifier(n_neighbors=k)
            model.fit(X_train, y_train)
            Y_pred = model.predict(X_test)
            Y_pred_train = model.predict(X_train)
            
            # Exactitud
            acc_test = metrics.accuracy_score(y_test, Y_pred)
            acc_train = metrics.accuracy_score(y_train, Y_pred_train)
            
            # Guardar los resultados de exactitud
            resultados_test[i, j] += acc_test / Nrep  
            resultados_train[i, j] += acc_train / Nrep  
            
            # Matriz de confusión para el conjunto de prueba
            conf_matrix_test = confusion_matrix(y_test, Y_pred)
            
            # Graficar la matriz de confusión con la exactitud en el título
            graficar_matriz_confusion(conf_matrix_test, acc_test, n_atributos, k, rep)

# Graficar promedios de exactitud para cada cantidad de atributos
for i, n_atributos in enumerate(valores_n):
    plt.plot(valores_k, resultados_train[i, :], label=f'Train (n={n_atributos})')
    plt.plot(valores_k, resultados_test[i, :], label=f'Test (n={n_atributos})')

plt.legend()
plt.title('Exactitud del modelo de KNN')
plt.xlabel('Cantidad de vecinos (k)')
plt.ylabel('Exactitud (accuracy)')
plt.savefig('exactitud_knn.png')
plt.show()

#%% ------------------------------------------------------------------------------------------
# 3a)

X = img_recortadas.drop(columns=['labels'])  
y = img_recortadas['labels']  

# Separamos en desarrollo y validacion 
X_dev, X_eval, y_dev, y_eval = train_test_split(X,y,random_state=1,test_size=0.1)

#%% ------------------------------------------------------------------------------------------
# 3b)

# Definir las profundidades del árbol y el número de splits
alturas = [1, 2, 3, 5, 10]
nsplits = 5
kf = KFold(n_splits=nsplits)

# lista para almacenar las matrices de confusión por cada profundidad
matrices_confusion_por_altura = {hmax: np.zeros((10, 10), dtype=int) for hmax in alturas}

# Matriz de resultados
resultados = np.zeros((nsplits, len(alturas)))

for i, (train_index, test_index) in enumerate(kf.split(X_dev)):
    kf_X_train, kf_X_test = X_dev.iloc[train_index], X_dev.iloc[test_index]
    kf_y_train, kf_y_test = y_dev.iloc[train_index], y_dev.iloc[test_index]
    
    for j, hmax in enumerate(alturas):
        # Entrenar el modelo de árbol de decisión con la profundidad maxima especificada
        arbol = DecisionTreeClassifier(max_depth=hmax)
        arbol.fit(kf_X_train, kf_y_train)
        pred = arbol.predict(kf_X_test)
        
        # calculo la matriz de confusión para este pliegue y profundidad
        cm = confusion_matrix(kf_y_test, pred, labels=range(10))
        
        # matriz para cada profundidad
        matrices_confusion_por_altura[hmax] += cm
        
        # calculo la precisión 
        score = np.trace(cm) / np.sum(cm)
        
        # Guardo el resultado de exactitud
        resultados[i, j] = score

# Matrices de confusión para cada profundidad
for hmax in alturas:
    print(f"\nMatriz de Confusión Acumulada para max_depth={hmax}:")
    print(matrices_confusion_por_altura[hmax])

# Resultados de exactitud promedio para cada profundidad
exactitudes_promedio = []
for j, hmax in enumerate(alturas):
    promedio = np.mean(resultados[:, j])
    exactitudes_promedio.append(promedio)
    print(f"Exactitud promedio para max_depth={hmax}: {promedio:.4f}")

# Graficar exactitud promedio en función de la profundidad del árbol
plt.figure(figsize=(8, 6))
plt.plot(alturas, exactitudes_promedio, marker='o', linestyle='-', color='k', label='Exactitud promedio')
plt.title('Exactitud promedio en función de la profundidad del árbol')
plt.xlabel('Profundidad máxima del árbol')
plt.ylabel('Exactitud promedio')
plt.xticks(alturas)  
plt.grid(alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()
#%% ------------------------------------------------------------------------------------------
# 3c)

alturas = [7, 8, 9, 10]
# Hiperparametros 
criterios = ['entropy', 'gini']  
nsplits = 5
kf = KFold(n_splits=nsplits)

# Lista para almacenar las matrices de confusion por cada combinación de profundidad y criterio
matrices_confusion_por_criterio_y_altura = {
    (hmax, crit): np.zeros((10, 10), dtype=int) 
    for hmax in alturas for crit in criterios
}

# Matriz de resultados 
resultados = np.zeros((nsplits, len(alturas), len(criterios)))

for i, (train_index, test_index) in enumerate(kf.split(X_dev)):
    kf_X_train, kf_X_test = X_dev.iloc[train_index], X_dev.iloc[test_index]
    kf_y_train, kf_y_test = y_dev.iloc[train_index], y_dev.iloc[test_index]
    
    for j, hmax in enumerate(alturas):
        for k, crit in enumerate(criterios):
            arbol = DecisionTreeClassifier(max_depth=hmax, criterion=crit)
            arbol.fit(kf_X_train, kf_y_train)
            pred = arbol.predict(kf_X_test)
            cm = confusion_matrix(kf_y_test, pred, labels=range(10))
            matrices_confusion_por_criterio_y_altura[(hmax, crit)] += cm
            score = np.trace(cm) / np.sum(cm)

            resultados[i, j, k] = score

# Resultados de exactitud promedio para cada profundidad y criterio
exactitudes_promedio_por_criterio = {crit: [] for crit in criterios}

for k, crit in enumerate(criterios):
    for j, hmax in enumerate(alturas):
        promedio = np.mean(resultados[:, j, k])
        exactitudes_promedio_por_criterio[crit].append(promedio)
        print(f"Exactitud promedio para max_depth={hmax}, criterion={crit}: {promedio:.4f}")

# Graficar exactitud promedio en función de la profundidad para ambos criterios
plt.figure(figsize=(8, 6))
for crit, color in zip(criterios, ['b', 'r']):  
    plt.plot(alturas, exactitudes_promedio_por_criterio[crit], marker='o', linestyle='-', color=color, label=f'Exactitud promedio ({crit})')

plt.title('Exactitud promedio en función de la profundidad')
plt.xlabel('Profundidad máxima del árbol')
plt.ylabel('Exactitud promedio')
plt.xticks(alturas)  
plt.grid(alpha=0.6)
plt.legend() 
plt.tight_layout()
plt.show()

#%% ------------------------------------------------------------------------------------------
# 3d) 

mejor_max_depth = 10
mejor_criterio = 'entropy'

modelo_final = DecisionTreeClassifier(max_depth=mejor_max_depth, criterion=mejor_criterio)
modelo_final.fit(X_dev, y_dev)
predicciones = modelo_final.predict(X_eval)
exactitud = accuracy_score(y_eval, predicciones)
matriz_confusion = confusion_matrix(y_eval, predicciones, labels=range(10))

# Mostrar los resultados
print(f"Exactitud en el conjunto held-out: {exactitud:.4f}")
print("Matriz de Confusión en el conjunto held-out:")
print(matriz_confusion)

# Graficar matriz de confusion
plt.figure(figsize=(8, 6))
sns.heatmap(matriz_confusion, annot=True, fmt='d', cmap='Blues', cbar=True, xticklabels=range(10), yticklabels=range(10))
plt.title('Matriz de confusion (Conjunto Held-Out)')
plt.xlabel('Predicciones')
plt.ylabel('Valor real')
plt.tight_layout()
plt.show()
