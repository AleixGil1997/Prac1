from sklearn.datasets import make_regression
import numpy as np
import pandas as pd
%matplotlib notebook
from matplotlib import pyplot as plt
import scipy.stats
import seaborn as sns
from scipy.stats import normaltest
import math
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Visualitzarem només 3 decimals per mostra
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Funcio per a llegir dades en format csv
def load_dataset(path):
    dataset = pd.read_csv(path, header=0, delimiter=',')
    return dataset

# Carreguem dataset d'exemple
dataset = load_dataset('forestfires.csv')
data = dataset.values

x = data[:, 4:]
y = data[:, 12]

xx = []
yy = []

noZero = y.nonzero()

for i in range(len(x)):
    if y[i] < 500:
        xx.append(x[i])
        yy.append(y[i])
    
xx = np.array(xx)
yy = np.array(yy)
x = xx
y = yy

print("Dimensionalitat de la BBDD:", dataset.shape)
print("Dimensionalitat de les entrades X", x.shape)
print("Dimensionalitat de l'atribut Y", y.shape)


print("Per comptar el nombre de valors no existents:")
print(dataset.isnull().sum())


print("Per veure estadístiques dels atributs numèrics de la BBDD:")
dataset.describe()


plt.figure()
plt.title("FFMC")
ax = plt.scatter(x[:,0], y)

plt.figure()
plt.title("DMC")
ax = plt.scatter(x[:,1], y)

plt.figure()
plt.title("DC")
ax = plt.scatter(x[:,2], y)

plt.figure()
plt.title("ISI")
ax = plt.scatter(x[:,3], y)

plt.figure()
plt.title("Temperatura")
ax = plt.scatter(x[:,4], y)

plt.figure()
plt.title("Humitat")
ax = plt.scatter(x[:,5], y)

plt.figure()
plt.title("Vent")
ax = plt.scatter(x[:,6], y)

plt.figure()
plt.title("Pluja")
ax = plt.scatter(x[:,7], y)


plt.figure()
plt.title("Histograma de l'atribut 0")
plt.xlabel("Attribute Value")
plt.ylabel("Count")
hist = plt.hist(x[:,0], bins=11, range=[np.min(x[:,0]), np.max(x[:,0])], histtype="bar", rwidth=0.8)


# Mirem la correlació entre els atributs d'entrada per entendre millor les dades
correlacio = dataset.corr()

plt.figure()

ax = sns.heatmap(correlacio, annot=True, linewidths=.5)


# Mirem la relació entre atributs utilitzant la funció pairplot
relacio = sns.pairplot(dataset)


atributs=['FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain', 'area']
  
for i in range(9):
    plt.figure()
    plt.title(atributs[i])
    ax=sns.distplot(x[:,i])
    stat, p = normaltest(x[:,i])
    print('stat=%.2f, p=%.5f\n' % (stat, p))
    if p > 0.05:
        print('Probably Gaussian\n\n')
    else:
        print('Probably not Gaussian\n\n')
        

def mean_squeared_error(y1, y2):
    # comprovem que y1 i y2 tenen la mateixa mida
    assert(len(y1) == len(y2))
    mse = 0
    for i in range(len(y1)):
        mse += (y1[i] - y2[i])**2
    return mse / len(y1)

mean_squeared_error([1,2,3,4], [1,2,1,4])


np.warnings.filterwarnings('ignore')

vector1 = np.array([1,2,3,4]) # convertim llista de python a numpy array
vector2 = np.array([1,2,1,4]) 

# podem sumar dos vectors element a element
print("Suma vector1 + vector2 ", vector1 + vector2)

# podem sumar tots els valors d'un vector
print("Suma valors vector1 ", vector1.sum())

# calculem la mitjana
print("Mitjana vector1", vector1.mean())

# utilitzem un vector com a índex de l'altre
# vector3 = vector1  # necesitem fer una copia del vector per no modificar el original
vector3 = vector1.copy()
vector3[vector2 == 1] = 5
print("Vector1 amb un 5 on el Vector2 te 1s ", vector3)

# es pot utilitzar numpy per a calcular el mse
def mse(v1, v2):
    return ((v1 - v2)**2).mean()

print("MSE: ", mse(vector1, vector2))


def regression(x, y):
    # Creem un objecte de regressió de sklearn
    regr = LinearRegression()

    # Entrenem el model per a predir y a partir de x
    regr.fit(x, y)

    # Retornem el model entrenat
    return regr


def standarize(x_train):
    mean = x_train.mean(0)
    std = x_train.std(0)
    x_t = x_train - mean[None, :]
    x_t /= std[None, :]
    return x_t

x_t = standarize(x.astype('int'))


plt.figure()
plt.title("Temperatura")
plt.xlabel("Attribute Value")
plt.ylabel("Count")
hist = plt.hist(x_t[:,4], bins=11, range=[np.min(x_t[:,4]), np.max(x_t[:,4])], histtype="bar", rwidth=0.8)


plt.figure()
plt.title("Temperatura")
plt.xlabel("Attribute Value")
plt.ylabel("Count")
hist = plt.hist(x[:,4], bins=11, range=[np.min(x[:,4]), np.max(x[:,4])], histtype="bar", rwidth=0.8)


for i in range(len(x[0])):
    # Extraiem el primer atribut de x i canviem la mida a #exemples, #dimensions de l'atribut.
    # En el vostre cas, haureu de triar un atribut com a y, i utilitzar la resta com a x.
    atribut1 = x[:,i].reshape(x.shape[0], 1)
    regr = regression(atribut1, y)
    predicted = regr.predict(atribut1)

    # Mostrem la predicció del model entrenat en color vermell a la Figura anterior 1
    plt.figure()
    plt.title(atributs[i])
    ax = plt.scatter(x[:,i], y)
    plt.plot(atribut1[:,0], predicted, 'r')

    # Mostrem l'error (MSE i R2)
    MSE = mse(y, predicted)
    r2 = r2_score(y, predicted)

    print("Mean squeared error: ", MSE)
    print("R2 score: ", r2)


""" Per a assegurar-nos que el model s'ajusta be a dades noves, no vistes, 
cal evaluar-lo en un conjunt de validacio (i un altre de test en situacions reals).
Com que en aquest cas no en tenim, el generarem separant les dades en 
un 80% d'entrenament i un 20% de validació.
"""
def split_data(x, y, train_ratio=0.8):
    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)
    n_train = int(np.floor(x.shape[0]*train_ratio))
    indices_train = indices[:n_train]
    indices_val = indices[n_train:] 
    x_train = x[indices_train, :]
    y_train = y[indices_train]
    x_val = x[indices_val, :]
    y_val = y[indices_val]
    return x_train, y_train, x_val, y_val

# Dividim dades d'entrenament
x_train, y_train, x_val, y_val = split_data(x, y)

for i in range(x_train.shape[1]):
    x_t = x_train[:,i] # seleccionem atribut i en conjunt de train
    x_v = x_val[:,i] # seleccionem atribut i en conjunt de val.
    x_t = np.reshape(x_t,(x_t.shape[0],1))
    x_v = np.reshape(x_v,(x_v.shape[0],1))

    regr = regression(x_t, y_train)    
    error = mse(y_val, regr.predict(x_v)) # calculem error
    r2 = r2_score(y_val, regr.predict(x_v))

    print("Error en atribut %s: %f" %(atributs[i], error))
    print("R2 score en atribut %s: %f\n" %(i, r2))