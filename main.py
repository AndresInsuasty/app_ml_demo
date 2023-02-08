import pandas as pd
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt

load_dotenv()

PATH_DATA= os.getenv("PATH_DATA")

df = pd.read_csv(PATH_DATA)

# Seleccionar solo las columnas numéricas
numeric_cols = df.select_dtypes(include=['int', 'float'])

# Sumar las columnas numéricas
total = numeric_cols.sum()

# Imprimir el resultado
#print(total)

#plt.plot(total)
#plt.show()

import numpy as np

# Asignar años a la variable x
x = total.keys().values
#x = [int(i) for i in x]
x = np.array(x).reshape(-1, 1)
# Asignar valores a la variable y
y = total.values

# Imprimir el resultado
print("x: ", type(x))
print("y: ", type(y))


# train_test_split
from sklearn.model_selection import train_test_split
TEST_SIZE=float(os.getenv("TEST_SIZE"))/100
SEED=int(os.getenv("SEED"))
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=TEST_SIZE, random_state=SEED)

# Entrenar el modelo

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(train_x, train_y)

# Predecir con el modelo
pred_y = model.predict(test_x)

# Calcular el error
from sklearn.metrics import mean_squared_error
error = mean_squared_error(test_y, pred_y)

# Imprimir el error
print("Error: ", error)

# Graficar los datos
#plt.scatter(test_x[:,0], test_y)
#plt.plot(test_x[:,0], pred_y, color='red')
#plt.show()

# Guardar el modelo joblib
import joblib
PATH_MODEL= os.getenv("PATH_MODEL")
joblib.dump(model, PATH_MODEL)


def prueba(a,b):
    return a+b