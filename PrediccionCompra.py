# Librerias importadas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Datos de los clientes
datos = {
    "Edad": [18,20,22,24,26,28,30,32,34,36,
              38,40,42,44,46,48,50,29,27,33,
              37,41,23,31,45],

    "Ingresos": [18000,20000,22000,25000,27000,
                 30000,35000,40000,45000,50000,
                 60000,65000,70000,75000,80000,
                 85000,90000,32000,28000,42000,
                 55000,62000,21000,38000,78000],

    "Visitas_Web": [1,2,3,3,4,5,6,7,8,9,
                    10,11,12,13,14,15,16,4,
                    3,6,9,10,2,5,13],

    "Compras_Previas": [0,0,0,0,1,1,1,2,2,2,
                        3,3,3,4,4,4,5,0,0,1,
                        2,3,0,1,4],

#0 = No Compra | 1 = Compra
    "Compra": [0,0,0,0,0,0,0,0,0,0,
               1,1,1,1,1,1,1,0,0,1,
               1,1,0,1,1]
}

# Creación del DataFrame
df = pd.DataFrame(datos)

# Mostrar dataset
print("===== DATASET DE CLIENTES =====")
print(df)

# Variables de entrada
X = df[["Edad", "Ingresos", "Visitas_Web", "Compras_Previas"]]

# Variable objetivo
y = df["Compra"]

# División de datos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Creación y entrenamiento del modelo
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Realizar predicciones
y_pred = model.predict(X_test)

# Evaluación del modelo
accuracy = accuracy_score(y_test, y_pred)
print("===== Resultados del Modelo =====")
print(f"Precisión del modelo: {accuracy:.2f}")

print("\nInforme de la clasificación:")
print(classification_report(y_test, y_pred))

# Separar clientes que compran y no compran
compran = df[df["Compra"] == 1]
no_compran = df[df["Compra"] == 0]

# Crear ventana del gráfico
plt.figure(figsize=(10,6))

# Clientes que NO compran
plt.scatter(
    no_compran["Ingresos"],
    no_compran["Visitas_Web"],
    label="No Compra"
)

# Clientes que SÍ compran
plt.scatter(
    compran["Ingresos"],
    compran["Visitas_Web"],
    label="Compra"
)

# Configuración del gráfico
plt.xlabel("Ingresos")
plt.ylabel("Visitas Web")
plt.title("Clientes que compran y no compran")
plt.legend()
plt.grid(True)

# Mostrar gráfico
plt.show()

# Probar con nuevos clientes
clientes = {
    "Cliente A": pd.DataFrame(
        [[35, 65000, 10, 3]],
        columns=["Edad", "Ingresos", "Visitas_Web", "Compras_Previas"]
    ),

    "Cliente B": pd.DataFrame(
        [[23, 20000, 1, 0]],
        columns=["Edad", "Ingresos", "Visitas_Web", "Compras_Previas"]
    ),

    "Cliente C": pd.DataFrame(
        [[42, 85000, 14, 5]],
        columns=["Edad", "Ingresos", "Visitas_Web", "Compras_Previas"]
    ),

    "Cliente D": pd.DataFrame(
        [[28, 30000, 2, 0]],
        columns=["Edad", "Ingresos", "Visitas_Web", "Compras_Previas"]
    )
}

print("===== Predicciones Individuales =====")

for nombre, datos in clientes.items():
    prediccion = model.predict(datos)[0]
    resultado = "COMPRA" if prediccion == 1 else "NO COMPRA"
    print(f"{nombre}: {resultado}")
