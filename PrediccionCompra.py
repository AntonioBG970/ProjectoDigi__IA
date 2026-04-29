# Librerias importadas
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Creacion de datos
#[edad, ingresos, visitas web, compras previas]
X = np.array([
    [18, 18000, 1, 0],
    [20, 20000, 2, 0],
    [22, 22000, 3, 0],
    [24, 25000, 3, 0],
    [26, 27000, 4, 1],
    [28, 30000, 5, 1],
    [30, 35000, 6, 1],
    [32, 40000, 7, 2],
    [34, 45000, 8, 2],
    [36, 50000, 9, 2],
    [38, 60000, 10, 3],
    [40, 65000, 11, 3],
    [42, 70000, 12, 3],
    [44, 75000, 13, 4],
    [46, 80000, 14, 4],
    [48, 85000, 15, 4],
    [50, 90000, 16, 5],
    [29, 32000, 4, 0],
    [27, 28000, 3, 0],
    [33, 42000, 6, 1],
    [37, 55000, 9, 2],
    [41, 62000, 10, 3],
    [23, 21000, 2, 0],
    [31, 38000, 5, 1],
    [45, 78000, 13, 4]
])

#0 = No Compra | 1 = Compra
y = np.array([
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1
])

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
print("----- Resultados del Modelo -----")
print(f"Precisión del modelo: {accuracy:.2f}")

print("\nInforme de la clasificación:")
print(classification_report(y_test, y_pred))

# Probar con nuevos clientes
clientes = {
    "Cliente A": [[35, 65000, 10, 3]],
    "Cliente B": [[23, 20000, 1, 0]],
    "Cliente C": [[42, 85000, 14, 5]],
    "Cliente D": [[28, 30000, 2, 0]]
}

print("Predicciones Individuales")

for nombre, datos in clientes.items():
    prediccion = model.predict(datos)[0]
    resultado = "COMPRA" if prediccion == 1 else "NO COMPRA"
    print(f"{nombre}: {resultado}")
