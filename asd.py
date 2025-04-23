import matplotlib.pyplot as plt
import numpy as np

# Datos
tareas = [
    "Conocimiento del negocio", "Conocimiento de tecnologías", "BD de entrenamiento",
    "Pipelines de NLP", "Integración BD", "Testeo chatbot",
    "Ajuste de modelos", "Ventana de chat personalizada"
]

horas_estimadas = [16, 24, 24, 30, 24, 30, 25, 50]
horas_utilizadas = [24, 20, 30, 40, 30, 40, 35, 0]  # La ventana de chat no tiene datos

x = np.arange(len(tareas))  # Posiciones en el eje X

# Crear la figura y el gráfico de barras
plt.figure(figsize=(12, 6))
plt.bar(x - 0.2, horas_estimadas, width=0.4, label="Horas Estimadas", color="royalblue")
plt.bar(x + 0.2, horas_utilizadas, width=0.4, label="Horas Utilizadas", color="orange")

# Etiquetas y formato
plt.xlabel("Tareas")
plt.ylabel("Horas")
plt.title("Comparación de Tiempos Estimados vs. Utilizados")
plt.xticks(ticks=x, labels=tareas, rotation=45, ha="right")
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Ajustar el diseño y mostrar el gráfico
plt.tight_layout()
plt.show()
