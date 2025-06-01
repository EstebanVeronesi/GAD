import os
import cv2
import random
import csv
import numpy as np

# Config
CARPETA_ORIGEN = r"C:\faces"
CARPETA_DESTINO = r"Testset"
GROUND_TRUTH_CSV = os.path.join(CARPETA_DESTINO, "ground_truth.csv")
TAMANO = (300, 300)


# Crear carpeta destino
os.makedirs(CARPETA_DESTINO, exist_ok=True)

# Obtener imágenes
imagenes = [f for f in os.listdir(CARPETA_ORIGEN) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
imagenes_seleccionadas = random.sample(imagenes, min(50, len(imagenes)))

def agregar_canal_alpha(imagen_bgr):
    imagen = cv2.resize(imagen_bgr, TAMANO)
    rgba = cv2.cvtColor(imagen, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = 255  # Alfa completo (opaco)
    return rgba

with open(GROUND_TRUTH_CSV, mode="w", newline="") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["imagen_consulta", "id_correcto"])

    for nombre in imagenes_seleccionadas:
        ruta_original = os.path.join(CARPETA_ORIGEN, nombre)
        imagen = cv2.imread(ruta_original)

        if imagen is None:
            print(f"❌ No se pudo leer {nombre}")
            continue

        nombre_base, _ = os.path.splitext(nombre)
        id_correcto = f"{nombre_base}.png"

        # ---------- Copia Original con canal alfa (sin quitar fondo) ----------
        imagen_rgba = agregar_canal_alpha(imagen)
        nombre_orig = f"{nombre_base}_original.png"
        ruta_destino_orig = os.path.join(CARPETA_DESTINO, nombre_orig)
        cv2.imwrite(ruta_destino_orig, imagen_rgba)
        writer.writerow([nombre_orig, id_correcto])
        print(f"✅ Guardado: {nombre_orig}")

        # ---------- Imagen Sintética: gris + rotada (ángulo random) + canal alfa ----------
        gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        gris_bgr = cv2.cvtColor(gris, cv2.COLOR_GRAY2BGR)
        gris_bgr = cv2.resize(gris_bgr, TAMANO)

        alto, ancho = gris_bgr.shape[:2]
        centro = (ancho // 2, alto // 2)
        angulo = random.uniform(10, 20) * random.choice([-1, 1])  # Aleatorio entre -20° y +20°
        rot_mat = cv2.getRotationMatrix2D(centro, angulo, 1.0)
        rotada = cv2.warpAffine(gris_bgr, rot_mat, (ancho, alto), borderMode=cv2.BORDER_REPLICATE)

        rotada_rgba = agregar_canal_alpha(rotada)
        nombre_sint = f"{nombre_base}_sintetica.png"
        ruta_destino_sint = os.path.join(CARPETA_DESTINO, nombre_sint)
        cv2.imwrite(ruta_destino_sint, rotada_rgba)
        writer.writerow([nombre_sint, id_correcto])
        print(f"🎨 Guardado: {nombre_sint} (rotada {angulo:.1f}°)")

print(f"\n📄 Ground truth guardado en: {GROUND_TRUTH_CSV}")