import cv2
import os
import re
import psycopg2
import numpy as np
import faiss
from deepface import DeepFace

# Datos de conexión
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "pruebas_final"
DB_USER = "postgres"
DB_PASSWORD = "159753"

try:
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )
    cursor = conn.cursor()
    print("✅ Conectado a PostgreSQL")
except Exception as e:
    print(f"❌ Error al conectar a la base de datos: {e}")
    exit()

# Crear tabla si no existe
cursor.execute("""
    CREATE TABLE IF NOT EXISTS faces (
        id TEXT PRIMARY KEY,
        embedding FLOAT8[]
    );
""")
conn.commit()

# Carpeta con las imágenes
carpeta_imagenes = r"Testset"

# Obtener imágenes con nombres numéricos
imagenes = sorted(
    [f for f in os.listdir(carpeta_imagenes) if re.fullmatch(r"\d+\.png", f)],
    key=lambda x: int(x.split(".")[0])
)

# Procesar imágenes
for img_nombre in imagenes:
    ruta_img = os.path.join(carpeta_imagenes, img_nombre)
    print(f"📷 Procesando: {ruta_img}")

    # Cargar imagen
    image = cv2.imread(ruta_img)

    if image is None:
        print(f"❌ Imagen corrupta o no compatible: {ruta_img}, se omite.")
        continue

    # Redimensionar imagen (sin conversión a gris ni ecualización)
    image_resized = cv2.resize(image, (300, 300))

    # Obtener embeddings
    try:
        embeddings = DeepFace.represent(
            image_resized,
            model_name="Facenet512",
            enforce_detection=False
        )

        for embedding in embeddings:
            embedding_vector = np.array(embedding['embedding'], dtype=np.float32)

            # Normalizar L2
            embedding_vector = np.expand_dims(embedding_vector, axis=0)
            faiss.normalize_L2(embedding_vector)
            normalized_embedding = embedding_vector[0].tolist()

            # Insertar en la base de datos
            cursor.execute(
                "INSERT INTO faces (id, embedding) VALUES (%s, %s) ON CONFLICT (id) DO NOTHING;",
                (img_nombre, normalized_embedding),
            )
        conn.commit()
        print(f"✅ Embedding de {img_nombre} guardado en la base de datos.")
    except Exception as e:
        print(f"❌ Error al obtener o guardar el embedding para {img_nombre}: {e}")
        conn.rollback()

# Cerrar conexión a PostgreSQL
cursor.close()
conn.close()
print("🔚 Proceso finalizado.")