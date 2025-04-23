import cv2
import os
import re
import psycopg2
from deepface import DeepFace

# Datos de conexión
DB_HOST = "172.17.0.2"  # O usa "127.0.0.1" si prefieres
DB_PORT = "5432"
DB_NAME = "postgres"  # O la base de datos que creaste
DB_USER = "admin"
DB_PASSWORD = "admin"

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
carpeta_imagenes = r"C:\faces"

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

    # Redimensionar a 640x640
    image_resized = cv2.resize(image, (300, 300))

    # Mejorar iluminación y contraste
    gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
    image_resized = cv2.cvtColor(cv2.equalizeHist(gray), cv2.COLOR_GRAY2BGR)

    # Usar DeepFace para detectar rostros y obtener embeddings
    try:
        # Procesar imagen para obtener embeddings
        embeddings = DeepFace.represent(image_resized, model_name="VGG-Face", enforce_detection=False)

        # Guardar los embeddings en la base de datos
        for embedding in embeddings:
            embedding_list = embedding['embedding']  # Ya es una lista, no es necesario usar tolist()

            cursor.execute(
                "INSERT INTO faces (id, embedding) VALUES (%s, %s) ON CONFLICT (id) DO NOTHING;",
                (img_nombre, embedding_list),
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
