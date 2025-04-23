import psycopg2
import faiss
import numpy as np

# Conexión a PostgreSQL
def get_embeddings_from_postgres():
    conn = psycopg2.connect(
        host="localhost",        # Cambia según tu configuración
        database="Proyecto_final",
        user="postgres",
        password="159753"
    )

    cursor = conn.cursor()

    # Aquí tomamos los embeddings de la tabla 'faces' (cambia según tu tabla y esquema)
    cursor.execute("SELECT id, embedding FROM faces;")
    
    data = cursor.fetchall()
    ids = [row[0] for row in data]  # Obtener los ids

    # Aquí usamos np.array directamente, porque los embeddings ya son números flotantes.
    embeddings = np.array([row[1] for row in data])  # Los embeddings están directamente como float64 (double precision)

    cursor.close()
    conn.close()

    return ids, embeddings

# Normalizar los embeddings (normalización L2)
def normalize_embeddings(embeddings):
    embeddings = embeddings.astype(np.float32)  # Convertimos los embeddings a float32
    faiss.normalize_L2(embeddings)  # Normaliza los embeddings en el espacio L2
    return embeddings

# Construcción del índice FAISS (usando IndexFlatIP para mejorar la precisión con similitud de coseno)
def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]  # La dimensión de los embeddings
    
    # Usamos el índice IndexFlatIP para obtener una búsqueda exacta por similitud de coseno (producto interno)
    index = faiss.IndexFlatIP(dimension)  # Índice exacto basado en producto interno (coseno)
    
    # Normalizamos los embeddings antes de añadirlos al índice
    embeddings = normalize_embeddings(embeddings)
    
    # Añadimos los embeddings al índice
    index.add(embeddings)  # Añadimos los embeddings al índice
    return index

# Buscar los 10 más similares a un embedding dado
def search_similar_embeddings(query_embedding, index, ids, k=10):
    # Normalizamos el embedding de consulta antes de la búsqueda
    query_embedding = np.array([query_embedding]).astype(np.float32)
    faiss.normalize_L2(query_embedding)  # Normalizar la consulta para usar similitud de coseno
    
    # Realizamos la búsqueda de los k más similares
    D, I = index.search(query_embedding, k)
    
    # Extraemos los ids de los resultados
    similar_ids = [ids[i] for i in I[0]]
    return similar_ids, D[0]

# Función principal
def main():
    # Paso 1: Obtener los embeddings desde PostgreSQL
    ids, embeddings = get_embeddings_from_postgres()

    # Paso 2: Construir el índice FAISS usando IndexFlatIP para similitud de coseno
    index = build_faiss_index(embeddings)

    # Paso 3: Pedir al usuario que ingrese el ID de la imagen a buscar
    image_id = input("Por favor, ingresa el ID de la imagen que deseas buscar: ")

    # Comprobar si el ID ingresado existe
    if image_id not in ids:
        print(f"El ID '{image_id}' no se encuentra en la base de datos.")
        return

    # Obtener el embedding de la imagen seleccionada
    query_embedding = embeddings[ids.index(image_id)]  # Obtener el embedding de la imagen seleccionada

    # Paso 4: Buscar los 10 más similares
    similar_ids, distances = search_similar_embeddings(query_embedding, index, ids)

    # Mostrar los resultados
    print("Top 10 más similares:")
    for i, (sim_id, dist) in enumerate(zip(similar_ids, distances)):
        print(f"{i + 1}: ID: {sim_id}, Distancia de Coseno: {dist}")

if __name__ == "__main__":
    main()
