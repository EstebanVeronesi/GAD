import psycopg2
import faiss
import numpy as np

# Conexión a PostgreSQL
def get_embeddings_from_postgres():
    conn = psycopg2.connect(
        host="localhost",
        database="Proyecto_final",
        user="postgres",
        password="159753"
    )

    cursor = conn.cursor()
    cursor.execute("SELECT id, embedding FROM faces;")
    data = cursor.fetchall()

    ids = [row[0] for row in data]
    embeddings = np.array([row[1] for row in data])

    cursor.close()
    conn.close()

    return ids, embeddings

# Normalizar los embeddings (normalización L2)
def normalize_embeddings(embeddings):
    embeddings = embeddings.astype(np.float32)
    faiss.normalize_L2(embeddings)
    return embeddings

# Construcción del índice FAISS
def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    embeddings = normalize_embeddings(embeddings)
    index.add(embeddings)
    return index

# Buscar los 10 más similares, excluyendo el propio ID
def search_similar_embeddings(query_embedding, index, ids, image_id, k=10):
    query_embedding = np.array([query_embedding]).astype(np.float32)
    faiss.normalize_L2(query_embedding)

    D, I = index.search(query_embedding, k + 1)

    similar_ids = []
    distances = []

    for idx, dist in zip(I[0], D[0]):
        if idx == -1:
            continue
        if ids[idx] != image_id:
            similar_ids.append(ids[idx])
            distances.append(dist)
        if len(similar_ids) == k:
            break

    return similar_ids, distances

# Función principal
def main():
    ids, embeddings = get_embeddings_from_postgres()
    index = build_faiss_index(embeddings)

    raw_id = input("Por favor, ingresa el ID de la imagen: ").strip()
    image_id = f"{raw_id}.png"  # Se agrega automáticamente la extensión

    if image_id not in ids:
        print(f"El ID '{image_id}' no se encuentra en la base de datos.")
        return

    query_embedding = embeddings[ids.index(image_id)]
    similar_ids, distances = search_similar_embeddings(query_embedding, index, ids, image_id)

    print("Top 10 más similares (excluyendo la imagen original):")
    for i, (sim_id, dist) in enumerate(zip(similar_ids, distances)):
        print(f"{i + 1}: ID: {sim_id}, Distancia de Coseno: {dist}")

if __name__ == "__main__":
    main()
