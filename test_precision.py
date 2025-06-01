import csv
from procesado import get_embeddings_from_postgres, build_faiss_index, search_similar_embeddings

# Cargar archivo CSV
def cargar_csv(ruta_csv):
    with open(ruta_csv, newline='') as f:
        reader = csv.reader(f)
        next(reader)  # Saltar encabezado
        datos = [(fila[0], fila[1]) for fila in reader]
    return datos

def evaluar_precision(csv_path, k=1):
    datos_prueba = cargar_csv(csv_path)
    ids, embeddings = get_embeddings_from_postgres()
    index = build_faiss_index(embeddings)

    aciertos = 0

    for id_consulta, id_esperado in datos_prueba:
        id_consulta_raw = id_consulta.replace(".png", "").strip()
        id_esperado_png = id_esperado.strip()

        image_id = f"{id_consulta_raw}.png"

        if image_id not in ids:
            print(f"ID {image_id} no está en la base de datos. Se omite.")
            continue

        query_embedding = embeddings[ids.index(image_id)]
        resultados, _ = search_similar_embeddings(query_embedding, index, ids, image_id, k=k)

        if id_esperado_png in resultados:
            aciertos += 1
        else:
            print(f"Falló: {image_id} → Esperado: {id_esperado_png}, Obtenido: {resultados}")

    total = len(datos_prueba)
    precision = aciertos / total * 100
    print(f"\nPrecisión Top-{k}: {precision:.2f}% ({aciertos}/{total})")

# Ejecutar pruebas
if __name__ == "__main__":
    evaluar_precision("Testset\ground_truth.csv", k=1)