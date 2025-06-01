import streamlit as st
from PIL import Image
import os
from procesado import get_embeddings_from_postgres, build_faiss_index, search_similar_embeddings

# Configuración
IMAGENES_PATH = "Testset"
TOP_K = 5

st.title("🔍 Búsqueda de Imágenes Similares")
st.write("Seleccioná una imagen por ID para ver los más parecidos.")

# Cargar embeddings y FAISS index
@st.cache_resource
def cargar_datos():
    ids, embeddings = get_embeddings_from_postgres()
    index = build_faiss_index(embeddings)
    return ids, embeddings, index

ids, embeddings, index = cargar_datos()

# Seleccionar ID
id_opciones = sorted([i.replace(".png", "") for i in ids])
id_consulta = st.selectbox("Seleccioná un ID de imagen", id_opciones)

# Mostrar imagen de consulta
image_path = os.path.join(IMAGENES_PATH, f"{id_consulta}.png")
if os.path.exists(image_path):
    st.image(image_path, caption="Imagen consultada", use_container_width=True)
else:
    st.warning("Imagen no encontrada en carpeta.")

# Búsqueda
if st.button("Buscar similares"):
    image_id = f"{id_consulta}.png"
    query_embedding = embeddings[ids.index(image_id)]
    similares, distancias = search_similar_embeddings(query_embedding, index, ids, image_id, k=TOP_K)

    st.subheader(f"Top {TOP_K} resultados:")
    for sim_id, dist in zip(similares, distancias):
        img_path = os.path.join(IMAGENES_PATH, sim_id)
        cols = st.columns([1, 4])
        with cols[0]:
            if os.path.exists(img_path):
                st.image(img_path, width=100)
        with cols[1]:
            st.markdown(f"**ID**: `{sim_id}`")
            st.markdown(f"**Similitud (coseno)**: `{dist:.4f}`")