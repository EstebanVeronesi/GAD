import os
from rembg import remove
from PIL import Image

# Carpeta con las imágenes originales
input_folder = 'Testset'
# Carpeta donde se guardarán las imágenes sin fondo
output_folder = 'test2'

# Crear carpeta de salida si no existe
os.makedirs(output_folder, exist_ok=True)

# Recorrer todos los archivos en la carpeta de entrada
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + '.png')

        # Abrir la imagen original
        with open(input_path, 'rb') as i:
            input_data = i.read()

        # Remover fondo
        output_data = remove(input_data)

        # Guardar la imagen resultante con fondo transparente
        with open(output_path, 'wb') as o:
            o.write(output_data)

        print(f'Procesada: {filename} -> {output_path}')

print('Proceso completado.')