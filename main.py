import struct
import zlib
import numpy as np
import psycopg2
import os

def read_png(file_path):
    # Open the specified PNG file in binary read mode
    with open(file_path, "rb") as file:
        # Read the first 8 bytes to check the PNG signature
        signature = file.read(8)
        if signature != b'\x89PNG\r\n\x1a\n':
            raise ValueError("The file is not a valid PNG")

        chunks = []  # List to store the chunks found in the PNG file
        while True:
            chunk_length_data = file.read(4)
            if len(chunk_length_data) == 0:
                break
            chunk_length = struct.unpack(">I", chunk_length_data)[0]
            chunk_type = file.read(4).decode("ascii")
            chunk_data = file.read(chunk_length)
            crc = file.read(4)
            chunks.append((chunk_type, chunk_data))
            if chunk_type == "IEND":
                break

    header = None
    image_data = b""
    palette = None

    for chunk_type, chunk_data in chunks:
        if chunk_type == "IHDR":
            header = struct.unpack(">IIBBBBB", chunk_data[:13])
        elif chunk_type == "PLTE":
            palette = chunk_data
        elif chunk_type == "IDAT":
            image_data += chunk_data

    if header is None:
        raise ValueError("PNG file without IHDR header")

    width, height, bit_depth, color_type, compression, filter_method, interlace = header
    print(f"Dimensions: {width}x{height}, Depth: {bit_depth}, Color type: {color_type}")

    decompressed_data = zlib.decompress(image_data)

    pixels = []
    if color_type == 3:
        for y in range(height):
            row_start = y * (width + 1)
            filter_type = decompressed_data[row_start]
            row_data = decompressed_data[row_start + 1: row_start + width + 1]
            color_row = []
            for index in row_data:
                start = index * 3
                r = palette[start]
                g = palette[start + 1]
                b = palette[start + 2]
                color_row.append((r, g, b))
            pixels.append(color_row)
    else:
        raise ValueError(f"Color type {color_type} not supported for this example.")

    return pixels


def create_table_if_not_exists(connection):
    # Create table if it doesn't exist
    create_table_query = """
    CREATE TABLE IF NOT EXISTS image_vectors (
        id VARCHAR(255) PRIMARY KEY,
        feature_vector FLOAT[]
    );
    """
    with connection.cursor() as cursor:
        cursor.execute(create_table_query)
        connection.commit()
        print("Table 'image_vectors' ensured to exist.")


def store_image_vector_in_db(connection, image_id, feature_vector):
    # Insert the image ID and feature vector into the database
    insert_query = """
    INSERT INTO image_vectors (id, feature_vector)
    VALUES (%s, %s)
    ON CONFLICT (id) DO NOTHING;  -- Avoid duplicate IDs
    """
    with connection.cursor() as cursor:
        cursor.execute(insert_query, (image_id, feature_vector))
        connection.commit()
        print(f"Image vector for '{image_id}' stored successfully.")


def process_and_store_images(folder_path, db_config):
    # Connect to the database
    with psycopg2.connect(**db_config) as connection:
        create_table_if_not_exists(connection)

        for filename in os.listdir(folder_path):
            if filename.endswith(".png"):  # Process only PNG files
                file_path = os.path.join(folder_path, filename)
                try:
                    # Read and process the PNG file
                    pixels = read_png(file_path)
                    if pixels is not None:
                        pixel_array = np.array(pixels, dtype=np.uint8)
                        image_vector = pixel_array.flatten().tolist()

                        # Store the vector in the database
                        image_id = os.path.splitext(filename)[0]  # Use the filename (without extension) as ID
                        store_image_vector_in_db(connection, image_id, image_vector)
                except Exception as e:
                    print(f"Error processing {filename}: {e}")


# Configuration for database connection
db_config = {
    'host': 'localhost',
    'port': 5432,
    'user': 'postgres',
    'password': '159753',
    'database': 'Proyecto_final'
}

# Folder path containing PNG files
folder_path = r"C:\Users\veron\OneDrive - ConcepcióndelUruguay\Gestion avanzada de datos\Proyecto final\faces\faces"

# Process all images in the folder and store them in the database
process_and_store_images(folder_path, db_config)

try:
    conn = psycopg2.connect(**db_config)
    print("Conexión exitosa")
except Exception as e:
    print(f"Error al conectar: {e}")
    



