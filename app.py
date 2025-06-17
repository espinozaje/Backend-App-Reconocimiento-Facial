from flask import Flask, request, jsonify
import cv2, os, json, mysql.connector
import numpy as np
from base64 import b64decode
from datetime import datetime
from twilio.rest import Client
import threading
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

app = Flask(__name__)
os.makedirs("modelo", exist_ok=True)
os.makedirs("modelo_entrenado", exist_ok=True)

# Envío de SMS con Twilio
def enviar_sms_alerta(numero_destino, mensaje):
    account_sid = 'ACdf4f31bbd04400119b690f6c7c09f53a'
    auth_token = '442b9e403d9cb25c710d508302aee8f8'
    client = Client(account_sid, auth_token)
    message = client.messages.create(body=mensaje, from_='+16282824764', to=numero_destino)
    print("Mensaje enviado con SID:", message.sid)

# Conexión a la base de datos

def conectar_bd():
    return mysql.connector.connect(
        host='localhost',
        user='root',
        password='upao',
        database='reconocimiento'
    )

# Crear tabla con campos ampliados

def inicializar_tabla():
    conn = conectar_bd()
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS usuarios (
        id INT AUTO_INCREMENT PRIMARY KEY,
        nombre VARCHAR(100),
        apellido VARCHAR(100),
        codigo VARCHAR(50) UNIQUE,
        email VARCHAR(100),
        requisitoriado BOOLEAN,
        direccion VARCHAR(255),
        imagen VARCHAR(255),
        keypoints JSON,
        descriptors JSON,
        fecha DATETIME
    )""")
    conn.commit()
    cursor.close()
    conn.close()

inicializar_tabla()

# ORB para características

def extraer_caracteristicas(imagen_array):
    gray = cv2.cvtColor(imagen_array, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        return None, None
    x, y, w, h = faces[0]
    roi = gray[y:y+h, x:x+w]
    orb = cv2.ORB_create()
    kp, desc = orb.detectAndCompute(roi, None)
    return kp, desc

def keypoints_to_json(kps):
    return [{"pt": kp.pt, "size": kp.size, "angle": kp.angle, "response": kp.response, "octave": kp.octave} for kp in kps]

def json_to_keypoints(json_kp):
    return [cv2.KeyPoint(kp["pt"][0], kp["pt"][1], kp["size"], kp["angle"], kp["response"], kp["octave"], 0) for kp in json_kp]

# CNN

def preprocess_img_for_cnn(img):
    img_resized = cv2.resize(img, (100, 100))
    img_array = img_to_array(img_resized) / 255.0
    return np.expand_dims(img_array, axis=0)

def cargar_modelo_cnn():
    if os.path.exists("modelo_entrenado/modelo_cnn.h5"):
        return load_model("modelo_entrenado/modelo_cnn.h5")
    return None

def entrenar_cnn_en_segundo_plano():
    try:
        datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
        train_gen = datagen.flow_from_directory('modelo', target_size=(100, 100), batch_size=16, class_mode='categorical', subset='training')
        val_gen = datagen.flow_from_directory('modelo', target_size=(100, 100), batch_size=16, class_mode='categorical', subset='validation')

        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
            MaxPooling2D(2, 2),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Flatten(),
            Dense(64, activation='relu'),
            Dense(train_gen.num_classes, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(train_gen, epochs=5, validation_data=val_gen)
        model.save("modelo_entrenado/modelo_cnn.h5")
        print("Entrenamiento CNN completado.")
    except Exception as e:
        print("Error entrenamiento CNN:", e)

# Registro
@app.route('/registro', methods=['POST'])
def registro():
    data = request.json
    image_data = b64decode(data['imagen'].split(',')[1])
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    nombre = data['nombre']
    apellido = data['apellido']
    codigo = data['codigo']
    email = data['email']
    direccion = data['direccion']
    requisitoriado = data['requisitoriado']

    kp, desc = extraer_caracteristicas(img)
    if kp is None or desc is None:
        return jsonify({"status": "error", "mensaje": "No se detectó rostro"})

    carpeta_usuario = f"modelo/{nombre}_{apellido}"
    os.makedirs(carpeta_usuario, exist_ok=True)
    filename = f"{carpeta_usuario}/{codigo}_{datetime.now().timestamp()}.jpg"
    cv2.imwrite(filename, img)

    try:
        conn = conectar_bd()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO usuarios (nombre, apellido, codigo, email, requisitoriado, direccion, imagen, keypoints, descriptors, fecha)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (nombre, apellido, codigo, email, requisitoriado, direccion, filename,
              json.dumps(keypoints_to_json(kp)), json.dumps(desc.tolist()), datetime.now()))
        conn.commit()
        cursor.close()
        conn.close()
        threading.Thread(target=entrenar_cnn_en_segundo_plano).start()
        return jsonify({"status": "ok", "mensaje": "Usuario registrado exitosamente"})
    except Exception as e:
        return jsonify({"status": "error", "mensaje": str(e)})

# Verificación
@app.route('/verificar', methods=['POST'])
def verificar():
    data = request.json
    image_data = b64decode(data['imagen'].split(',')[1])
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    modelo_cnn = cargar_modelo_cnn()
    if modelo_cnn:
        try:
            img_cnn = preprocess_img_for_cnn(img)
            pred = modelo_cnn.predict(img_cnn)
            clase_predicha = np.argmax(pred)
            confianza = round(np.max(pred) * 100, 2)

            datagen = ImageDataGenerator(rescale=1./255)
            flow = datagen.flow_from_directory('modelo', target_size=(100, 100), batch_size=1, class_mode='categorical')
            nombre_predicho = list(flow.class_indices.keys())[clase_predicha]

            # Verificar si es requisitoriado
            conn = conectar_bd()
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT * FROM usuarios WHERE CONCAT(nombre, '_', apellido) = %s", (nombre_predicho,))
            usuario = cursor.fetchone()
            cursor.close()
            conn.close()

            if confianza > 60:
                if usuario and usuario['requisitoriado']:
                    enviar_sms_alerta('+51952272635', f"ALERTA: Usuario Requisitoriado Detectado: {usuario['nombre']} {usuario['apellido']}")
                    return jsonify({"status": "alerta", "mensaje": f"¡ALERTA! {usuario['nombre']} {usuario['apellido']} es requisitoriado"})
                else:
                    return jsonify({"status": "ok", "mensaje": f"Rostro reconocido: {nombre_predicho} ({confianza}% confianza)"})
            else:
                return jsonify({"status": "ok", "mensaje": f"Rostro no reconocido ({confianza}% confianza)"})
        except Exception as e:
            print("Error CNN:", e)

    return jsonify({"status": "error", "mensaje": "No se pudo verificar"})

# CRUD usuarios
@app.route('/usuarios', methods=['GET'])
def listar_usuarios():
    try:
        conn = conectar_bd()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM usuarios")
        usuarios = cursor.fetchall()
        cursor.close()
        conn.close()
        return jsonify(usuarios)
    except Exception as e:
        return jsonify({"status": "error", "mensaje": str(e)})

@app.route('/usuario/<int:uid>', methods=['GET'])
def obtener_usuario(uid):
    conn = conectar_bd()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM usuarios WHERE id = %s", (uid,))
    usuario = cursor.fetchone()
    cursor.close()
    conn.close()
    return jsonify(usuario)

@app.route('/usuario/<int:uid>', methods=['PUT'])
def actualizar_usuario(uid):
    data = request.json
    campos = ["nombre", "apellido", "codigo", "email", "requisitoriado", "direccion"]
    valores = [data[campo] for campo in campos]
    consulta = ", ".join([f"{campo} = %s" for campo in campos])
    conn = conectar_bd()
    cursor = conn.cursor()
    cursor.execute(f"UPDATE usuarios SET {consulta} WHERE id = %s", (*valores, uid))
    conn.commit()
    cursor.close()
    conn.close()
    return jsonify({"status": "ok", "mensaje": "Usuario actualizado"})

@app.route('/usuario/<int:uid>', methods=['DELETE   '])
def eliminar_usuario(uid):
    conn = conectar_bd()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM usuarios WHERE id = %s", (uid,))
    conn.commit()
    cursor.close()
    conn.close()
    return jsonify({"status": "ok", "mensaje": "Usuario eliminado"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)