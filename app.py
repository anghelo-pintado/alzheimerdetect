import os
import io
import pickle

import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template, make_response
from flask_cors import CORS
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas as pdf_canvas
from reportlab.platypus import Table, TableStyle
from reportlab.lib import colors

# Configuración de rutas absolutas
d_basedir = os.path.abspath(os.path.dirname(__file__))
model_path = os.path.join(d_basedir, 'model_mlp.h5')
scaler_path = os.path.join(d_basedir, 'scaler.pkl')
logo_path = os.path.join(d_basedir, 'static', 'logo.png')

def create_app():
    app = Flask(__name__, template_folder=os.path.join(d_basedir, 'templates'), static_folder=os.path.join(d_basedir, 'static'))
    CORS(app)

    # Carga del modelo y scaler
    try:
        model = tf.keras.models.load_model(model_path)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        print("✅ Modelo y scaler cargados correctamente.")
    except Exception as e:
        print(f"❌ Error al cargar los artefactos: {e}")
        model, scaler = None, None

    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/predict', methods=['POST'])
    def predict():
        if not model or not scaler:
            return jsonify({"error": "El modelo o el scaler no están disponibles."}), 500
        data = request.get_json()
        if data is None:
            return jsonify({"error": "JSON inválido o no enviado"}), 400
        try:
            feature_order = ['Age', 'MMSE', 'EDUC', 'SES', 'eTIV', 'nWBV', 'ASF']
            missing = [k for k in feature_order if k not in data]
            if missing:
                return jsonify({"error": f"Faltan los siguientes campos: {', '.join(missing)}"}), 400

            vals = np.array([float(data[k]) for k in feature_order]).reshape(1, -1)
            scaled = scaler.transform(vals)
            prob = model.predict(scaled, verbose=0)[0][0]
            cls = int(prob >= 0.5)
            label = "Riesgo de Alzheimer" if cls else "Sin Riesgo de Alzheimer"
            return jsonify({
                'prediction': cls,
                'label': label,
                'confidence': f"{prob:.2%}"
            })
        except Exception as e:
            return jsonify({"error": f"Ocurrió un error al realizar la predicción: {e}"}), 500

    @app.route('/generate_pdf', methods=['POST'])
    def generate_pdf():
        data = request.get_json()
        if data is None:
            return jsonify({"error": "JSON inválido o no enviado"}), 400

        paciente = data.get('name', 'Paciente Desconocido')
        paciente_id = data.get('idPaciente', '')
        edad = data.get('Age', '')
        fecha = data.get('date', '')
        label = data.get('label', '')
        confidence = data.get('confidence', '')

        buffer = io.BytesIO()
        try:
            c = pdf_canvas.Canvas(buffer, pagesize=letter)
            width, height = letter

            # --- Logo proporcional ---
            img = Image.open(logo_path)
            aspect = img.height / img.width
            logo_w = 30 * mm
            logo_h = logo_w * aspect
            c.drawImage(logo_path, 20*mm, height - 30*mm, width=logo_w, height=logo_h, preserveAspectRatio=True)

            # --- Encabezado de texto ---
            x0, y0 = width - 80*mm, height - 20*mm
            c.setFont("Helvetica", 8)
            c.drawString(x0, y0 - 4*mm, "Central telefónica: (044) 480730 anexo 1")
            c.drawString(x0, y0 - 8*mm, "WhatsApp: 960 945 908")
            c.drawString(x0, y0 - 12*mm, "Sede principal: Jr. Bolognesi 334")

            # Línea separadora
            c.setStrokeColor(colors.grey)
            c.setLineWidth(0.5)
            c.line(15*mm, height - 35*mm, width - 15*mm, height - 35*mm)

            # Título
            c.setFont("Helvetica-Bold", 14)
            c.setFillColor(colors.green)
            c.drawCentredString(width/2, height - 45*mm, "INFORME DE RESULTADOS DE ANÁLISIS")
            c.setFillColor(colors.black)

            # Datos del paciente
            c.setFont("Helvetica-Bold", 10)
            c.drawString(20*mm, height - 60*mm, "Paciente:")
            c.drawString(100*mm, height - 60*mm, paciente)
            c.drawString(20*mm, height - 68*mm, "ID Paciente:")
            c.drawString(100*mm, height - 68*mm, paciente_id)
            c.drawString(20*mm, height - 76*mm, "Edad:")
            c.drawString(100*mm, height - 76*mm, str(edad))
            c.drawString(20*mm, height - 84*mm, "Fecha de Informe:")
            c.drawString(100*mm, height - 84*mm, fecha)

            # Tabla de resultados
            table_data = [
                ['Análisis', 'Resultado', 'Unidad / Notas'],
                ['Predicción Alzheimer', label, confidence]
            ]
            tbl = Table(table_data, colWidths=[70*mm, 60*mm, 50*mm])
            tbl.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
                ('TEXTCOLOR',   (0,0), (-1,0), colors.black),
                ('ALIGN',       (0,0), (-1,-1), 'CENTER'),
                ('FONTNAME',    (0,0), (-1,0), 'Helvetica-Bold'),
                ('FONTSIZE',    (0,0), (-1,0), 10),
                ('GRID',        (0,0), (-1,-1), 0.5, colors.grey),
                ('FONTSIZE',    (0,1), (-1,-1), 9),
            ]))
            w, h = tbl.wrap(0, 0)
            tbl.drawOn(c, 20*mm, height - 120*mm - h)

            # Firma
            c.line(30*mm, 40*mm, 80*mm, 40*mm)
            c.setFont("Helvetica", 8)
            c.drawString(30*mm, 36*mm, "Dr. Nombre del Especialista")
            c.drawString(30*mm, 32*mm, "Especialista en Neurología")

            # Pie de página
            c.setFont("Helvetica-Oblique", 7)
            c.drawRightString(width - 20*mm, 30*mm, "Generado automáticamente por escalabs")

            c.showPage()
            c.save()
        except Exception as e:
            return jsonify({"error": f"Error al generar PDF: {e}"}), 500

        buffer.seek(0)
        response = make_response(buffer.read())
        response.headers.set('Content-Type', 'application/pdf')
        response.headers.set('Content-Disposition', f'attachment; filename={paciente}_reporte.pdf')
        return response

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=5000, debug=True)
