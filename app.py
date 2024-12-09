from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import os

app = Flask(__name__)

# Daftar model yang tersedia
models = {
    'model_1': 'models/model_ambon.h5',
    'model_2': 'models/model_balikpapan.h5',
    'model_3': 'models/model_banda_aceh.h5',
    'model_4': 'models/model_bandar_lampung.h5',
    'model_5': 'models/model_bandung.h5',
    'model_6': 'models/model_banjarmasin.h5',
    'model_7': 'models/model_banyuwangi.h5',
    'model_8': 'models/model_batam.h5',
    'model_9': 'models/model_bau_bau.h5',
    'model_10': 'models/model_bekasi.h5',
    'model_11': 'models/model_bengkulu.h5',
    'model_12': 'models/model_bima.h5',
    'model_13': 'models/model_bogor.h5',
    'model_14': 'models/model_bukittinggi.h5',
    'model_15': 'models/model_bulukumba.h5',
    'model_16': 'models/model_cilacap.h5',
    'model_17': 'models/model_cilegon.h5',
    'model_18': 'models/model_cirebon.h5',
    'model_19': 'models/model_denpasar.h5',
    'model_20': 'models/model_depok.h5',
    'model_21': 'models/model_dumai.h5',
    'model_22': 'models/model_gorontalo.h5',
    'model_23': 'models/model_jakarta_pusat.h5',
    'model_24': 'models/model_jambi.h5',
    'model_25': 'models/model_jayapura.h5',
    'model_26': 'models/model_jember.h5',
    'model_27': 'models/model_kediri.h5',
    'model_28': 'models/model_kendari.h5',
    'model_29': 'models/model_kudus.h5',
    'model_30': 'models/model_kupang.h5',
    'model_31': 'models/model_lhoksuemawe.h5',
    'model_32': 'models/model_lubuk_linggau.h5',
    'model_33': 'models/model_masiun.h5',
    'model_34': 'models/model_makassar.h5',
    'model_35': 'models/model_malang.h5',
    'model_36': 'models/model_mamuju.h5',
    'model_37': 'models/model_manado.h5',
    'model_38': 'models/model_mataram.h5',
    'model_39': 'models/model_maumere.h5',
    'model_40': 'models/model_medan.h5',
    'model_41': 'models/model_merauke.h5',
    'model_42': 'models/model_metro.h5',
    'model_43': 'models/model_meulaboh.h5',
    'model_44': 'models/model_padang_sidempuan.h5',
    'model_45': 'models/model_padang.h5',
    'model_46': 'models/model_palangkaraya.h5',
    'model_47': 'models/model_palembang.h5',
    'model_48': 'models/model_palopo.h5',
    'model_49': 'models/model_palu.h5',
    'model_50': 'models/model_pangkal_pinang.h5',
    'model_51': 'models/model_parepare.h5',
    'model_52': 'models/model_pontianak.h5',
    'model_53': 'models/model_samarinda.h5',
    'model_54': 'models/model_sampit.h5',
    'model_55': 'models/model_semarang.h5',
    'model_56': 'models/model_sumenep.h5',
}

# Fungsi untuk memuat model berdasarkan nama
def load_model_by_name(model_name):
    model_path = models.get(model_name)
    if not model_path or not os.path.exists(model_path):
        return None
    return tf.keras.models.load_model(model_path)

@app.route('/')
def home():
    return "Welcome to the prediction API! Use /predict for predictions."

@app.route('/favicon.ico')
def favicon():
    return '', 204  # No content, just to handle the favicon request

@app.route('/predict', methods=['POST'])
def predict():
    # Mendapatkan data JSON dari request
    data = request.get_json()

    # Mendapatkan nama model dari parameter
    model_name = data.get('model_name', 'model_1')  # Default ke 'model_1' jika tidak ada

    # Memuat model yang dipilih
    model = load_model_by_name(model_name)
    if model is None:
        return jsonify({'error': 'Model not found'}), 404

    # Memastikan data memiliki semua fitur yang diperlukan
    required_features = [
        'Daging_Ayam', 'Daging_Ayam_Ras_Segar', 'Bawang_Merah', 'Bawang_Merah_Ukuran_Sedang',
        'Bawang_Putih', 'Bawang_Putih_Ukuran_Sedang', 'Cabai_Merah', 'Cabai_Merah_Keriting',
        'Cabai_Rawit', 'Cabai_Rawit_Hijau', 'Cabai_Rawit_Merah'
    ]
    if not all(feature in data for feature in required_features):
        return jsonify({'error': 'Missing required features'}), 400

    # Mendapatkan periode waktu untuk prediksi
    time_period = data.get('time_period', '1_month')  # Default '1_month' jika tidak ada

    # Menyiapkan data input untuk prediksi
    input_data = np.array([[
        data['Daging_Ayam'],
        data['Daging_Ayam_Ras_Segar'],
        data['Bawang_Merah'],
        data['Bawang_Merah_Ukuran_Sedang'],
        data['Bawang_Putih'],
        data['Bawang_Putih_Ukuran_Sedang'],
        data['Cabai_Merah'],
        data['Cabai_Merah_Keriting'],
        data['Cabai_Rawit'],
        data['Cabai_Rawit_Hijau'],
        data['Cabai_Rawit_Merah']
    ]])

    # Melakukan prediksi dengan model yang dipilih
    prediction = model.predict(input_data)

    # Menyesuaikan hasil prediksi berdasarkan periode waktu
    if time_period == '1_month':
        predicted_value = prediction[0][0]
    elif time_period == '3_months':
        predicted_value = prediction[0][0] * 3  # Perkalian dengan 3 untuk 3 bulan
    elif time_period == '6_months':
        predicted_value = prediction[0][0] * 6  # Perkalian dengan 6 untuk 6 bulan
    elif time_period == '1_year':
        predicted_value = prediction[0][0] * 12  # Perkalian dengan 12 untuk 1 tahun
    else:
        return jsonify({'error': 'Invalid time period'}), 400

    # Mengembalikan hasil prediksi dalam bentuk JSON
    return jsonify({'predicted_inflation': predicted_value})

if __name__ == '__main__':
    app.run(debug=True)
