from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
from groq import Groq
import os
from dotenv import load_dotenv
from groq import Groq # Pastikan import Groq tetap ada

app = Flask(__name__)

# ─── Load Model, Scaler, Label Encoder & Daftar Fitur (8 Fitur) ───
model_xgb    = joblib.load('model_xgb_8fitur.pkl')
model_rf     = joblib.load('model_rf_8fitur.pkl')
scaler       = joblib.load('scaler_8fitur.pkl')
fitur_list   = joblib.load('feature_columns_8fitur.pkl')
label_encoders = joblib.load('label_encoders_8fitur.pkl')

# smoking_status encoder: classes_ = ['Current', 'Former', 'Never']
# Mapping dari nilai numerik UI → string kategori
SMOKING_MAP = {
    0: 'Never',
    1: 'Former',
    2: 'Current'
}

# ─── Konfigurasi Groq AI ───
# Memuat variabel dari file .env
load_dotenv()

# Mengambil API Key dari environment variable
api_key = os.getenv("GROQ_API_KEY")

# Inisialisasi client Groq
client = Groq(api_key=api_key)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data_raw = request.json

        # ─── Default values untuk fitur yang kosong ───
        default_values = {
            'age': 40.0,
            'bmi': 25.0,
            'hba1c': 5.4,
            'glucose_fasting': 95.0,
            'hypertension_history': 0,
            'family_history_diabetes': 0,
            'smoking_status': 0,
            'physical_activity_minutes_per_week': 150.0,
        }

        final_data = {}
        for kolom in fitur_list:
            val = data_raw.get(kolom)
            if val is None or val == '':
                final_data[kolom] = default_values.get(kolom, 0)
            else:
                final_data[kolom] = float(val)

        # ─── Encode smoking_status: numeric (0/1/2) → string → label encode ───
        smoking_num = int(final_data['smoking_status'])
        smoking_str = SMOKING_MAP.get(smoking_num, 'Never')
        smoking_encoded = int(label_encoders['smoking_status'].transform([smoking_str])[0])
        final_data['smoking_status'] = float(smoking_encoded)

        # ─── Preprocessing ───
        df_input = pd.DataFrame([final_data])[fitur_list]
        df_input_scaled = scaler.transform(df_input)

        # ─── Prediksi dengan XGBoost (model utama) ───
        prediction   = model_xgb.predict(df_input_scaled)[0]
        probability  = model_xgb.predict_proba(df_input_scaled)[0][1] * 100

        # ─── Prediksi juga dengan Random Forest (ensemble insight) ───
        prob_rf      = model_rf.predict_proba(df_input_scaled)[0][1] * 100
        prob_avg     = (probability + prob_rf) / 2   # rata-rata ensemble

        # ─── Klasifikasi Risiko ───
        if prediction == 1:
            hasil_teks = 'RISIKO TINGGI'
        elif prob_avg > 25:
            hasil_teks = 'RISIKO SEDANG'
        else:
            hasil_teks = 'RISIKO RENDAH'

        return jsonify({
            'status': 'success',
            'hasil': hasil_teks,
            'skor': float(probability),
            'skor_rf': float(prob_rf),
            'skor_ensemble': float(prob_avg),
            'data_pasien': {k: v for k, v in data_raw.items()},
            'smoking_label': smoking_str
        })

    except Exception as e:
        print(f"Error Predict: {str(e)}")
        import traceback; traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 400


@app.route('/get_recommendation', methods=['POST'])
def get_recommendation():
    try:
        req_data = request.json
        hasil    = req_data.get('hasil')
        p        = req_data.get('profil', {})

        # ─── Label mappings ───
        smoking_map   = {0: 'Tidak Pernah Merokok', 1: 'Mantan Perokok', 2: 'Perokok Aktif'}
        smoking_label = smoking_map.get(int(float(p.get('smoking_status', 0))), 'Tidak diketahui')
        family_label  = 'Ada' if int(float(p.get('family_history_diabetes', 0))) == 1 else 'Tidak Ada'
        hyper_label   = 'Ada' if int(float(p.get('hypertension_history', 0))) == 1 else 'Tidak Ada'

        prompt_content = f"""
Anda adalah dokter spesialis endokrinologi dan ahli gizi klinis yang sedang membuat laporan medis komprehensif untuk pasien berikut.

## DATA KLINIS PASIEN

**Profil Umum:**
- Usia: {p.get('age')} tahun
- Hasil Screening Diabetes AI: **{hasil}**

**Data Antropometri:**
- BMI: {p.get('bmi')} kg/m²

**Hasil Laboratorium:**
- Glukosa Puasa: {p.get('glucose_fasting')} mg/dL
- HbA1c: {p.get('hba1c')}%

**Riwayat Medis:**
- Riwayat Hipertensi: {hyper_label}
- Riwayat Diabetes Keluarga: {family_label}

**Gaya Hidup:**
- Aktivitas Fisik: {p.get('physical_activity_minutes_per_week')} menit/minggu
- Status Merokok: {smoking_label}

---

Buatkan **Laporan Klinis Komprehensif** yang SANGAT DETAIL dalam Bahasa Indonesia profesional dengan struktur berikut. Setiap bagian HARUS membahas angka spesifik dari data pasien di atas, BUKAN panduan umum:

## 1. 🔬 INTERPRETASI BIOMARKER
Analisis setiap nilai secara individual: BMI, glukosa puasa, dan HbA1c. Untuk setiap nilai, sebutkan rentang normalnya, bandingkan dengan nilai pasien, dan jelaskan implikasinya. Gunakan bahasa yang mudah dipahami awam.

## 2. ⚠️ FAKTOR RISIKO YANG DITEMUKAN
Identifikasi minimal 4-6 faktor risiko spesifik dari data pasien ini. Jelaskan mekanisme biologis mengapa setiap faktor tersebut berkontribusi pada risiko diabetes. Kaitkan dengan data aktual pasien.

## 3. 🥗 PROGRAM NUTRISI PERSONAL (Khusus untuk Pasien Ini)
Buat rencana makan SPESIFIK yang disesuaikan dengan kondisi pasien berdasarkan angka biomarkernya:
- Estimasi kebutuhan kalori harian berdasarkan BMI pasien
- Daftar 10 makanan yang direkomendasikan beserta manfaatnya
- Daftar 8 makanan/minuman yang harus dihindari beserta alasannya
- Contoh menu sarapan, makan siang, dan makan malam yang konkret
- Tips khusus berdasarkan kadar glukosa/HbA1c pasien

## 4. 🏃 PROGRAM LATIHAN FISIK (Disesuaikan Kondisi Pasien)
Buat program olahraga bertahap yang spesifik mengacu pada tingkat aktivitas fisik pasien saat ini ({p.get('physical_activity_minutes_per_week')} menit/minggu):
- Minggu 1-2: Jenis, durasi, dan intensitas latihan ringan
- Minggu 3-4: Peningkatan bertahap
- Bulan 2-3: Target latihan jangka menengah
- Olahraga spesifik yang paling efektif untuk kondisi pasien ini

## 5. 😴 OPTIMASI TIDUR & MANAJEMEN STRES
Jelaskan hubungan antara kualitas tidur dan kadar glukosa darah, serta strategi konkret untuk perbaikan.

## 6. 📅 JADWAL MONITORING RUTIN
Buat jadwal pemantauan yang terstruktur:
- Pemeriksaan apa, seberapa sering, dan target nilai yang harus dicapai dalam 3 bulan ke depan
- Cara memantau kadar gula secara mandiri di rumah
- Tanda-tanda perbaikan yang harus diperhatikan

## 7. 🏥 TANDA BAHAYA & KAPAN HARUS KE DOKTER
Sebutkan gejala atau nilai spesifik yang harus segera dikonsultasikan ke dokter berdasarkan kondisi pasien saat ini.

---
⚠️ *Laporan ini dihasilkan secara otomatis oleh sistem AI HerPredict berdasarkan data yang diinput dan bukan merupakan diagnosis medis resmi. Selalu konsultasikan kondisi Anda dengan dokter atau tenaga medis profesional sebelum mengambil keputusan terkait kesehatan.*
"""

        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Anda adalah dokter spesialis endokrinologi dan ahli gizi klinis senior dengan pengalaman 20 tahun. "
                        "Tugas Anda adalah membuat laporan klinis yang sangat komprehensif, panjang, detail, dan personal. "
                        "Gunakan Bahasa Indonesia yang profesional namun mudah dipahami pasien awam. "
                        "WAJIB: Setiap bagian harus mengacu pada angka aktual dari data pasien — JANGAN beri jawaban generik. "
                        "Laporan harus sangat panjang dan mendetail, minimal 800 kata."
                    )
                },
                {
                    "role": "user",
                    "content": prompt_content,
                }
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.6,
            max_tokens=3000,
        )

        saran_teks = chat_completion.choices[0].message.content

        if not saran_teks:
            return jsonify({'status': 'error', 'message': 'AI sedang sibuk, coba lagi nanti.'}), 200

        return jsonify({'status': 'success', 'saran': saran_teks})

    except Exception as e:
        print(f"ERROR GROQ: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)