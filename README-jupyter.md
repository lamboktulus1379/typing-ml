# 📊 Jupyter Notebook: Data Validation & Offline Experimentation

Repositori ini berisi Jupyter Notebook yang berfungsi sebagai alat **Eksplorasi Data (EDA)** dan **Pembuatan Visualisasi** untuk keperluan validasi Bab 4 Skripsi.

⚠️ **PENTING (Konteks MLOps):** Notebook ini **bukan** bagian dari *pipeline* produksi *real-time*. Di sistem produksi (*live*), pembersihan data (IQR) dan inferensi model dijalankan secara otomatis di dalam memori oleh **Python FastAPI**. Notebook ini hanya bertindak sebagai "cermin" (Read-Only) untuk memvisualisasikan apa yang terjadi di dalam FastAPI tersebut agar dapat didokumentasikan ke dalam naskah skripsi.

## 🔄 Mekanisme Sinkronisasi Data (Tanpa Duplikasi)
Untuk mencegah duplikasi data (seperti format CSV yang tercecer), Notebook ini dikonfigurasi untuk melakukan **Direct Read-Only Query** ke SQL Server. 

Referensi query inti yang disejajarkan dengan alur retraining MLOps didokumentasikan secara terpisah di `docs/mlops-training-query-reference.md` agar perubahan query tidak hanya tersimpan di notebook.

**Alur Sinkronisasi:**
1. Pengguna mengetik di Angular (Frontend).
2. .NET 10 menyimpan data telemetri mentah ke SQL Server.
3. Jupyter Notebook mengeksekusi `pandas.read_sql()` untuk menarik data terbaru secara langsung dari SQL Server.
4. Notebook menerapkan ulang logika IQR (sama persis dengan FastAPI) untuk membandingkan metrik "Sebelum vs Sesudah IQR" dan men-generate grafik.

## 🛠 Cara Penggunaan (Manual Trigger)
1. Pastikan SQL Server database Anda sedang berjalan.
2. Buka file `visualize_outliers.ipynb` di VS Code atau Jupyter Lab.
3. Sesuaikan `CONNECTION_STRING` pada cell pertama dengan kredensial database lokal Anda.
4. Klik **"Run All"**.
5. Grafik *Scatter Plot* dan *Boxplot* akan muncul di cell paling bawah. Anda bisa klik kanan pada gambar -> **Save Image As...** untuk dimasukkan ke dokumen Word (Bab 4).
