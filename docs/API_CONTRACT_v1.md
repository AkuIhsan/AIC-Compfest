# API Contract v0.1 — Endpoint Analyze

Dokumen ini adalah kontrak bareng antara Backend dan ML untuk endpoint `/api/v1/analyze`. Karena tim kita cuma 4 orang dan masing-masing pegang bagian yang beda (ML, Data, Backend, Frontend), dokumen ini ditulis biar semua orang, bukan cuma yang bikin modelnya,  bisa ngerti alur request-response-nya, termasuk field apa aja yang dikirim, format response, dan aturan main di balik `zone`/`decision`.

Kalau ada bagian yang masih berubah-ubah atau belum final, sudah ditandai di bagian catatan paling bawah.

---

## Endpoint

```
POST /api/v1/analyze
Content-Type: application/json
```

## Request

```json
{
  "batch_id": "B00123",
  "timestamp": "2026-07-23T23:30:00",
  "machine_id": "M2",
  "shift": "malam",
  "operator_id": "OP03",
  "material_batch": "B",
  "curing_time_actual": 24.5,
  "temp_actual": 81.2,
  "pressure_actual": 106.0,
  "manual_override": true,
  "visual_check_done": false
}
```

Semua field di atas wajib diisi oleh client, kecuali kalau backend memang diatur untuk generate `batch_id` dan `timestamp` sendiri di sisi server. **`[OPEN]`**: ini masih belum diputusin, behavior mana yang mau dipakai.

## Response sukses (ada deviasi SOP)

```json
{
  "batch_id": "B00123",
  "timestamp": "2026-07-23T23:30:00",
  "sop_check": {
    "has_deviation": true,
    "deviation_count": 3,
    "deviations": [
      {
        "parameter": "curing_time",
        "status": "below_sop",
        "actual_value": 24.5,
        "sop_min": 28.0,
        "sop_max": 32.0,
        "deviation": -3.5,
        "unit": "minute"
      },
      {
        "parameter": "manual_override",
        "status": "violation",
        "actual_value": true,
        "expected_value": false
      },
      {
        "parameter": "visual_check",
        "status": "skipped",
        "actual_value": false,
        "expected_value": true
      }
    ]
  },
  "risk_analysis": {
    "risk_score": 0.92,
    "zone": "alarm",
    "decision": "alarm",
    "explanation_status": "success",
    "sop_risk_contributors": [
      {
        "id": "curing_time_below_sop",
        "label": "Curing time below SOP minimum",
        "contribution": 0.24
      },
      {
        "id": "manual_override",
        "label": "Manual override used",
        "contribution": 0.19
      }
    ],
    "context_contributors": [
      {
        "id": "mesin_M2",
        "label": "Machine M2",
        "contribution": 0.11
      },
      {
        "id": "shift_malam",
        "label": "Night shift",
        "contribution": 0.09
      },
      {
        "id": "material_B",
        "label": "Material batch B",
        "contribution": 0.06
      }
    ],
    "model_version": "pending"
  }
}
```

> Angka `risk_score` dan `contribution` di contoh di atas cuma ilustrasi, bukan angka asli.
>
> Pemisahan `sop_risk_contributors` vs `context_contributors` ini ngikutin hasil kategorisasi SHAP terbaru dari pipeline ML:
> * **`sop_risk_contributors`**: driver yang memang berkaitan langsung dengan pelanggaran/deviasi SOP.
> * **`context_contributors`**: faktor kontekstual seperti mesin, shift, atau material. Ini bukan pelanggaran SOP, tapi tetap ikut memengaruhi skor risiko dari model.

## Nilai enum

```
zone:               monitor | review | alarm
decision:           monitor | review | alarm       
status (sop_check): within_sop | below_sop | above_sop | violation | skipped
explanation_status: success | no_positive_driver | unexplained_high_risk
```

Penjelasan singkat tiap nilai `explanation_status`:

* **`success`**: dipakai kalau model menghasilkan contributor positif yang cukup untuk menjelaskan skor risikonya.
* **`no_positive_driver`**: dipakai kalau model gak menemukan driver risiko yang cukup signifikan untuk batch tersebut. Biasanya muncul di zona `monitor`, dengan `sop_risk_contributors` dan `context_contributors` sama-sama kosong.
* **`within_sop`** kemungkinan besar gak akan muncul sebagai item di array `deviations`, karena array ini hanya diisi kalau memang ada pelanggaran. Nilai ini tetap dicantumkan di enum karena konsep SOP checkernya memang kenal sama status ini, tapi backend gak perlu mengirim status `within_sop` di response manapun.

## Threshold model saat ini

```
risk_score < 0.39           -> monitor
0.39 <= risk_score < 0.89   -> review
risk_score >= 0.89          -> alarm
```

Threshold ini bukan angka sembarangan, hasil langsung dari evaluasi pipeline: `threshold_review = 0.39` dipilih di alert rate ≤30%, sedangkan `threshold_alarm = 0.89` dipilih di rentang alert rate 1–5% dengan jumlah alert ≥50. Kalau modelnya nanti diretrain ulang, kedua angka ini berpotensi berubah lagi.

Satu hal yang perlu diingat: `risk_score` itu skor risiko dari model, bukan probabilitas kegagalan yang sudah terkalibrasi. Jadi nilai `0.92` gak otomatis berarti peluang gagalnya 92%, dia cuma menunjukkan seberapa tinggi risiko relatif menurut model, bukan angka probabilitas yang bisa dibaca apa adanya.

**Aturan override:**

Kalau `explanation_status = "unexplained_high_risk"`, maka `decision` diturunkan jadi `"review"` — meskipun `risk_score`-nya sebenarnya sudah masuk zona `"alarm"`.

Tapi kalau `zone = "alarm"` **dan** ada contributor positif yang cukup di `sop_risk_contributors`/`context_contributors` (misalnya kombinasi dari mesin, shift, dan material), maka `decision` tetap `"alarm"`. Jadi status alarm tidak otomatis turun jadi `review`, hanya turun kalau memang tidak ada penjelasan yang cukup kuat di baliknya.

## Response tanpa deviasi SOP, model juga tenang

```json
{
  "batch_id": "B00124",
  "timestamp": "2026-07-23T23:40:00",
  "sop_check": {
    "has_deviation": false,
    "deviation_count": 0,
    "deviations": []
  },
  "risk_analysis": {
    "risk_score": 0.08,
    "zone": "monitor",
    "decision": "monitor",
    "explanation_status": "no_positive_driver",
    "sop_risk_contributors": [],
    "context_contributors": [],
    "model_version": "pending"
  }
}
```

## Kasus yang agak janggal: SOP-nya bersih, tapi modelnya curiga

Ini kasus ketika model memberikan `risk_score` di zona `alarm`, tetapi tidak ada contributor positif yang cukup untuk menjelaskan hasil tersebut. Kondisi ini tidak ditentukan hanya dari ada atau tidaknya pelanggaran SOP — kalau faktor context seperti mesin, shift, atau material masih bisa menjelaskan risikonya, keputusan tetap boleh `"alarm"`. Keputusan baru diturunkan menjadi `"review"` ketika kedua daftar contributornya kosong, dan batch ditandai perlu inspeksi manual.

Supaya jelas, aturan override-nya ditulis eksplisit begini:

```
Jika:
  zone = "alarm"
  dan sop_risk_contributors = []
  dan context_contributors = []

Maka:
  explanation_status = "unexplained_high_risk"
  decision = "review"
```

```json
{
  "batch_id": "B00125",
  "timestamp": "2026-07-23T23:50:00",
  "sop_check": {
    "has_deviation": false,
    "deviation_count": 0,
    "deviations": []
  },
  "risk_analysis": {
    "risk_score": 0.91,
    "zone": "alarm",
    "decision": "review",
    "explanation_status": "unexplained_high_risk",
    "sop_risk_contributors": [],
    "context_contributors": [],
    "message": "Risiko model tinggi tetapi tidak ada deviasi SOP yang cukup menjelaskan. Perlu inspeksi manual.",
    "model_version": "pending"
  }
}
```

## Response error

```json
{
  "error": {
    "code": "INVALID_INPUT",
    "message": "Field temp_actual harus berupa angka",
    "field": "temp_actual"
  }
}
```

## Alur di backend

```
request mentah
→ validasi input
→ sop_checker.py
→ f_engineer.py
→ model predict_proba
→ SHAP extraction + kategorisasi (sop_risk vs context)
→ gabungkan response
```

---

## Catatan sebelum dokumen ini dipakai

* Struktur `sop_check` sudah final dan dasarnya rule-based, jadi aman dijadikan pegangan.
* Struktur `risk_analysis` di atas sudah mengikuti pemisahan `sop_risk_contributors` / `context_contributors` dari pipeline ML terbaru, tapi ini masih draft, belum di-publish, jadi masih mungkin berubah kalau ada penyesuaian dari sisi model artifact.
* Threshold `0.39` dan `0.89` berlaku untuk model versi saat ini dan sudah tervalidasi dari hasil evaluasi pipeline, tapi tetap bisa berubah kalau modelnya di-retrain.
* Backend sebaiknya ambil threshold ini dari configuration artifact model, bukan hardcode ke source code. Angka `0.39` dan `0.89` di dokumen ini cuma buat dokumentasi konfigurasi model versi saat ini, bukan nilai yang harus ditulis permanen di kode.
* Semua angka `risk_score` dan `contribution` di contoh-contoh atas hanya ilustrasi, bukan data real dari model.
* Tolong jangan masukkan `scenario_ground_truth.csv` ke runtime atau dijadikan fitur model, file itu murni untuk evaluasi/audit, bukan bahan training atau inference.