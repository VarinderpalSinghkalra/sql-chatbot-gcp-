# Spend Analytics AI (Natural Language → SQL)

An **AI-powered Spend Analytics system** that converts **natural language business questions** into **safe, validated BigQuery SQL**, executes them, and returns results via a **modern browser UI**.

This project is built using **Google Cloud Run / Cloud Functions (Gen2)**, **BigQuery**, and a **pure HTML + JavaScript frontend**.

---

## 🚀 Features

- 🧠 Natural Language → SQL (Spend, Count, Volume, Savings)
- 📊 Dynamic grouping (Business Unit, Category, Buyer, Supplier, etc.)
- 🛡️ SQL safety using AST validation (`sqlglot`)
- 🧹 Dirty-date tolerant (fail-open time filtering)
- 🌐 Browser-friendly API (CORS enabled)
- 🎨 Clean, modern UI (HTML + CSS + JS)
- ☁️ Cloud-native (Cloud Run / Cloud Functions Gen2)

---

## 🏗️ Architecture

```
Browser UI (HTML + JS)
        |
        |  POST /question
        v
Cloud Run / Cloud Functions (Flask API)
        |
        |  Generated SQL
        v
BigQuery (Spend Dataset)
        |
        v
Results + Metadata (JSON)
```

---

## 📂 Project Structure

```
.
├── main.py                  # Backend (Flask API)
├── index.html               # Frontend UI
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
└── docs/
    ├── lesson_spend_sql_bot.txt
    └── architecture_and_commands.txt
```

---

## 🧠 Example Prompts

```
Total spend
Total spend last 12 months
Total spend by business unit
Spend by category
Top suppliers by spend
Spend by buyer
```

---

## ⚙️ Backend Logic Highlights

### ✅ Safe Aggregation
Never returns NULL:
```sql
COALESCE(SUM(SAFE_CAST(amt_local AS NUMERIC)), 0)
```

### ✅ Fail-Open Date Logic
- Tries multiple date formats
- If no valid dates → ignores time filter
- Returns best possible business answer with warning

### ✅ Dynamic Prompt Handling
- Auto-detects metric (spend, count, volume)
- Auto-detects dimension
- Applies GROUP BY automatically

---

## ☁️ Deployment – Cloud Run

```bash
gcloud run deploy sqlbotmainup   --region us-central1   --source .   --platform managed
```

### Allow public access (for UI)
```bash
gcloud run services add-iam-policy-binding sqlbotmainup   --region us-central1   --member="allUsers"   --role="roles/run.invoker"
```

---

## ☁️ Deployment – Cloud Functions (Gen2)

```bash
gcloud functions deploy sqlbotmainup   --gen2   --runtime python311   --region us-central1   --source .   --entry-point entry_point   --trigger-http   --allow-unauthenticated
```

---

## 🌐 Run UI Locally (Cloud Shell)

```bash
nano index.html
python3 -m http.server 8080
```

Then:
```
Cloud Shell → Web Preview → Port 8080
```

---

## 🧪 Test API using curl

```bash
curl -X POST "https://SERVICE_URL"   -H "Content-Type: application/json"   -d '{"question":"Total spend"}'
```

---

## 📊 Data Source

- BigQuery Dataset: `conversational_demo`
- Table: `sample_spenddata`
- Source: CSV uploaded to GCS and loaded via `bq load`

---

## 🧑‍💼 Interview-Ready Summary

> I built an AI-driven spend analytics system that allows users to ask natural language questions via a web UI. The backend dynamically generates and validates SQL, handles dirty enterprise data gracefully, and executes analytics on BigQuery. The system is cloud-native, secure, browser-friendly, and production-ready.

---

## 🔮 Future Enhancements

- 📈 Charts (Bar / Pie)
- 🔝 Top-N queries
- 📥 Export results to CSV
- 🔐 Authentication (Firebase / IAP)
- 🧠 LLM-powered prompt understanding

---

## 👨‍💻 Author

**Varinder Pal Singh**  
Cloud & Data Engineer  
GCP | BigQuery | Cloud Run | DevOps | AI-driven Analytics

---

⭐ If you find this project useful, feel free to star the repo!
