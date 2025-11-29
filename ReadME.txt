

====================================================================================

ARCHITECTURE OVERVIEW
====================================================================================

USER → Cloud Function (Gen2 Python) → Vertex AI Gemini → BigQuery → Response to User

Components

Cloud Function Gen2 (Python 3.12)

Vertex AI Model: gemini-2.5-pro OR gemini-2.5-flash

BigQuery dataset + table (sales_data)

IAM roles for calling Vertex AI & BigQuery

REST API endpoint exposed via HTTPS

ASCII Architecture Diagram
            +---------------------------+
            |        User / Client      |
            +-------------+-------------+
                          |
                          v
            +---------------------------+
            |  Cloud Function (Gen2)    |
            |  main.py (your logic)     |
            +-------------+-------------+
                          |
                 NL Query | Generates SQL
                          v
            +---------------------------+
            |    Vertex AI Gemini       |
            |  (Publisher REST Model)   |
            +-------------+-------------+
                          |
                 SQL      v
            +---------------------------+
            |         BigQuery          |
            | conversational_demo.sales |
            +-------------+-------------+
                          |
                        Results
                          |
                          v
            +---------------------------+
            |         Response           |
            +---------------------------+

====================================================================================
2. GCP PROJECT INITIALIZATION

Replace <PROJECT_ID> before running commands.

Set project

gcloud config set project <PROJECT_ID>

Set region

gcloud config set functions/region us-central1

====================================================================================
3. ENABLE REQUIRED APIS

gcloud services enable
cloudfunctions.googleapis.com
run.googleapis.com
artifactregistry.googleapis.com
aiplatform.googleapis.com
bigquery.googleapis.com
bigqueryconnection.googleapis.com
iam.googleapis.com

====================================================================================
4. IAM CONFIGURATION

The Cloud Function uses this service account:
<PROJECT_NUMBER>-compute@developer.gserviceaccount.com

Assign required roles:

gcloud projects add-iam-policy-binding <PROJECT_ID>
--member="serviceAccount:<PROJECT_NUMBER>-compute@developer.gserviceaccount.com
"
--role="roles/aiplatform.user"

gcloud projects add-iam-policy-binding <PROJECT_ID>
--member="serviceAccount:<PROJECT_NUMBER>-compute@developer.gserviceaccount.com
"
--role="roles/bigquery.user"

gcloud projects add-iam-policy-binding <PROJECT_ID>
--member="serviceAccount:<PROJECT_NUMBER>-compute@developer.gserviceaccount.com
"
--role="roles/bigquery.dataViewer"

(Optional - if modifying tables)
gcloud projects add-iam-policy-binding <PROJECT_ID>
--member="serviceAccount:<PROJECT_NUMBER>-compute@developer.gserviceaccount.com
"
--role="roles/bigquery.dataEditor"

====================================================================================
5. BIGQUERY SETUP
Create Dataset

bq --location=US mk conversational_demo

Create Table

bq mk --table conversational_demo.sales_data
customer:STRING,amount:FLOAT,purchase_date:DATE

Insert Sample Data

bq query --use_legacy_sql=false '
INSERT INTO <PROJECT_ID>.conversational_demo.sales_data
(customer, amount, purchase_date)
VALUES ("Alice", 100, "2024-01-01"),("Bob", 200, "2024-02-01")
'

====================================================================================
6. PROJECT FILES
6.1 main.py (COPY–PASTE THIS EXACT FILE)

[PASTE YOUR main.py HERE – it will work exactly as-is]

6.2 requirements.txt

functions-framework>=3.5.0
Flask>=2.3.3
google-auth>=2.23.0
requests>=2.31.0
google-cloud-bigquery>=3.14.0

====================================================================================
7. DEPLOYMENT (CLOUD FUNCTIONS GEN2)

Run this from the same folder containing:
main.py
requirements.txt

gcloud functions deploy sqlgen-func
--gen2
--runtime=python312
--region=us-central1
--entry-point=app
--source=.
--trigger-http
--allow-unauthenticated
--set-env-vars PROJECT_ID=<PROJECT_ID>
--set-env-vars BQ_DATASET=conversational_demo
--set-env-vars BQ_TABLE=sales_data
--set-env-vars VERTEX_AI_LOCATION=global
--set-env-vars VERTEX_AI_MODEL=gemini-2.5-pro
--set-env-vars MAX_ROWS=1000

====================================================================================
8. TESTING YOUR CLOUD FUNCTION
Health Check

curl https://<FUNCTION_URL>

NL → SQL Query Test

curl -X POST -H "Content-Type: application/json"
-d '{"query":"show me customers who spent more than 100"}'
https://<FUNCTION_URL>

Expected Output

{
"results": [...],
"sql_used": "SELECT customer, amount, purchase_date FROM <PROJECT>.conversational_demo.sales_data WHERE amount > 100 LIMIT 1000",
"model": "gemini-2.5-pro",
"location": "global"
}

====================================================================================
9. ENVIRONMENT VARIABLES (REFERENCE)

PROJECT_ID=<PROJECT_ID>
BQ_DATASET=conversational_demo
BQ_TABLE=sales_data
VERTEX_AI_LOCATION=global
VERTEX_AI_MODEL=gemini-2.5-pro
MAX_ROWS=1000
COLUMN_SYNONYMS={"customer_name":"customer"} (optional)

====================================================================================
10. TROUBLESHOOTING
ERROR: "Permission denied" when calling Vertex AI

→ Grant: roles/aiplatform.user

ERROR: "Unrecognized name" in SQL

→ The app auto-corrects column names
→ Add synonyms using env var COLUMN_SYNONYMS

ERROR: BigQuery query execution failed

→ Ensure table exists
→ Ensure dataset is US region
→ Ensure service account has BigQuery roles

ERROR: Vertex AI model not found

→ Switch region to: us-central1
→ Or change model to: gemini-2.5-flash

====================================================================================
11. OPTIONAL: ARCHITECTURE DIAGRAM (MERMAID)

flowchart TD
A(User Query) --> B(Cloud Function Python)
B --> C(Vertex AI Gemini)
C --> D(BigQuery)
D --> B
B --> E(Return JSON Response)

====================================================================================
12. FINAL FOLDER STRUCTURE
project-folder/
│
├── main.py
├── requirements.txt
├── project_setup.txt (THIS FILE)
└── README.md (optional)

====================================================================================
