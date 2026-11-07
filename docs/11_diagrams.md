# 11 — Technical Diagrams (Mermaid)

These render on GitHub, in VS Code (Markdown Preview Mermaid Support) and at https://mermaid.live (paste a block to export PNG/SVG for slides).

## 1. System architecture
```mermaid
flowchart TB
    subgraph Offline["Offline training pipeline (scripts/, run.py pipeline)"]
        RAW[Raw CSV + PDF<br/>read-only, SHA-256] --> ING[Ingestion & validation]
        ING --> PQ[(Typed Parquet<br/>231 segments)]
        PQ --> LAB[Flags & targets<br/>frozen · uncertain · y_detect · y_warn]
        LAB --> SPL[Chronological splits<br/>+12 h embargo]
        SPL --> FE[1-min features<br/>132 past-only]
        FE --> ML[Supervised models<br/>LR · DT · RF · SVM · XGB]
        FE --> AD[Anomaly models<br/>IF · LOF · OCSVM · robust-z]
        SPL --> DL[DL experiments<br/>LSTM · 1D-CNN]
        ML & AD & DL --> EV[Evaluation<br/>row + event level]
        EV --> ART[(Artifacts + metadata<br/>models/ · MLflow)]
    end
    subgraph Online["Online application"]
        API[FastAPI backend] --> PRED[Inference pipeline<br/>same feature code]
        PRED --> HE[Health & risk engine]
        PRED --> SH[SHAP / deviations]
        API <--> DB[(SQLite)]
        UI[Streamlit dashboard<br/>10 pages] <--> API
        API --> REP[CSV / PDF reports]
    end
    ART --> PRED
```
**Explanation:** the offline part turns raw data into versioned artifacts; the online part loads them once and serves assessments.
Both use the same `src/predictops` code, so training and inference preprocessing can't diverge.

## 2. Data flow diagram (Level 1)
```mermaid
flowchart LR
    U((Maintenance<br/>engineer)) -- CSV upload / sample / replay time --> P1[1. Validate & store data]
    P1 -- file reference --> D1[(datasets)]
    P1 --> P2[2. Clean & build features]
    P2 --> P3[3. Score: anomaly + signature]
    P3 --> P4[4. Health, risk, reasons]
    P4 --> D2[(predictions · alerts)]
    D2 --> P5[5. Visualise / explain / report]
    P5 -- dashboard, PDF, CSV --> U
    U -- acknowledge alert --> D2
    D3[(model artifacts)] --> P3
```

## 3. ML pipeline
```mermaid
flowchart LR
    A[10 s readings] --> B[sort · dedupe · validate]
    B --> C[segment at gaps > 10 min]
    C --> D[flag frozen / uncertain]
    D --> E[1-min aggregation<br/>(t−1, t]]
    E --> F[rolling 5 m / 30 m / 2 h<br/>lags · baselines · duty cycle]
    F --> G{training only}
    G --> H[variance + correlation filter<br/>XGB-gain top 33]
    G --> I[imputer + scaler fit]
    H & I --> J[models]
    J --> K[threshold on OOF validation]
    K --> L[select BEFORE test]
    L --> M[single test on event 4]
```

## 4. Use-case diagram
```mermaid
flowchart LR
    Eng((Maintenance<br/>engineer))
    Ana((Data scientist))
    subgraph PredictOps
        UC1([Upload & validate sensor data])
        UC2([Run batch / individual assessment])
        UC3([Monitor sensors & health])
        UC4([Replay historical period])
        UC5([View anomaly alerts])
        UC6([Explain a prediction])
        UC7([Acknowledge alert / add note])
        UC8([Download CSV / PDF report])
        UC9([Review model performance])
        UC10([Check data & prediction drift])
        UC11([Retrain pipeline])
    end
    Eng --- UC1 & UC2 & UC3 & UC4 & UC5 & UC6 & UC7 & UC8
    Ana --- UC9 & UC10 & UC11 & UC6
```

## 5. Sequence diagram — batch prediction
```mermaid
sequenceDiagram
    actor User
    participant UI as Streamlit
    participant API as FastAPI
    participant SVC as Services
    participant INF as Inference pipeline
    participant DB as SQLite
    User->>UI: choose CSV, click "Run"
    UI->>API: POST /datasets/upload (file)
    API->>SVC: size/type check, validate_frame
    SVC->>DB: INSERT dataset (file reference)
    API-->>UI: 201 {id, validation}
    UI->>API: POST /predict {dataset_id}
    API->>INF: predict_raw(df)
    INF->>INF: clean → features → OCSVM + XGBoost → health engine
    INF-->>API: per-minute results
    API->>DB: INSERT run, predictions, alerts
    API-->>UI: {run, latest, alerts}
    UI->>API: POST /explain {run_id, minute}
    API->>INF: SHAP + deviations
    API-->>UI: contributions
    UI-->>User: charts, reasons, recommendations
```

## 6. Database ER diagram
```mermaid
erDiagram
    EQUIPMENT ||--o{ DATASETS : has
    EQUIPMENT ||--o{ PREDICTION_RUNS : has
    DATASETS ||--o{ PREDICTION_RUNS : "assessed by"
    PREDICTION_RUNS ||--o{ PREDICTIONS : contains
    PREDICTION_RUNS ||--o{ ALERTS : raises
    EQUIPMENT { int id PK  string name }
    DATASETS { int id PK  int equipment_id FK  string stored_path  string sha256  json validation }
    PREDICTION_RUNS { int id PK  int dataset_id FK  string model_version_detector  string model_version_anomaly  json summary }
    PREDICTIONS { int id PK  int run_id FK  datetime timestamp  float health_score  string health_status  string failure_risk }
    ALERTS { int id PK  int run_id FK  datetime start_time  string severity  bool acknowledged  text note }
    MODEL_VERSIONS { int id PK  string version  string artifact }
```
**Explanation:** runs record which model versions produced them (traceability); sensor files are referenced, not duplicated.

## 7. Deployment architecture
```mermaid
flowchart LR
    B[Browser] -- HTTPS :8501 --> D[dashboard container<br/>Streamlit]
    D -- HTTP + X-API-Key --> A[api container<br/>FastAPI + Uvicorn]
    A --> V[(volume: SQLite)]
    A --> M[(baked-in artifacts<br/>models · history · samples)]
    subgraph Host["Docker host · or Render (2 web services)"]
        D
        A
        V
        M
    end
```
**Explanation:** one image, two services (different start commands). Locally: `docker compose up`. Render: `render.yaml`
blueprint, with an auto-generated API key and a health check on `/health`.
