# disaster_prediction
Project Name : [https://disasterprediction-efd9s3efgnap7wwoudzwjz.streamlit.app/]
# 🌊 Flood Risk Intelligence System

A complete Python + Streamlit ML-powered dashboard for predicting flood probability
across 20 environmental and infrastructure risk factors.

---

## 📁 Project Structure

```
flood_project/
│
├── app.py               ← Main Streamlit dashboard (run this)
├── style.css            ← Custom dark-theme CSS (auto-loaded by app.py)
├── requirements.txt     ← Python dependencies
├── README.md            ← This file
│
└── Notebook/
    ├── scaler.pkl       ← Your trained StandardScaler  (place here)
    └── model.pkl        ← Your trained ML model        (place here)
```

---

## ⚙️ Setup & Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Add your model files

Copy your trained model artifacts into the `Notebook/` folder:

```
flood_project/
  Notebook/
    scaler.pkl   ← StandardScaler fitted on your training data
    model.pkl    ← LinearRegression (or any sklearn regressor)
```

> **Without model files:** The app runs in **Demo Mode** using a
> weighted-average formula that mirrors your original JS logic.

### 3. Run the app

```bash
streamlit run app.py
```

The browser opens automatically at **http://localhost:8501**

---

## 🔑 Feature Keys (must match your model training columns)

| Key | Description |
|-----|-------------|
| `MonsoonIntensity` | Seasonal monsoon rainfall intensity |
| `TopographyDrainage` | Natural terrain drainage capability |
| `ClimateChange` | Long-term climate shift impact |
| `Landslides` | Slope instability & soil erosion |
| `CoastalVulnerability` | Storm surges & sea level rise |
| `Watersheds` | Upstream catchment health |
| `WetlandLoss` | Natural flood buffer destruction |
| `Siltation` | River bed silt buildup |
| `AgriculturalPractices` | Farming runoff impact |
| `Deforestation` | Tree cover loss |
| `RiverManagement` | River channel maintenance quality |
| `DamsQuality` | Dam structural integrity |
| `DrainageSystems` | Urban drainage network adequacy |
| `DeterioratingInfrastructure` | Age/decay of flood-control structures |
| `Urbanization` | Impervious surface expansion |
| `Encroachments` | Illegal construction on floodplains |
| `IneffectiveDisasterPreparedness` | Emergency response readiness |
| `PopulationScore` | People per sq km in flood zones |
| `InadequatePlanning` | Zoning & land-use policy gaps |
| `PoliticalFactors` | Governance & policy implementation |

---

## 📊 Risk Classification

| Score (1–10) | Risk Level | Action |
|:---:|:---:|---|
| 1.0 – 3.9 | 🟢 Low | Monitor periodically |
| 4.0 – 6.9 | 🟡 Medium | Increase vigilance |
| 7.0 – 10.0 | 🔴 High | Immediate action required |

---

## 🧠 Scoring Logic

**With model loaded:**
```
features → scaler.transform() → model.predict() → normalize to 1-10
normalized = 1 + 9 × (raw - score_min) / (score_max - score_min)
```

**Demo Mode (no model files):**
```
score = 0.55 × env_avg + 0.45 × infra_avg + 0.08 × peak_factor
```

---

## 📈 Dashboard Sections

1. **Risk Factor Controls** — 20 sliders (1–10) in Environmental / Infrastructure tabs
2. **Flood Risk Score** — Gauge chart with animated prediction
3. **Category Contribution** — Donut chart: Environmental vs Infrastructure split
4. **Risk Profile Radar** — Spider chart of 10 key factors
5. **All 20 Factors Bar** — Horizontal ranked bar chart (color-coded by risk)
6. **Factor Value Scatter** — Bubble chart by category and priority
7. **Prediction History** — Area trend chart across multiple runs
8. **Stat Cards** — Env avg, Infra avg, Peak factor, Total runs
9. **Factor Heatmap** — 20-cell color-coded grid

---

## 🛠️ Customization

- **Change model path:** Edit `SCALER_PATH` and `MODEL_PATH` at the top of `app.py`
- **Change scoring weights:** Edit `fallback_score()` in `app.py`
- **Change risk thresholds:** Edit `get_risk()` in `app.py`
- **Styling:** All visual changes go in `style.css`
