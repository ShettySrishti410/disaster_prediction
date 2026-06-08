import pickle, os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ══════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Flood Risk Intelligence System",
                   page_icon="🌊", layout="wide",
                   initial_sidebar_state="collapsed")

def load_css():
    p = os.path.join(os.path.dirname(__file__), "style.css")
    if os.path.exists(p):
        with open(p) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
load_css()

# ── FACTORS ──────────────────────────────────────────────────────
ENV_FACTORS = [
    ("MonsoonIntensity",      "🌧️ Monsoon Intensity",       "Intensity of seasonal monsoon rainfall"),
    ("TopographyDrainage",    "⛰️ Topography & Drainage",    "Natural terrain drainage capability"),
    ("ClimateChange",         "🌡️ Climate Change",           "Long-term climate shift impact"),
    ("Landslides",            "🪨 Landslide Risk",            "Slope instability & soil erosion"),
    ("CoastalVulnerability",  "🌊 Coastal Vulnerability",    "Exposure to storm surges & sea level rise"),
    ("Watersheds",            "💧 Watershed Condition",       "Upstream catchment health"),
    ("WetlandLoss",           "🌿 Wetland Loss",              "Destruction of natural flood buffers"),
    ("Siltation",             "🏔️ Siltation Level",          "River bed & channel silt buildup"),
    ("AgriculturalPractices", "🌾 Agricultural Impact",       "Farming practices affecting runoff"),
    ("Deforestation",         "🌳 Deforestation Level",       "Tree cover loss increasing runoff"),
]
INFRA_FACTORS = [
    ("RiverManagement",                 "🏞️ River Management",      "Quality of river channel maintenance"),
    ("DamsQuality",                     "🏗️ Dam Quality",           "Structural integrity of dams"),
    ("DrainageSystems",                 "🚰 Drainage Systems",       "Urban drainage network adequacy"),
    ("DeterioratingInfrastructure",     "🏚️ Deteriorating Infra",   "Age & decay of flood-control structures"),
    ("Urbanization",                    "🏙️ Urbanization",          "Impervious surface expansion"),
    ("Encroachments",                   "🚧 Encroachments",          "Illegal construction on floodplains"),
    ("IneffectiveDisasterPreparedness", "⚠️ Disaster Preparedness", "Emergency response readiness"),
    ("PopulationScore",                 "👥 Population Density",     "People per sq km in flood zones"),
    ("InadequatePlanning",              "📋 Urban Planning",         "Zoning & land-use policy gaps"),
    ("PoliticalFactors",                "🏛️ Political Factors",      "Governance & policy implementation"),
]
ALL_FACTORS = ENV_FACTORS + INFRA_FACTORS
ALL_KEYS    = [f[0] for f in ALL_FACTORS]

# ── MODEL ────────────────────────────────────────────────────────
SCALER_PATH = os.path.join(os.path.dirname(__file__), "Notebook", "scaler.pkl")
MODEL_PATH  = os.path.join(os.path.dirname(__file__), "Notebook", "model.pkl")

def load_model_and_scaler():
    try:
        with open(SCALER_PATH,"rb") as f: scaler = pickle.load(f)
        with open(MODEL_PATH,"rb")  as f: model  = pickle.load(f)
        return scaler, model
    except FileNotFoundError: return None, None
    except Exception as e:
        st.warning(f"Model load error: {e}"); return None, None

def predict_flood(features, scaler, model):
    try:
        x = pd.DataFrame(features)
        return float(np.squeeze(model.predict(scaler.transform(x))))
    except Exception as e:
        st.error(f"Prediction error: {e}"); return None

def normalize_score(raw, scaler, model):
    s_min = predict_flood({k:[1]  for k in ALL_KEYS}, scaler, model)
    s_max = predict_flood({k:[10] for k in ALL_KEYS}, scaler, model)
    if s_min is not None and s_max is not None and s_max != s_min:
        norm = 1 + 9*(raw-s_min)/(s_max-s_min)
    else:
        norm = raw
    return float(np.clip(norm,1.0,10.0))

def fallback_score(values):
    ea = np.mean([values[k] for k,*_ in ENV_FACTORS])
    ia = np.mean([values[k] for k,*_ in INFRA_FACTORS])
    pk = max(values.values())
    return float(np.clip(ea*0.55+ia*0.45+pk*0.08,1.0,10.0))

def get_risk(s):
    if s < 4:  return dict(label="LOW RISK",    emoji="🟢",color="#22c55e",css="low",
                           action="✅ Conditions relatively safe. Continue monitoring and maintain preparedness.")
    if s < 7:  return dict(label="MEDIUM RISK", emoji="🟡",color="#f59e0b",css="medium",
                           action="⚠️ Moderate risk. Increase monitoring, review disaster preparedness plans.")
    return         dict(label="HIGH RISK",       emoji="🔴",color="#ef4444",css="high",
                        action="🚨 HIGH RISK! Immediate action required. Alert authorities, prepare evacuation.")

# ── CHART BASE LAYOUT ─────────────────────────────────────────────
def CL(h=None, title=None, title_color="#a78bfa"):
    d = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
             font=dict(color="#e2e8f0", size=13, family="Segoe UI, sans-serif"),
             margin=dict(l=15,r=15,t=45,b=15))
    if h:     d["height"] = h
    if title: d["title"]  = dict(text=f"<b>{title}</b>",
                                  font=dict(color=title_color,size=15,family="Segoe UI, sans-serif"),
                                  x=0.01)
    return d

# ══════════════════════════════════════════════════════════════════
# CHARTS
# ══════════════════════════════════════════════════════════════════
def gauge_chart(score):
    risk = get_risk(score)
    fig  = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        delta=dict(reference=5, valueformat=".2f",
                   increasing=dict(color="#ef4444"),
                   decreasing=dict(color="#22c55e")),
        number=dict(font=dict(color=risk["color"],size=52,
                               family="Segoe UI, sans-serif"), suffix=" /10",
                    valueformat=".2f"),
        gauge=dict(
            axis=dict(range=[1,10], tickwidth=2, tickcolor="#94a3b8",
                      tickfont=dict(color="#e2e8f0",size=13), nticks=10),
            bar=dict(color=risk["color"], thickness=0.3),
            bgcolor="rgba(0,0,0,0)", borderwidth=0,
            steps=[dict(range=[1,4],  color="rgba(34,197,94,0.18)"),
                   dict(range=[4,7],  color="rgba(245,158,11,0.18)"),
                   dict(range=[7,10], color="rgba(239,68,68,0.18)")],
            threshold=dict(line=dict(color="white",width=4),thickness=0.8,value=score),
        ),
    ))
    fig.update_layout(**CL(260,"🌊 Flood Risk Score", risk["color"]))
    return fig

def radar_chart(values):
    keys   = ["MonsoonIntensity","ClimateChange","CoastalVulnerability","Deforestation",
               "Siltation","DamsQuality","DrainageSystems","Urbanization",
               "IneffectiveDisasterPreparedness","InadequatePlanning"]
    labels = ["Monsoon","Climate","Coastal","Deforest",
               "Siltation","Dams","Drainage","Urban","Disaster","Planning"]
    vals   = [values[k] for k in keys]+[values[keys[0]]]
    lbls   = labels+[labels[0]]
    fig = go.Figure(go.Scatterpolar(
        r=vals, theta=lbls, fill="toself",
        fillcolor="rgba(99,102,241,0.25)",
        line=dict(color="#818cf8",width=2.5),
        marker=dict(color="#c7d2fe",size=8),
        name="Risk Profile",
    ))
    fig.update_layout(**CL(320,"🕸️ Risk Profile Radar — 10 Key Factors"),
        polar=dict(bgcolor="rgba(0,0,0,0)",
                   radialaxis=dict(visible=True,range=[0,10],
                                   gridcolor="rgba(99,102,241,0.2)",
                                   tickfont=dict(color="#e2e8f0",size=11),
                                   linecolor="rgba(99,102,241,0.3)"),
                   angularaxis=dict(gridcolor="rgba(99,102,241,0.15)",
                                    tickfont=dict(color="#f1f5f9",size=13,
                                                  family="Segoe UI, sans-serif"))))
    return fig

def bar_chart(values):
    data = sorted([{"name":lbl.split(" ",1)[-1],"value":values[k],
                    "cat":"env" if any(k==f[0] for f in ENV_FACTORS) else "infra"}
                   for k,lbl,_ in ALL_FACTORS],
                  key=lambda x:x["value"],reverse=True)
    colors = ["#ef4444" if d["value"]>=7 else "#f59e0b" if d["value"]>=4 else "#22c55e"
              for d in data]
    fig = go.Figure(go.Bar(
        x=[d["value"] for d in data], y=[d["name"] for d in data],
        orientation="h", marker=dict(color=colors,opacity=0.9,
                                     line=dict(color="rgba(255,255,255,0.15)",width=0.5)),
        text=[f"<b>{d['value']}</b>" for d in data], textposition="outside",
        textfont=dict(size=13,color="#f1f5f9"),
    ))
    fig.update_layout(**CL(380,"📊 All 20 Risk Factors — Ranked"),
        xaxis=dict(range=[0,12],gridcolor="rgba(148,163,184,0.1)",
                   tickfont=dict(color="#e2e8f0",size=12),showgrid=True),
        yaxis=dict(tickfont=dict(color="#f1f5f9",size=12),autorange="reversed",
                   ticklabelstandoff=4),
        bargap=0.25)
    return fig

def donut_chart(env_avg, infra_avg):
    fig = go.Figure(go.Pie(
        labels=["Environmental","Infrastructure"],
        values=[round(env_avg,2), round(infra_avg,2)],
        hole=0.55, marker=dict(colors=["#3b82f6","#f59e0b"],
                                line=dict(color="#0f172a",width=2)),
        textfont=dict(size=14,color="#ffffff"),
        textinfo="label+percent",
    ))
    fig.update_layout(**CL(240,"🥧 Category Contribution Split"),
        legend=dict(font=dict(color="#f1f5f9",size=13),orientation="h",y=-0.1,x=0.1))
    return fig

def history_chart(history):
    if not history: return None
    runs   = [h["run"]   for h in history]
    scores = [h["score"] for h in history]
    envs   = [h["env"]   for h in history]
    infras = [h["infra"] for h in history]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=runs,y=scores,name="Total Score",mode="lines+markers",
        line=dict(color="#818cf8",width=3),marker=dict(size=9,color="#818cf8",
        line=dict(color="white",width=2)),fill="tozeroy",fillcolor="rgba(99,102,241,0.12)"))
    fig.add_trace(go.Scatter(x=runs,y=envs,name="Env Avg",mode="lines+markers",
        line=dict(color="#3b82f6",width=2,dash="dot"),marker=dict(size=6)))
    fig.add_trace(go.Scatter(x=runs,y=infras,name="Infra Avg",mode="lines+markers",
        line=dict(color="#f59e0b",width=2,dash="dot"),marker=dict(size=6)))
    fig.add_hline(y=4,line=dict(color="#22c55e",dash="dash",width=1.5),
        annotation_text="<b>Low/Med (4)</b>",annotation_font=dict(color="#22c55e",size=12))
    fig.add_hline(y=7,line=dict(color="#ef4444",dash="dash",width=1.5),
        annotation_text="<b>Med/High (7)</b>",annotation_font=dict(color="#ef4444",size=12))
    fig.update_layout(**CL(300,"📈 Prediction History — Score Trend"),
        yaxis=dict(range=[0,11],gridcolor="rgba(148,163,184,0.1)",
                   tickfont=dict(color="#e2e8f0",size=12)),
        xaxis=dict(gridcolor="rgba(148,163,184,0.1)",tickfont=dict(color="#e2e8f0",size=12)),
        legend=dict(font=dict(color="#f1f5f9",size=13),orientation="h",y=-0.18,x=0))
    return fig

def scatter_chart(values):
    fig = go.Figure()
    for i,(k,lbl,_) in enumerate(ENV_FACTORS):
        v = values[k]
        fig.add_trace(go.Scatter(x=[v],y=[len(ALL_FACTORS)-i],mode="markers",
            marker=dict(color="#3b82f6",size=max(8,v*4),opacity=0.8,
                        line=dict(color="white",width=1.5)),
            name=lbl.split(" ",1)[-1],showlegend=False,
            hovertemplate=f"<b>{lbl.split(' ',1)[-1]}</b><br>Value: {v}/10<extra></extra>"))
    for i,(k,lbl,_) in enumerate(INFRA_FACTORS):
        v = values[k]
        fig.add_trace(go.Scatter(x=[v],y=[len(ENV_FACTORS)-i],mode="markers",
            marker=dict(color="#f59e0b",size=max(8,v*4),opacity=0.8,
                        line=dict(color="white",width=1.5)),
            name=lbl.split(" ",1)[-1],showlegend=False,
            hovertemplate=f"<b>{lbl.split(' ',1)[-1]}</b><br>Value: {v}/10<extra></extra>"))
    for nm,cl in [("Environmental","#3b82f6"),("Infrastructure","#f59e0b")]:
        fig.add_trace(go.Scatter(x=[None],y=[None],mode="markers",
            marker=dict(color=cl,size=12),name=nm))
    fig.update_layout(**CL(300,"🔵 Factor Value Bubble Chart"),
        xaxis=dict(range=[0,11],title=dict(text="<b>Factor Value (1–10)</b>",
                   font=dict(color="#e2e8f0",size=13)),
                   gridcolor="rgba(148,163,184,0.1)",tickfont=dict(color="#e2e8f0",size=12)),
        yaxis=dict(title=dict(text="<b>Priority Rank</b>",
                   font=dict(color="#e2e8f0",size=13)),tickfont=dict(color="#e2e8f0",size=11)),
        legend=dict(font=dict(color="#f1f5f9",size=13),orientation="h",y=-0.18))
    return fig

def waterfall_chart(values):
    env_keys = [k for k,*_ in ENV_FACTORS]
    inf_keys = [k for k,*_ in INFRA_FACTORS]
    env_avg  = round(np.mean([values[k] for k in env_keys]),2)
    inf_avg  = round(np.mean([values[k] for k in inf_keys]),2)
    peak     = max(values.values())
    final    = fallback_score(values)

    measures = ["relative","relative","relative","total"]
    x        = ["Environmental\nAvg","Infrastructure\nAvg","Peak\nBonus","Final\nScore"]
    y        = [env_avg*0.55, inf_avg*0.45, peak*0.08, final]
    colors   = ["#3b82f6","#f59e0b","#ef4444","#818cf8"]

    fig = go.Figure(go.Waterfall(
        measure=measures, x=x, y=y,
        text=[f"<b>{v:.2f}</b>" for v in y],
        textposition="outside",textfont=dict(color="#f1f5f9",size=14),
        connector=dict(line=dict(color="#475569",width=1.5)),
        increasing=dict(marker=dict(color="#3b82f6")),
        decreasing=dict(marker=dict(color="#ef4444")),
        totals=dict(marker=dict(color="#818cf8")),
    ))
    fig.update_layout(**CL(300,"🌊 Score Waterfall Breakdown"),
        xaxis=dict(tickfont=dict(color="#f1f5f9",size=13)),
        yaxis=dict(gridcolor="rgba(148,163,184,0.1)",tickfont=dict(color="#e2e8f0",size=12)))
    return fig

def env_vs_infra_bar(values):
    env_vals  = [values[k] for k,*_ in ENV_FACTORS]
    infra_vals= [values[k] for k,*_ in INFRA_FACTORS]
    env_lbl   = [lbl.split(" ",1)[-1] for _,lbl,_ in ENV_FACTORS]
    infra_lbl = [lbl.split(" ",1)[-1] for _,lbl,_ in INFRA_FACTORS]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="🌱 Environmental",x=env_lbl,y=env_vals,
        marker=dict(color="#3b82f6",opacity=0.88,line=dict(color="rgba(255,255,255,0.15)",width=0.5)),
        text=[f"<b>{v}</b>" for v in env_vals],textposition="outside",
        textfont=dict(color="#f1f5f9",size=12)))
    fig.add_trace(go.Bar(name="🏗️ Infrastructure",x=infra_lbl,y=infra_vals,
        marker=dict(color="#f59e0b",opacity=0.88,line=dict(color="rgba(255,255,255,0.15)",width=0.5)),
        text=[f"<b>{v}</b>" for v in infra_vals],textposition="outside",
        textfont=dict(color="#f1f5f9",size=12)))
    fig.update_layout(**CL(340,"🏗️ Environmental vs Infrastructure — Side by Side"),
        xaxis=dict(tickfont=dict(color="#f1f5f9",size=11),tickangle=-35),
        yaxis=dict(range=[0,12],gridcolor="rgba(148,163,184,0.1)",
                   tickfont=dict(color="#e2e8f0",size=12)),
        barmode="group",bargap=0.2,bargroupgap=0.05,
        legend=dict(font=dict(color="#f1f5f9",size=13),orientation="h",y=-0.25))
    return fig

def risk_gauge_breakdown(values):
    """Mini gauges for env avg, infra avg, peak"""
    env_avg  = round(np.mean([values[k] for k,*_ in ENV_FACTORS]),2)
    infra_avg= round(np.mean([values[k] for k,*_ in INFRA_FACTORS]),2)
    peak     = float(max(values.values()))

    fig = go.Figure()
    specs  = [(0.0,0.3),(0.37,0.67),(0.73,1.0)]
    labels = ["Env Avg","Infra Avg","Peak Factor"]
    vals   = [env_avg, infra_avg, peak]
    colors = ["#3b82f6","#f59e0b","#ef4444"]

    for (x0,x1),lbl,val,col in zip(specs,labels,vals,colors):
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=val,
            number=dict(font=dict(color=col,size=26),valueformat=".1f"),
            title=dict(text=f"<b>{lbl}</b>",font=dict(color="#e2e8f0",size=13)),
            gauge=dict(axis=dict(range=[1,10],tickfont=dict(color="#e2e8f0",size=10)),
                       bar=dict(color=col,thickness=0.3),bgcolor="rgba(0,0,0,0)",borderwidth=0,
                       steps=[dict(range=[1,4],color="rgba(34,197,94,0.15)"),
                              dict(range=[4,7],color="rgba(245,158,11,0.15)"),
                              dict(range=[7,10],color="rgba(239,68,68,0.15)")]),
            domain=dict(x=[x0,x1],y=[0,1])
        ))
    fig.update_layout(**CL(220,"🎯 Sub-Score Gauges"))
    return fig

def heatmap_html(values):
    cells=""
    for k,lbl,desc in ALL_FACTORS:
        v  = values[k]
        bg = ("#ef4444cc" if v>=7 else "#f59e0bcc" if v>=4 else "#22c55ecc")
        cells+=f'<div class="hm-cell" title="{lbl.split(" ",1)[-1]}: {v}&#10;{desc}" style="background:{bg};">{v}</div>'
    leg="".join([
        '<div class="hm-legend-item"><span style="background:#22c55ecc;"></span><b>Low (1–3)</b></div>',
        '<div class="hm-legend-item"><span style="background:#f59e0bcc;"></span><b>Medium (4–6)</b></div>',
        '<div class="hm-legend-item"><span style="background:#ef4444cc;"></span><b>High (7–10)</b></div>',
    ])
    return f"""
    <div class="hm-wrapper">
      <div class="section-label">🟥 Factor Heatmap — All 20 Factors</div>
      <div class="hm-grid">{cells}</div>
      <div class="hm-legend">{leg}</div>
    </div>"""

# ══════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════
for k,v in [("history",[]),("score",None),("do_reset",False)]:
    if k not in st.session_state: st.session_state[k]=v

if st.session_state.do_reset:
    st.session_state.do_reset=False
    st.session_state.score=None
    for _k in ALL_KEYS:
        if _k in st.session_state: del st.session_state[_k]

# ══════════════════════════════════════════════════════════════════
# ▌SECTION 1 — HEADER
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<div class="header-wrap">
  <div class="header-badge">🌊 FLOOD RISK INTELLIGENCE SYSTEM</div>
  <h1 class="header-title">Flood Probability Predictor</h1>
  <p class="header-sub">
    Set each risk factor from <strong>1 (safest)</strong> to <strong>10 (most dangerous)</strong>
    across <strong>20 environmental &amp; infrastructure dimensions</strong>
  </p>
</div>""", unsafe_allow_html=True)

scaler, model = load_model_and_scaler()
if scaler is None:
    st.markdown("""<div class="demo-banner">⚡ <strong>DEMO MODE</strong> —
    Place <code>Notebook/scaler.pkl</code> &amp; <code>Notebook/model.pkl</code>
    to enable real ML predictions. Currently using weighted-average formula.</div>""",
    unsafe_allow_html=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════
# ▌SECTION 2 — SLIDERS  +  GAUGE + DONUT  (side by side)
# ══════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">⚙️ SECTION 1 — Risk Factor Controls</div>', unsafe_allow_html=True)

col_sliders, col_right = st.columns([1.2, 0.8], gap="large")

with col_sliders:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    tab_env, tab_infra = st.tabs(["🌱 Environmental Factors (10)", "🏗️ Infrastructure Factors (10)"])
    values: dict[str, int] = {}
    with tab_env:
        for key,label,desc in ENV_FACTORS:
            values[key] = st.slider(label,1,10,1,help=desc,key=key)
    with tab_infra:
        for key,label,desc in INFRA_FACTORS:
            values[key] = st.slider(label,1,10,1,help=desc,key=key)
    if st.button("🔄 Reset All to Default (1)", use_container_width=True, key="reset"):
        st.session_state.do_reset=True; st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

with col_right:
    score    = st.session_state.score
    env_avg  = round(np.mean([values[k] for k,*_ in ENV_FACTORS]),2)
    infra_avg= round(np.mean([values[k] for k,*_ in INFRA_FACTORS]),2)
    peak     = int(max(values.values()))

    # Gauge
    st.markdown('<div class="card">', unsafe_allow_html=True)
    if score is not None:
        risk = get_risk(score)
        st.plotly_chart(gauge_chart(score),use_container_width=True,config={"displayModeBar":False})
        st.markdown(f'<div class="risk-box {risk["css"]}">{risk["emoji"]} {risk["label"]} &nbsp;|&nbsp; {score:.2f} / 10</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="action-box">{risk["action"]}</div>', unsafe_allow_html=True)
        c1,c2,c3 = st.columns(3)
        for col,(rng,lbl,css) in zip([c1,c2,c3],[("1–3.9","Low","low"),("4–6.9","Medium","medium"),("7–10","High","high")]):
            col.markdown(f'<div class="range-chip {css}"><b>{lbl}</b><br><small>{rng}</small></div>',unsafe_allow_html=True)
    else:
        st.markdown('<div class="waiting-box">🎯 Adjust sliders then click<br><strong>PREDICT FLOOD RISK</strong></div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Predict button
    if st.button("🔍  PREDICT FLOOD RISK",use_container_width=True,key="predict_btn",type="primary"):
        with st.spinner("🔄 Analyzing all 20 factors..."):
            if scaler and model:
                raw  = predict_flood({k:[values[k]] for k in ALL_KEYS},scaler,model)
                norm = normalize_score(raw,scaler,model) if raw is not None else fallback_score(values)
            else:
                norm = fallback_score(values)
            st.session_state.score=norm
            st.session_state.history.append({"run":f"#{len(st.session_state.history)+1}",
                "score":norm,"env":env_avg,"infra":infra_avg})
            st.rerun()

    # Donut
    st.markdown('<div class="card" style="margin-top:14px;">', unsafe_allow_html=True)
    st.plotly_chart(donut_chart(env_avg,infra_avg),use_container_width=True,config={"displayModeBar":False})
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════
# ▌SECTION 3 — STAT CARDS
# ══════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">📋 SECTION 2 — Live Statistics Summary</div>', unsafe_allow_html=True)

sc1,sc2,sc3,sc4,sc5 = st.columns(5, gap="small")
score_display = f"{score:.2f}" if score is not None else "—"
risk_lbl = get_risk(score)["label"] if score is not None else "NOT RUN"
stat_data = [
    (sc1,"🌱 Env. Average",   f"{env_avg:.1f}",  "/10","#3b82f6"),
    (sc2,"🏗️ Infra Average",  f"{infra_avg:.1f}","/10","#f59e0b"),
    (sc3,"🔺 Peak Factor",    str(peak),          "/10","#ef4444"),
    (sc4,"🌊 Flood Score",    score_display,      "/10","#818cf8"),
    (sc5,"🔁 Runs Done",      str(len(st.session_state.history)),"","#34d399"),
]
for col,lbl,val,unit,color in stat_data:
    col.markdown(f"""
    <div class="stat-card">
      <div class="stat-label">{lbl}</div>
      <div class="stat-value" style="color:{color};">{val}<span class="stat-unit">{unit}</span></div>
    </div>""", unsafe_allow_html=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════
# ▌SECTION 4 — MINI GAUGES
# ══════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">🎯 SECTION 3 — Sub-Score Breakdown Gauges</div>', unsafe_allow_html=True)
st.markdown('<div class="card">', unsafe_allow_html=True)
st.plotly_chart(risk_gauge_breakdown(values),use_container_width=True,config={"displayModeBar":False})
st.markdown("</div>", unsafe_allow_html=True)
st.markdown("---")

# ══════════════════════════════════════════════════════════════════
# ▌SECTION 5 — RADAR + WATERFALL
# ══════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">📡 SECTION 4 — Radar Profile & Score Waterfall</div>', unsafe_allow_html=True)
col_r, col_w = st.columns([1,1], gap="large")
with col_r:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.plotly_chart(radar_chart(values),use_container_width=True,config={"displayModeBar":False})
    st.markdown("</div>", unsafe_allow_html=True)
with col_w:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.plotly_chart(waterfall_chart(values),use_container_width=True,config={"displayModeBar":False})
    st.markdown("</div>", unsafe_allow_html=True)
st.markdown("---")

# ══════════════════════════════════════════════════════════════════
# ▌SECTION 6 — RANKED BAR
# ══════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">📊 SECTION 5 — All 20 Factors Ranked by Risk Value</div>', unsafe_allow_html=True)
st.markdown('<div class="card">', unsafe_allow_html=True)
st.plotly_chart(bar_chart(values),use_container_width=True,config={"displayModeBar":False})
st.markdown("</div>", unsafe_allow_html=True)
st.markdown("---")

# ══════════════════════════════════════════════════════════════════
# ▌SECTION 7 — ENV vs INFRA GROUPED BAR
# ══════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">🏗️ SECTION 6 — Environmental vs Infrastructure Comparison</div>', unsafe_allow_html=True)
st.markdown('<div class="card">', unsafe_allow_html=True)
st.plotly_chart(env_vs_infra_bar(values),use_container_width=True,config={"displayModeBar":False})
st.markdown("</div>", unsafe_allow_html=True)
st.markdown("---")

# ══════════════════════════════════════════════════════════════════
# ▌SECTION 8 — SCATTER + HISTORY
# ══════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">🔵 SECTION 7 — Bubble Chart & Prediction History</div>', unsafe_allow_html=True)
col_sc, col_hist = st.columns([1,1.2], gap="large")
with col_sc:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.plotly_chart(scatter_chart(values),use_container_width=True,config={"displayModeBar":False})
    st.markdown("</div>", unsafe_allow_html=True)
with col_hist:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    hf = history_chart(st.session_state.history)
    if hf:
        st.plotly_chart(hf,use_container_width=True,config={"displayModeBar":False})
    else:
        st.markdown('<div class="waiting-box">📉 Run predictions to build history trend</div>',unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
st.markdown("---")

# ══════════════════════════════════════════════════════════════════
# ▌SECTION 9 — HEATMAP
# ══════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">🟥 SECTION 8 — Factor Heatmap Grid</div>', unsafe_allow_html=True)
st.markdown(heatmap_html(values), unsafe_allow_html=True)
st.markdown("---")

# ══════════════════════════════════════════════════════════════════
# ▌SECTION 10 — DATA TABLE
# ══════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">📋 SECTION 9 — Full Factor Data Table</div>', unsafe_allow_html=True)
st.markdown('<div class="card">', unsafe_allow_html=True)
rows = []
for k,lbl,desc in ALL_FACTORS:
    v   = values[k]
    cat = "🌱 Environmental" if any(k==f[0] for f in ENV_FACTORS) else "🏗️ Infrastructure"
    rl  = "🔴 HIGH" if v>=7 else "🟡 MEDIUM" if v>=4 else "🟢 LOW"
    rows.append({"Factor":lbl.split(" ",1)[-1],"Category":cat,
                 "Score":v,"Risk Level":rl,"Description":desc})
df = pd.DataFrame(rows)
st.dataframe(df, use_container_width=True, hide_index=True,
    column_config={
        "Factor":     st.column_config.TextColumn("📌 Factor",width="medium"),
        "Category":   st.column_config.TextColumn("🗂️ Category",width="medium"),
        "Score":      st.column_config.ProgressColumn("🔢 Score (1–10)",min_value=1,max_value=10,format="%d"),
        "Risk Level": st.column_config.TextColumn("⚠️ Risk Level",width="small"),
        "Description":st.column_config.TextColumn("📝 Description",width="large"),
    })
st.markdown("</div>", unsafe_allow_html=True)
st.markdown("---")

# ══════════════════════════════════════════════════════════════════
# ▌SECTION 11 — PREDICTION HISTORY TABLE
# ══════════════════════════════════════════════════════════════════
if st.session_state.history:
    st.markdown('<div class="section-header">🔁 SECTION 10 — Prediction History Log</div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    hrows=[]
    for h in st.session_state.history:
        r=get_risk(h["score"])
        hrows.append({"Run":h["run"],"Total Score":round(h["score"],2),
                      "Risk Level":f"{r['emoji']} {r['label']}",
                      "Env Avg":h["env"],"Infra Avg":h["infra"]})
    hdf=pd.DataFrame(hrows)
    st.dataframe(hdf,use_container_width=True,hide_index=True,
        column_config={
            "Run":        st.column_config.TextColumn("🔁 Run",width="small"),
            "Total Score":st.column_config.ProgressColumn("🌊 Score",min_value=1,max_value=10,format="%.2f"),
            "Risk Level": st.column_config.TextColumn("⚠️ Risk Level"),
            "Env Avg":    st.column_config.NumberColumn("🌱 Env Avg",format="%.2f"),
            "Infra Avg":  st.column_config.NumberColumn("🏗️ Infra Avg",format="%.2f"),
        })
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("---")

# ── FOOTER ────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  🌊 <strong>Flood Risk Intelligence System</strong> &nbsp;·&nbsp;
  20 Factors &nbsp;·&nbsp; Normalized 1–10 Scale &nbsp;·&nbsp;
  <span>Place <code>Notebook/scaler.pkl</code> &amp; <code>Notebook/model.pkl</code> for real ML predictions</span>
</div>""", unsafe_allow_html=True)