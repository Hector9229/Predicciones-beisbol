import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# ==================== CONFIGURACIÓN ====================
st.set_page_config(
    page_title="Béisbol MLB | Análisis de Bateadores",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    [data-testid="stSidebar"] {display: none;}
    [data-testid="collapsedControl"] {display: none;}
    
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        color: #1a365d;
        margin-bottom: 0.2rem;
    }
    .main-subtitle {
        font-size: 1.1rem;
        color: #4a5568;
        margin-bottom: 1.8rem;
    }
    
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background-color: #f7fafc;
        padding: 6px;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 10px 16px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2b6cb0 !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# ==================== CARGA DE DATOS ====================
@st.cache_data
def cargar_datos():
    try:
        df = pd.read_excel("data/bateadores_mlb_2022_2025.xlsx", sheet_name="Bateadores", header=4)
    except:
        try:
            df = pd.read_excel("data/bateadores_mlb_2022_2025.xlsx", sheet_name="Bateadores")
        except:
            st.error("No se encontró el archivo de datos en la carpeta 'data'.")
            st.stop()
    df.columns = df.columns.str.strip()
    return df

df_raw = cargar_datos()

def clasificar(ops):
    if ops >= 0.900: return "⭐ Estrella"
    elif ops >= 0.800: return "🔥 Bueno"
    elif ops >= 0.700: return "👍 Promedio"
    else: return "📉 Bajo"

df = df_raw.copy()
df["Nivel"] = df["OPS"].apply(clasificar)
df = df[df["AB"] >= 100].copy()
df = df.sort_values(["Jugador", "Temporada"]).reset_index(drop=True)

# ==================== PREPARAR DATOS PARA PREDECIR 2026 ====================
@st.cache_data
def preparar_datos_prediccion(df):
    """
    Crea un dataset donde las características son del año T
    y el objetivo es el rendimiento del año T+1.
    """
    features = ["Edad", "Juegos", "AB", "Hits", "Home_Runs", "RBI", "BB", "SO", "AVG", "OBP", "SLG", "OPS"]
    
    filas = []
    for jugador, grupo in df.groupby("Jugador"):
        grupo = grupo.sort_values("Temporada")
        for i in range(len(grupo) - 1):
            actual = grupo.iloc[i]
            siguiente = grupo.iloc[i + 1]
            
            # Solo usamos si el siguiente año es consecutivo
            if siguiente["Temporada"] == actual["Temporada"] + 1:
                fila = {
                    "Jugador": jugador,
                    "Temporada_base": actual["Temporada"],
                    "Temporada_objetivo": siguiente["Temporada"],
                }
                for f in features:
                    fila[f] = actual[f]
                
                # Objetivos del año siguiente
                fila["OPS_siguiente"] = siguiente["OPS"]
                fila["HR_siguiente"] = siguiente["Home_Runs"]
                fila["AVG_siguiente"] = siguiente["AVG"]
                filas.append(fila)
    
    return pd.DataFrame(filas)

df_pred = preparar_datos_prediccion(df)

# Entrenar modelos
features_modelo = ["Edad", "Juegos", "AB", "Hits", "Home_Runs", "RBI", "BB", "SO", "AVG", "OBP", "SLG", "OPS"]

df_train = df_pred.dropna(subset=features_modelo + ["OPS_siguiente", "HR_siguiente"])

X = df_train[features_modelo]
y_ops = df_train["OPS_siguiente"]
y_hr = df_train["HR_siguiente"]

modelo_ops = LinearRegression()
modelo_ops.fit(X, y_ops)

modelo_hr = LinearRegression()
modelo_hr.fit(X, y_hr)

# Predecir 2026 usando datos de 2025
df_2025 = df[df["Temporada"] == 2025].copy()
df_2025 = df_2025.dropna(subset=features_modelo)

if len(df_2025) > 0:
    X_2026 = df_2025[features_modelo]
    df_2025["OPS_2026_pred"] = modelo_ops.predict(X_2026)
    df_2025["HR_2026_pred"] = modelo_hr.predict(X_2026)
    df_2025["Nivel_2026_pred"] = df_2025["OPS_2026_pred"].apply(clasificar)
    df_2025["HR_2026_pred"] = df_2025["HR_2026_pred"].clip(lower=0).round(0).astype(int)
else:
    df_2025 = pd.DataFrame()

# ==================== ENCABEZADO ====================
st.markdown('<p class="main-title">⚾ Análisis de Bateadores MLB</p>', unsafe_allow_html=True)
st.markdown('<p class="main-subtitle">Explora el rendimiento de los jugadores y descubre las predicciones para 2026</p>', unsafe_allow_html=True)

# Métricas
m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    st.metric("Jugadores", df["Jugador"].nunique())
with m2:
    st.metric("OPS promedio", f"{df['OPS'].mean():.3f}")
with m3:
    mejor = df.loc[df["OPS"].idxmax()]
    st.metric("Mejor OPS histórico", f"{mejor['OPS']:.3f}", mejor["Jugador"].split()[-1])
with m4:
    st.metric("Temporadas", "2022-2025")
with m5:
    if len(df_2025) > 0:
        st.metric("Predicciones 2026", len(df_2025))
    else:
        st.metric("Predicciones 2026", "—")

st.markdown("---")

# ==================== PESTAÑAS ====================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏆 Mejores Jugadores",
    "📈 Evolución",
    "🔮 Predicción 2026",
    "🌳 Árbol de Decisión",
    "ℹ️ Guía de Estadísticas"
])

# ========== TAB 1 ==========
with tab1:
    st.subheader("Los bateadores más destacados (2022-2025)")
    st.caption("Ordenados por OPS, la estadística más completa de un bateador.")
    
    col1, col2 = st.columns(2)
    with col1:
        top_ops = df.nlargest(12, "OPS")[["Jugador", "Temporada", "OPS", "Home_Runs"]]
        fig1 = px.bar(top_ops, x="OPS", y="Jugador", color="Temporada", orientation="h",
                      text="OPS", title="Top 12 por OPS")
        fig1.update_layout(yaxis={"categoryorder": "total ascending"}, height=480)
        fig1.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        top_hr = df.nlargest(12, "Home_Runs")[["Jugador", "Temporada", "Home_Runs"]]
        fig2 = px.bar(top_hr, x="Home_Runs", y="Jugador", color="Temporada", orientation="h",
                      text="Home_Runs", title="Top 12 en Jonrones")
        fig2.update_layout(yaxis={"categoryorder": "total ascending"}, height=480)
        st.plotly_chart(fig2, use_container_width=True)
    
    st.markdown("##### Tabla de los 15 mejores por OPS")
    st.dataframe(
        df.nlargest(15, "OPS")[["Jugador", "Equipo", "Temporada", "AVG", "OBP", "SLG", "OPS", "Home_Runs", "RBI", "Nivel"]],
        use_container_width=True, hide_index=True
    )

# ========== TAB 2 ==========
with tab2:
    st.subheader("Evolución de un jugador")
    st.caption("Observa cómo ha cambiado el rendimiento a lo largo de los años.")
    
    lista = sorted(df["Jugador"].unique())
    idx = lista.index("Aaron Judge") if "Aaron Judge" in lista else 0
    jugador = st.selectbox("Selecciona un jugador", lista, index=idx)
    
    df_j = df[df["Jugador"] == jugador].sort_values("Temporada")
    
    if len(df_j) > 0:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_j["Temporada"], y=df_j["OPS"], mode="lines+markers",
                                 name="OPS", line=dict(width=3.5, color="#2b6cb0"), marker=dict(size=10)))
        fig.add_trace(go.Scatter(x=df_j["Temporada"], y=df_j["AVG"], mode="lines+markers",
                                 name="AVG", line=dict(width=2.5, color="#38a169")))
        fig.add_trace(go.Scatter(x=df_j["Temporada"], y=df_j["SLG"], mode="lines+markers",
                                 name="SLG", line=dict(width=2.5, color="#d69e2e")))
        fig.update_layout(title=f"Evolución de {jugador}", height=420, hovermode="x unified",
                          legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("##### Detalle por temporada")
        cols = st.columns(len(df_j))
        for i, (_, row) in enumerate(df_j.iterrows()):
            with cols[i]:
                st.markdown(f"**{int(row['Temporada'])}**")
                st.write(f"Equipo: `{row['Equipo']}`")
                st.write(f"**OPS:** {row['OPS']:.3f}")
                st.write(f"AVG: {row['AVG']:.3f}")
                st.write(f"HR: {int(row['Home_Runs'])}")
                st.caption(row["Nivel"])

# ========== TAB 3: PREDICCIÓN 2026 (REGRESIÓN LINEAL) ==========
with tab3:
    st.subheader("🔮 Predicción de rendimiento para 2026")
    st.markdown("""
    Esta sección usa **Regresión Lineal** para predecir el rendimiento de 2026.
    
    **¿Cómo funciona?**  
    El modelo aprende de cómo cambiaron los jugadores de un año a otro (2022→2023, 2023→2024, 2024→2025)  
    y luego usa las estadísticas de **2025** para estimar cómo les irá en **2026**.
    """)
    
    if len(df_2025) == 0:
        st.warning("No hay suficientes datos de 2025 para generar predicciones.")
    else:
        st.markdown("### Ranking predicho para 2026")
        
        # Selector de qué ranking ver
        tipo_ranking = st.radio(
            "¿Qué quieres ver?",
            ["Mejor OPS predicho 2026", "Más Jonrones predichos 2026"],
            horizontal=True
        )
        
        if tipo_ranking == "Mejor OPS predicho 2026":
            ranking = df_2025.nlargest(15, "OPS_2026_pred")[
                ["Jugador", "Equipo", "OPS", "OPS_2026_pred", "HR_2026_pred", "Nivel_2026_pred"]
            ].copy()
            ranking.columns = ["Jugador", "Equipo", "OPS 2025", "OPS predicho 2026", "HR predichos 2026", "Nivel estimado 2026"]
            ranking["OPS 2025"] = ranking["OPS 2025"].round(3)
            ranking["OPS predicho 2026"] = ranking["OPS predicho 2026"].round(3)
            
            fig = px.bar(
                ranking.head(12),
                x="OPS predicho 2026",
                y="Jugador",
                orientation="h",
                text="OPS predicho 2026",
                title="Top 12 – OPS predicho para 2026",
                color="OPS predicho 2026",
                color_continuous_scale="Blues"
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=480, showlegend=False)
            fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            ranking = df_2025.nlargest(15, "HR_2026_pred")[
                ["Jugador", "Equipo", "Home_Runs", "HR_2026_pred", "OPS_2026_pred", "Nivel_2026_pred"]
            ].copy()
            ranking.columns = ["Jugador", "Equipo", "HR 2025", "HR predichos 2026", "OPS predicho 2026", "Nivel estimado 2026"]
            ranking["OPS predicho 2026"] = ranking["OPS predicho 2026"].round(3)
            
            fig = px.bar(
                ranking.head(12),
                x="HR predichos 2026",
                y="Jugador",
                orientation="h",
                text="HR predichos 2026",
                title="Top 12 – Jonrones predichos para 2026",
                color="HR predichos 2026",
                color_continuous_scale="Oranges"
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=480, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("##### Tabla completa de predicciones 2026")
        st.dataframe(ranking, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        st.markdown("### Consultar predicción de un jugador")
        
        jugador_pred = st.selectbox(
            "Elige un jugador para ver su predicción 2026",
            sorted(df_2025["Jugador"].unique())
        )
        
        fila = df_2025[df_2025["Jugador"] == jugador_pred].iloc[0]
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("OPS 2025", f"{fila['OPS']:.3f}")
        c2.metric("OPS predicho 2026", f"{fila['OPS_2026_pred']:.3f}", 
                  f"{fila['OPS_2026_pred'] - fila['OPS']:+.3f}")
        c3.metric("HR 2025", int(fila["Home_Runs"]))
        c4.metric("HR predichos 2026", int(fila["HR_2026_pred"]),
                  f"{int(fila['HR_2026_pred'] - fila['Home_Runs']):+d}")
        
        st.info(f"**Nivel estimado para 2026:** {fila['Nivel_2026_pred']}")

# ========== TAB 4: ÁRBOL DE DECISIÓN ==========
with tab4:
    st.subheader("🌳 Árbol de Decisión")
    st.markdown("""
    El **Árbol de Decisión** clasifica automáticamente a los jugadores en  
    **Estrella**, **Bueno**, **Promedio** o **Bajo** según sus estadísticas.
    
    Funciona como un cuestionario: va haciendo preguntas hasta llegar a una conclusión.
    """)
    
    features_tree = ["Edad", "Juegos", "AB", "Hits", "Home_Runs", "RBI", "BB", "SO", "AVG", "OBP", "SLG"]
    df_tree = df.dropna(subset=features_tree + ["Nivel"]).copy()
    
    X_t = df_tree[features_tree]
    y_t = df_tree["Nivel"]
    
    le = LabelEncoder()
    y_enc = le.fit_transform(y_t)
    
    arbol = DecisionTreeClassifier(max_depth=3, random_state=42)
    arbol.fit(X_t, y_enc)
    
    st.markdown("#### ¿Qué estadísticas son más importantes?")
    importancia = pd.DataFrame({
        "Estadística": features_tree,
        "Importancia": arbol.feature_importances_
    }).sort_values("Importancia", ascending=True)
    
    fig_imp = px.bar(importancia.tail(8), x="Importancia", y="Estadística", orientation="h",
                     title="Importancia de cada estadística", color="Importancia",
                     color_continuous_scale="Blues")
    fig_imp.update_layout(height=380, showlegend=False)
    st.plotly_chart(fig_imp, use_container_width=True)
    
    st.markdown("#### Visualización del Árbol de Decisión")
    st.caption("Cada caja es una pregunta. Según la respuesta, el jugador se clasifica en un nivel.")
    
    fig_arbol, ax = plt.subplots(figsize=(15, 7))
    plot_tree(arbol, feature_names=features_tree, class_names=le.classes_,
              filled=True, rounded=True, fontsize=9, ax=ax)
    st.pyplot(fig_arbol)
    
    st.markdown("---")
    st.markdown("#### Prueba el Árbol de Decisión")
    st.caption("Ingresa estadísticas y el árbol te dirá el nivel del jugador.")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        edad = st.number_input("Edad", 20, 40, 28)
        juegos = st.number_input("Juegos", 50, 162, 150)
        ab = st.number_input("AB", 100, 700, 550)
        hits = st.number_input("Hits", 50, 250, 160)
    with c2:
        hr = st.number_input("Jonrones (HR)", 0, 70, 30)
        rbi = st.number_input("RBI", 20, 150, 90)
        bb = st.number_input("BB", 10, 150, 60)
        so = st.number_input("SO", 30, 220, 130)
    with c3:
        avg = st.number_input("AVG", 0.150, 0.400, 0.280, format="%.3f")
        obp = st.number_input("OBP", 0.200, 0.500, 0.350, format="%.3f")
        slg = st.number_input("SLG", 0.250, 0.800, 0.480, format="%.3f")
    
    if st.button("Clasificar con el Árbol de Decisión", type="primary", use_container_width=True):
        entrada = np.array([[edad, juegos, ab, hits, hr, rbi, bb, so, avg, obp, slg]])
        pred = arbol.predict(entrada)[0]
        resultado = le.inverse_transform([pred])[0]
        st.success(f"### Resultado: {resultado}")

# ========== TAB 5 ==========
with tab5:
    st.subheader("Guía sencilla de estadísticas")
    st.markdown("""
    | Estadística | Significado | ¿Qué se considera bueno? |
    |-------------|-------------|---------------------------|
    | **AVG** | Promedio de bateo | +**.280** bueno · +**.300** excelente |
    | **OBP** | Frecuencia con la que llega a base | +**.350** muy bueno |
    | **SLG** | Poder (premia dobles, triples y jonrones) | +**.480** bueno |
    | **OPS** | OBP + SLG (la más completa) | +**.800** bueno · +**.900** estrella |
    | **HR** | Jonrones | **30+** gran temporada de poder |
    | **RBI** | Carreras impulsadas | **90+** muy buena cifra |
    
    ---
    
    #### Niveles de rendimiento:
    | Nivel | OPS | Significado |
    |-------|-----|-------------|
    | ⭐ Estrella | 0.900 o más | De los mejores de la liga |
    | 🔥 Bueno | 0.800 – 0.899 | Bateador destacado |
    | 👍 Promedio | 0.700 – 0.799 | Rendimiento aceptable |
    | 📉 Bajo | Menos de 0.700 | Por debajo del promedio |
    """)

st.markdown("---")
