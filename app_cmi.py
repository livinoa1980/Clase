# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 14:42:11 2025

@author: Livino Armijos
"""

# Correr en terminal->  streamlit run "C:\Users\Lenovo\OneDrive\Escritorio\app_cmi.py"

# app.py
import os
import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# =============================
# Config general
# =============================
st.set_page_config(page_title="Cuadro de Mando", layout="wide")

# =============================
# Parámetros (puedes editar)
# =============================
RUTA_EXCEL_DEF = r"base_clase3.xlsx"
HOJA = "base"
LOGO_IZQ = r"image1.jpg"
LOGO_DER = r".jpg"

CATEG_LABELS = ["Deficiente", "Aceptable", "Excelente"]
CATEG_BINS = [-np.inf, 0.33, 0.66, np.inf]
COLORES = {"Deficiente": "#FF5733", "Aceptable": "#FFC300", "Excelente": "#2ECC71"}

COLUMNAS = [
    "Número", "Semestre", "Año", "Medición", "Componente", "Aula_Canvas",
    "Carrera", "Tipo_evaluado", "Participantes", "Criterio1", "Criterio2",
    "Criterio3", "Criterio4", "Resultados"
]

COLS_NUM = ["Año", "Medición", "Semestre", "Participantes",
            "Criterio1", "Criterio2", "Criterio3", "Criterio4", "Resultados"]

# =============================
# Carga de datos (con caché)
# =============================
@st.cache_data
def leer_excel(fh_or_path: str | io.BytesIO, sheet: str) -> pd.DataFrame:
    df = pd.read_excel(fh_or_path, sheet_name=sheet, header=None)
    # Intentar detectar si la primera fila son encabezados
    # y coinciden con los nombres que tú esperas.
    df.columns = COLUMNAS[:len(df.columns)]
    # Si en tu archivo la primera fila contenía los títulos originales, elimínala.
    # Detección simple: si en la primera fila hay strings como 'Número', 'Semestre', etc.
    primera = df.iloc[0].astype(str).str.lower().str.strip().tolist()
    coincide = any(word in " ".join(primera) for word in ["número", "semestre", "año", "medición", "componente"])
    if coincide:
        df = df.iloc[1:].reset_index(drop=True)
    return df.reset_index(drop=True)

def coerce_numericos(df: pd.DataFrame) -> pd.DataFrame:
    for c in (set(COLS_NUM) & set(df.columns)):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def expandir_por_participantes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Participantes nulos o <1 -> 1 (cada fila cuenta al menos 1)
    df["Participantes"] = df["Participantes"].fillna(1)
    df.loc[df["Participantes"] < 1, "Participantes"] = 1
    df["Participantes"] = df["Participantes"].astype(int)
    return df.loc[df.index.repeat(df["Participantes"])].reset_index(drop=True)

# =============================
# UI: Encabezado con logos
# =============================
c1, c2, c3 = st.columns([1, 6, 1])
with c1:
    if os.path.exists(LOGO_IZQ):
        st.image(LOGO_IZQ, use_container_width=True)
with c2:
    st.title("📊 Cuadro de Mando")
with c3:
    if os.path.exists(LOGO_DER):
        st.image(LOGO_DER, use_container_width=True)

# =============================
# Sidebar: carga y filtros
# =============================
st.sidebar.header("Archivo de datos")
uploaded = st.sidebar.file_uploader("Cargar Excel (.xlsx) – Hoja 'base'", type=["xlsx"])
ruta = uploaded if uploaded is not None else (RUTA_EXCEL_DEF if os.path.exists(RUTA_EXCEL_DEF) else None)

if ruta is None:
    st.warning("⚠️ Sube el archivo Excel o actualiza la ruta por defecto.")
    st.stop()

df = leer_excel(ruta, HOJA)
df = coerce_numericos(df)
df = df.dropna(subset=["Resultados"])

# Normalización defensiva (por si los resultados están en 0–100 en lugar de 0–1)
if df["Resultados"].max() > 1.5:  # heurística
    df["Resultados"] = df["Resultados"] / 100.0

# Expansión por participantes
df_expanded = expandir_por_participantes(df)

# Categoría de resultado
df_expanded["Categoría Resultados"] = pd.cut(
    df_expanded["Resultados"],
    bins=CATEG_BINS,
    labels=CATEG_LABELS,
    include_lowest=True,
    right=True
)

# Medición como entero ordenable (si procede)
if pd.api.types.is_numeric_dtype(df_expanded["Medición"]):
    df_expanded["Medición"] = df_expanded["Medición"].astype(int)

# ----------------- Filtros -----------------
st.sidebar.header("Filtros")
carreras = ["Todos"] + sorted(df_expanded["Carrera"].dropna().unique().tolist())
carrera_sel = st.sidebar.selectbox("Carrera", carreras, index=0)

if carrera_sel != "Todos":
    componentes = ["Todos"] + sorted(
        df_expanded.loc[df_expanded["Carrera"] == carrera_sel, "Componente"].dropna().unique().tolist()
    )
else:
    componentes = ["Todos"] + sorted(df_expanded["Componente"].dropna().unique().tolist())
comp_sel = st.sidebar.selectbox("Componente", componentes, index=0)

df_filtrado = df_expanded.copy()
if carrera_sel != "Todos":
    df_filtrado = df_filtrado[df_filtrado["Carrera"] == carrera_sel]
if comp_sel != "Todos":
    df_filtrado = df_filtrado[df_filtrado["Componente"] == comp_sel]

# =============================
# KPIs
# =============================
k1, k2, k3, k4 = st.columns(4)
k1.metric("Mediciones", f"{df_filtrado['Medición'].nunique():,}")
k2.metric("Registros", f"{len(df_filtrado):,}")
k3.metric("Participantes (exp.)", f"{df_filtrado.groupby('Medición').size().sum():,}")
k4.metric("Carreras activas", f"{df_filtrado['Carrera'].nunique():,}")

st.divider()

# =============================
# Gráfico principal: Evolución por medición
# =============================
st.subheader("Evolución de resultados por medición")

modo = st.radio(
    "Modo de barras",
    ["Conteo", "Porcentaje (100%)"],
    horizontal=True
)

# Conteos por medición y categoría (completa categorías ausentes)
base = (
    df_filtrado
    .groupby(["Medición", "Categoría Resultados"])
    .size()
    .rename("Cuenta")
    .reset_index()
)

# Asegurar presencia de todas las categorías
meds = sorted(base["Medición"].dropna().unique().tolist())
cat_idx = pd.MultiIndex.from_product([meds, CATEG_LABELS], names=["Medición", "Categoría Resultados"])
base = base.set_index(["Medición", "Categoría Resultados"]).reindex(cat_idx, fill_value=0).reset_index()

if modo.startswith("Porcentaje"):
    tot = base.groupby("Medición")["Cuenta"].transform("sum").replace(0, np.nan)
    base["Porcentaje"] = (base["Cuenta"] / tot * 100).fillna(0)
    ycol, ylab = "Porcentaje", "Porcentaje (%)"
else:
    ycol, ylab = "Cuenta", "Número de casos"

fig = px.bar(
    base.sort_values("Medición"),
    x="Medición",
    y=ycol,
    color="Categoría Resultados",
    color_discrete_map=COLORES,
    barmode="group" if modo == "Conteo" else "relative",
    text=ycol,
    labels={ycol: ylab}
)
fig.update_traces(textposition="outside")
fig.update_layout(title="Evolución de resultados por medición", legend_title_text="Categoría")
st.plotly_chart(fig, use_container_width=True)

# =============================
# Gráficos adicionales (EDA)
# =============================
st.subheader("Distribución y comparación")

cA, cB = st.columns(2)

with cA:
    fig2 = px.histogram(
        df_filtrado, x="Resultados", nbins=20, color="Categoría Resultados",
        color_discrete_map=COLORES, barmode="overlay", opacity=0.7,
        title="Distribución de Resultados"
    )
    st.plotly_chart(fig2, use_container_width=True)

with cB:
    agg = (
        df_filtrado
        .groupby(["Carrera", "Categoría Resultados"])
        .size()
        .rename("Conteo")
        .reset_index()
    )
    if not agg.empty:
        fig3 = px.bar(
            agg, x="Carrera", y="Conteo", color="Categoría Resultados",
            color_discrete_map=COLORES, barmode="group",
            title="Resultados por Carrera"
        )
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("No hay datos para el filtro seleccionado.")

# =============================
# Tabla y descarga
# =============================
st.subheader("Datos filtrados")
st.dataframe(df_filtrado.head(100), use_container_width=True)

csv = df_filtrado.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️ Descargar CSV filtrado",
    data=csv,
    file_name="resultados_filtrados.csv",
    mime="text/csv",
)

st.caption("Tip: si los resultados originales están en 0–100, el tablero los normaliza a 0–1 automáticamente.")

