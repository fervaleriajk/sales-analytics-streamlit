import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Configuración de la aplicación
st.set_page_config(
    page_title="Sales Analytics",
    page_icon="📊",
    layout="wide"
)

# Título principal
st.title("📊 Sales Analytics")
st.subheader("Dashboard Inteligente de Ventas")
st.write("Aplicación de analítica avanzada con Machine Learning")

# Cargar dataset
@st.cache_data
def cargar_datos():
    datos = pd.read_csv("ventas.csv")
    datos["Fecha"] = pd.to_datetime(datos["Fecha"])
    return datos

df = cargar_datos()

# Menú lateral
st.sidebar.title("Panel de control")

pagina = st.sidebar.radio(
    "Selecciona una sección:",
    [
        "Resumen General",
        "Análisis Exploratorio",
        "Machine Learning"
    ]
)

st.sidebar.markdown("---")

# Filtro de categoría
categorias = st.sidebar.multiselect(
    "Filtrar por categoría",
    options=sorted(df["Categoria"].unique()),
    default=sorted(df["Categoria"].unique())
)

df_filtrado = df[df["Categoria"].isin(categorias)]

# PÁGINA 1: RESUMEN GENERAL
if pagina == "Resumen General":

    st.header("Resumen General de Ventas")

    ventas_totales = df_filtrado["Total_Venta"].sum()
    unidades = df_filtrado["Cantidad"].sum()
    transacciones = len(df_filtrado)
    promedio = df_filtrado["Total_Venta"].mean()

    if pd.isna(promedio):
        promedio = 0

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Ventas Totales", f"S/ {ventas_totales:,.2f}")
    col2.metric("Unidades Vendidas", f"{unidades:,}")
    col3.metric("Transacciones", f"{transacciones:,}")
    col4.metric("Promedio por Venta", f"S/ {promedio:,.2f}")

    st.markdown("---")

    if df_filtrado.empty:
        st.warning("Selecciona al menos una categoría.")
    else:
        ventas_categoria = (
            df_filtrado.groupby("Categoria", as_index=False)["Total_Venta"]
            .sum()
        )

        fig1 = px.bar(
            ventas_categoria,
            x="Categoria",
            y="Total_Venta",
            color="Categoria",
            title="Ventas por Categoría"
        )

        ventas_producto = (
            df_filtrado.groupby("Producto", as_index=False)["Cantidad"]
            .sum()
            .sort_values("Cantidad", ascending=False)
        )

        fig2 = px.bar(
            ventas_producto,
            x="Producto",
            y="Cantidad",
            color="Producto",
            title="Productos Más Vendidos"
        )

        col1, col2 = st.columns(2)

        with col1:
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            st.plotly_chart(fig2, use_container_width=True)

# PÁGINA 2: ANÁLISIS EXPLORATORIO
elif pagina == "Análisis Exploratorio":

    st.header("Análisis Exploratorio de Datos")

    if df_filtrado.empty:
        st.warning("Selecciona al menos una categoría.")
    else:
        st.subheader("Dataset de Ventas")
        st.dataframe(df_filtrado, use_container_width=True)

        st.subheader("Estadísticas Descriptivas")
        st.dataframe(df_filtrado.describe())

        df_temp = df_filtrado.copy()
        df_temp["Mes"] = df_temp["Fecha"].dt.to_period("M").astype(str)

        ventas_mes = (
            df_temp.groupby("Mes", as_index=False)["Total_Venta"]
            .sum()
        )

        fig3 = px.line(
            ventas_mes,
            x="Mes",
            y="Total_Venta",
            markers=True,
            title="Evolución Mensual de Ventas"
        )

        st.plotly_chart(fig3, use_container_width=True)

        fig4 = px.pie(
            df_filtrado,
            names="Categoria",
            values="Total_Venta",
            title="Distribución de Ingresos por Categoría"
        )

        st.plotly_chart(fig4, use_container_width=True)

        st.subheader("Descargar Datos")

        csv = df_filtrado.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Descargar dataset CSV",
            data=csv,
            file_name="ventas_filtradas.csv",
            mime="text/csv"
        )

# PÁGINA 3: MACHINE LEARNING
elif pagina == "Machine Learning":

    st.header("Predicción de Ventas con Machine Learning")

    st.info(
        "Modelo demostrativo Random Forest entrenado con datos "
        "sintéticos de ventas mensuales de 2025."
    )

    # Cargar modelo guardado
    modelo = joblib.load("modelo_ventas.pkl")

    # Usar todo el dataset porque el modelo se
    # entrenó con ventas de todas las categorías
    ventas_mensuales = (
        df.set_index("Fecha")
        .resample("MS")["Total_Venta"]
        .sum()
        .reset_index()
    )

    ventas_mensuales["Mes_Numero"] = np.arange(
        1, len(ventas_mensuales) + 1
    )

    # Evaluación con los últimos 3 meses
    X_test = ventas_mensuales[["Mes_Numero"]].iloc[9:]
    y_test = ventas_mensuales["Total_Venta"].iloc[9:]

    predicciones = modelo.predict(X_test)

    mae = mean_absolute_error(y_test, predicciones)
    rmse = np.sqrt(mean_squared_error(y_test, predicciones))
    r2 = r2_score(y_test, predicciones)

    col1, col2, col3 = st.columns(3)

    col1.metric("MAE", f"S/ {mae:,.2f}")
    col2.metric("RMSE", f"S/ {rmse:,.2f}")
    col3.metric("R²", f"{r2:.4f}")

    st.subheader("Ventas Reales vs Predicciones")

    fig5 = go.Figure()

    fig5.add_trace(go.Scatter(
        x=ventas_mensuales["Fecha"],
        y=ventas_mensuales["Total_Venta"],
        mode="lines+markers",
        name="Ventas Reales"
    ))

    fig5.add_trace(go.Scatter(
        x=ventas_mensuales["Fecha"].iloc[9:],
        y=predicciones,
        mode="lines+markers",
        name="Predicciones",
        line=dict(dash="dash")
    ))

    fig5.update_layout(
        xaxis_title="Fecha",
        yaxis_title="Ventas (S/)"
    )

    st.plotly_chart(fig5, use_container_width=True)

    st.subheader("Tabla Comparativa")

    resultados = pd.DataFrame({
        "Mes": ventas_mensuales["Fecha"].iloc[9:].dt.strftime("%Y-%m").values,
        "Venta Real": y_test.values,
        "Venta Predicha": predicciones
    })

    st.dataframe(resultados, use_container_width=True)

    st.subheader("Estimación de Ventas")

    mes_futuro = st.selectbox(
        "Selecciona un mes:",
        ["2026-01", "2026-02", "2026-03"]
    )

    numero_mes = {
        "2026-01": 13,
        "2026-02": 14,
        "2026-03": 15
    }[mes_futuro]

    if st.button("Predecir ventas"):
        entrada = pd.DataFrame({
            "Mes_Numero": [numero_mes]
        })

        prediccion = modelo.predict(entrada)[0]

        st.success(
            f"Estimación demostrativa: S/ {prediccion:,.2f}"
        )

        st.caption(
            "Random Forest no extrapola bien tendencias fuera "
            "del periodo de entrenamiento. Esta estimación "
            "no es un pronóstico comercial confiable."
        )

# Pie de página
st.markdown("---")
st.caption(
    "Actividad 1 | Analítica Avanzada | "
    "Python + Streamlit + Random Forest"
)
