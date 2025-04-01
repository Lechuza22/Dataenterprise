import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn.objects as so
import folium
import statsmodels.api as sm
import plotly.express as px
import calendar
from streamlit_folium import st_folium
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import HuberRegressor
from sklearn.neighbors import NearestNeighbors

# -----------------------------
# CONFIGURACION INICIAL
# -----------------------------
st.set_page_config(page_title="📊 DataEnterprise", page_icon="🏢", layout="wide")

# -----------------------------
# LOGIN SIMPLE CON SESSION_STATE
# -----------------------------
# Inicializamos el estado de autenticación si no existe
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

# Si no está autenticado, mostrar input de contraseña
if not st.session_state.authenticated:
    st.title("📊 DataEnterprise")
    password = st.text_input("🔐 Ingresá la clave para acceder a la app:", type="password")
    if password == st.secrets["acceso"]["clave"]:
        st.session_state.authenticated = True
        st.success("Acceso concedido ✅")
    elif password != "":
        st.error("Clave incorrecta ❌")

# Si está autenticado, mostrar la app completa
if st.session_state.authenticated:
    st.title("📊 DataEnterprise")

    # -----------------------------
    # MENU PRINCIPAL
    # -----------------------------
    menu = st.sidebar.selectbox("📂 Secciones", [
        "Inicio",
        "Análisis exploratorio",
        "Análisis cruzado",
        "Modelos de ML",
        "Mapa de sucursales y empleados"
    ])

    st.sidebar.markdown("---")
    st.sidebar.markdown("👤 Usuario: Admin")

    # -----------------------------
    # CONTENIDO POR SECCION
    # -----------------------------
    if menu == "Inicio":
        st.header("📊 DataEnterprise - Proyecto de Análisis de Datos Empresariales")
        st.markdown("Bienvenido al panel interactivo de análisis, exploración y predicción.")
        st.markdown("Usá el menú de la izquierda para navegar por las secciones.")

    elif menu == "Análisis exploratorio":
        st.header("📈 Análisis exploratorio de datos (EDA)")

        dataset_opcion = st.selectbox("Seleccioná el dataset a explorar:", [
            "Clientes", "Compras", "Empleados", "Gastos", "Productos", "Proveedores", "Sucursales", "Ventas"
        ])

        if dataset_opcion == "Clientes":
            st.subheader("🧍‍♂️ Exploración de Clientes")
            st.markdown("✅ Conclusiones preliminares del análisis del dataset Clientes: - Edad promedio de los clientes es de 40 años, con una alta concentración entre los 25 y 55.\n- Hay una clara concentración geográfica en el AMBA, especialmente Ciudad de Buenos Aires.\n- El 100% de los clientes están activos (no hay marca de baja).\n- La diversidad de localidades es grande (527), pero unas pocas concentran la mayoría.\n- La base de clientes parece limpia y homogénea, con pocos outliers.")

            df_clientes = pd.read_csv("Clientes_transformados.csv")

            # Histograma de edades
            st.markdown("### 📊 Distribución de edades")
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.histplot(df_clientes["Edad"], bins=20, kde=True, ax=ax, color="skyblue")
            ax.set_title("Distribución de edades de los clientes")
            ax.set_xlabel("Edad")
            ax.set_ylabel("Cantidad")
            st.pyplot(fig)

            # Top 10 localidades
            st.markdown("### 🏙️ Top 10 Localidades con más clientes")
            top_localidades = df_clientes["Localidad"].value_counts().head(10)
            fig2, ax2 = plt.subplots()
            top_localidades.plot(kind="barh", ax=ax2, color="teal")
            ax2.invert_yaxis()
            ax2.set_title("Top 10 Localidades")
            ax2.set_xlabel("Cantidad de clientes")
            st.pyplot(fig2)

            # Mapa geográfico de clientes (si hay coordenadas)
            if "X" in df_clientes.columns and "Y" in df_clientes.columns:
                st.markdown("### 🌍 Mapa de distribución geográfica")
                mapa = folium.Map(location=[df_clientes["Y"].mean(), df_clientes["X"].mean()], zoom_start=5)
                for _, row in df_clientes.iterrows():
                    folium.CircleMarker(
                        location=[row["Y"], row["X"]],
                        radius=2,
                        color='blue',
                        fill=True,
                        fill_opacity=0.6
                    ).add_to(mapa)
                st_folium(mapa, width=700, height=400)

            # Heatmap de correlaciones
            st.markdown("### 🔥 Correlación entre variables numéricas")
            corr = df_clientes.select_dtypes(include=['float64', 'int64']).corr()
            fig3, ax3 = plt.subplots()
            sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax3)
            ax3.set_title("Heatmap de correlaciones")
            st.pyplot(fig3)

            # Estadísticas descriptivas
            st.subheader("📋 Estadísticas descriptivas")
            st.dataframe(df_clientes.describe())

        elif dataset_opcion == "Compras":
            st.subheader("🛒 Exploración de Compras")
            st.markdown("✅ Conclusiones preliminares del análisis de Compras: - El volumen principal de compras se concentra en productos de bajo a mediano precio (menos de $1200).\n-Se compran en promedio 9 unidades por operación, con pocas compras mayores a 25 unidades..\n- Proveedor 8, seguido de 12 y 7, domina en volumen de compras..\n- No hay relación directa entre Precio y Cantidad, lo que sugiere que el tipo de producto define el patrón más que el monto.\n- Existen outliers en precios que podrían representar productos premium, errores de carga o compras especiales.")
          
            df_compras = pd.read_csv("Compra_transformada.csv")

            # Histograma de cantidad de compras
            st.markdown("### 📦 Distribución de cantidad por compra")
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.histplot(df_compras["Cantidad"], bins=30, kde=True, ax=ax, color="orange")
            ax.set_title("Distribución de cantidades por compra")
            ax.set_xlabel("Cantidad")
            ax.set_ylabel("Frecuencia")
            st.pyplot(fig)

            # Top 10 productos más comprados
            st.markdown("### 🥇 Top 10 productos más comprados")
            top_productos = df_compras["IdProducto"].value_counts().head(10)
            fig2, ax2 = plt.subplots()
            top_productos.plot(kind="bar", ax=ax2, color="green")
            ax2.set_title("Top 10 productos por frecuencia de compra")
            ax2.set_xlabel("IdProducto")
            ax2.set_ylabel("Número de compras")
            st.pyplot(fig2)

            # Heatmap de correlaciones
            st.markdown("### 🔥 Correlación entre variables numéricas")
            corr_compras = df_compras.select_dtypes(include=['float64', 'int64']).corr()
            fig4, ax4 = plt.subplots()
            sns.heatmap(corr_compras, annot=True, cmap="coolwarm", ax=ax4)
            ax4.set_title("Heatmap de correlaciones - Compras")
            st.pyplot(fig4)

            # Visualización bivariada: IdProducto vs Cantidad
            st.markdown("### 📊 Relación entre Producto y Cantidad Comprada")
            fig3, ax3 = plt.subplots(figsize=(10, 4))
            top_ids = df_compras['IdProducto'].value_counts().head(10).index
            sns.boxplot(data=df_compras[df_compras['IdProducto'].isin(top_ids)],
                        x="IdProducto", y="Cantidad", ax=ax3, palette="pastel")
            ax3.set_title("Distribución de cantidades por producto (Top 10)")
            st.pyplot(fig3)

            # Estadísticas descriptivas
            st.subheader("📋 Estadísticas descriptivas")
            st.dataframe(df_compras.describe())


        elif dataset_opcion == "Empleados":
            st.subheader("👔 Exploración de Empleados")
            st.markdown("✅ Conclusiones preliminares del dataset Empleados:\n- El salario más frecuente es $32.000, y la mayoría de empleados cobra entre $15.000 y $36.000.\n- El rol de vendedor domina la estructura laboral (más del 60% del total).\n- El sector más numeroso es ventas, seguido de administración y logística.\n- Los salarios más altos se encuentran en administración y sistemas.\n- Las sucursales están bastante equilibradas, con una leve concentración en morón, caseros y cabildo.")
        
            df_empleados = pd.read_csv("Empleados_transformados.csv")
        
            # Histograma de salarios
            st.markdown("### 💵 Distribución de Salarios")
            fig, ax = plt.subplots()
            sns.histplot(df_empleados["Salario"], bins=30, kde=True, ax=ax, color="lightgreen")
            ax.set_title("Distribución de salarios")
            st.pyplot(fig)
        
            # Empleados por cargo
            st.markdown("### 👷‍♂️ Distribución por Cargo")
            fig2, ax2 = plt.subplots()
            df_empleados["Cargo"].value_counts().plot(kind="bar", ax=ax2, color="steelblue")
            ax2.set_title("Cantidad de empleados por cargo")
            ax2.set_ylabel("Cantidad")
            st.pyplot(fig2)
        
            # Boxplot salario por cargo
            st.markdown("### 📊 Salario por Cargo")
            fig3, ax3 = plt.subplots(figsize=(10, 5))
            sns.boxplot(data=df_empleados, x="Cargo", y="Salario", ax=ax3, palette="pastel")
            ax3.set_title("Distribución de salario por cargo")
            ax3.tick_params(axis='x', rotation=45)
            st.pyplot(fig3)

            # Gráfico de conteo por Sucursal
            st.markdown("### 🏢 Empleados por Sucursal")
            fig1, ax1 = plt.subplots()
            df_empleados['Sucursal'].value_counts().plot(kind='bar', ax=ax1, color='lightblue')
            ax1.set_title("Cantidad de empleados por sucursal")
            st.pyplot(fig1)

            # Gráfico de conteo por Sector
            st.markdown("### 🗂️ Empleados por Sector")
            fig2, ax2 = plt.subplots()
            df_empleados['Sector'].value_counts().plot(kind='bar', ax=ax2, color='lightgreen')
            ax2.set_title("Cantidad de empleados por sector")
            st.pyplot(fig2)

            # Gráfico de conteo por Cargo
            st.markdown("### 👷‍♂️ Empleados por Cargo")
            fig3, ax3 = plt.subplots()
            df_empleados['Cargo'].value_counts().plot(kind='bar', ax=ax3, color='salmon')
            ax3.set_title("Cantidad de empleados por cargo")
            st.pyplot(fig3)
           
            # Estadísticas descriptivas
            st.subheader("📋 Estadísticas descriptivas")
            st.dataframe(df_empleados.describe())


        elif dataset_opcion == "Gastos":
            st.subheader("💸 Exploración de Gastos")
            st.markdown("✅ Conclusiones preliminares del dataset Gasto:\n- El monto promedio por gasto es de $660, con un máximo de casi $1.200.\n- El gasto diario es estable, con picos regulares, lo que sugiere planificación.\n- Las sucursales 18, 1 y 2 son las de mayor gasto.\n- Los tipos de gasto 1 y 4 concentran la mayor parte del presupuesto.\n- No se observan outliers ni anomalías significativas.")
        
            df_gastos = pd.read_csv("Gasto_transformado.csv")
        
            # Histograma de montos
            st.markdown("### 💰 Distribución de Montos de Gasto")
            fig1, ax1 = plt.subplots()
            sns.histplot(df_gastos["Monto"], bins=30, kde=True, color="coral", ax=ax1)
            ax1.set_title("Distribución de montos de gasto")
            st.pyplot(fig1)
        
            # Gasto por tipo
            st.markdown("### 🧾 Gasto por Tipo")
            fig2, ax2 = plt.subplots()
            df_gastos["IdTipoGasto"].value_counts().plot(kind="bar", ax=ax2, color="orchid")
            ax2.set_title("Cantidad de registros por tipo de gasto")
            st.pyplot(fig2)
        
            # Gasto por sucursal
            st.markdown("### 🏢 Gasto total por Sucursal")
            gasto_sucursal = df_gastos.groupby("IdSucursal")["Monto"].sum().sort_values(ascending=False)
            fig3, ax3 = plt.subplots()
            gasto_sucursal.plot(kind="bar", ax=ax3, color="skyblue")
            ax3.set_title("Gasto total por sucursal")
            st.pyplot(fig3)
        
            # Serie temporal de gastos
            st.markdown("### 📅 Evolución temporal de los gastos")
            df_gastos["Fecha"] = pd.to_datetime(df_gastos["Fecha"])
            serie = df_gastos.groupby("Fecha")["Monto"].sum()
            fig4, ax4 = plt.subplots()
            serie.plot(ax=ax4, color="green")
            ax4.set_title("Gastos diarios totales")
            st.pyplot(fig4)
            
            # Heatmap de correlación
            st.markdown("### 🔥 Correlación entre variables numéricas")
            fig5, ax5 = plt.subplots()
            sns.heatmap(df_gastos.select_dtypes(include="number").corr(), annot=True, cmap="coolwarm", ax=ax5)
            ax5.set_title("Matriz de correlaciones - Gastos")
            st.pyplot(fig5)
            
            # Estadísticas
            st.subheader("📋 Estadísticas descriptivas")
            st.dataframe(df_gastos.describe())
        

        elif dataset_opcion == "Productos":
            st.subheader("📦 Exploración de Productos")
            st.markdown("✅ Conclusiones del análisis del dataset PRODUCTOS_transformado.csv + Compras:\n- Catálogo con 291 productos únicos; destacan impresión e informática.\n- 10 tipos de producto; revisar duplicados por concepto.\n- Precios entre $400 y $2000; algunos outliers elevan el promedio.\n- Producto más caro real: NAS QNAP ($9555). Más barato: funda para tablet ($3).\n- Top comprados: valijas, cartuchos, mouse pad, etc.\n- Alta rotación de insumos sugiere operación comercial o institucional.\n- Posible análisis futuro de rentabilidad y rotación con datos de ventas.")
        
            df_productos = pd.read_csv("PRODUCTOS_transformado.csv")
            df_compras = pd.read_csv("Compra_transformada.csv")
        
            # Histograma de precios
            st.markdown("### 💰 Distribución de precios (con outliers)")
            fig1, ax1 = plt.subplots()
            sns.histplot(df_productos["Precio"], bins=50, ax=ax1, color="skyblue")
            ax1.set_title("Distribución de precios de productos")
            st.pyplot(fig1)
        
            # Productos más comprados con nombres
            st.markdown("### 🏆 Top 10 productos más comprados (con nombre)")
            top_ids = df_compras["IdProducto"].value_counts().head(10).reset_index()
            top_ids.columns = ["IdProducto", "Total"]
            
            # Merge con productos para obtener nombres
            top_nombres = top_ids.merge(df_productos[["ID_PRODUCTO", "Concepto"]], left_on="IdProducto", right_on="ID_PRODUCTO")
            
            fig, ax = plt.subplots()
            sns.barplot(data=top_nombres, x="Total", y="Concepto", ax=ax, palette="Blues_d")
            ax.set_title("Productos más comprados (por nombre)")
            ax.set_xlabel("Cantidad comprada")
            ax.set_ylabel("Producto")
            st.pyplot(fig)

        
            # Top productos más comprados
            st.markdown("### 🥇 Productos más comprados (Top 10)")
            top_ids = df_compras["IdProducto"].value_counts().head(10)
            fig3, ax3 = plt.subplots()
            top_ids.plot(kind="bar", ax=ax3, color="lightgreen")
            ax3.set_title("Top productos más comprados")
            ax3.set_xlabel("IdProducto")
            st.pyplot(fig3)
        
            # Estadísticas descriptivas
            st.subheader("📋 Estadísticas descriptivas de precios")
            st.dataframe(df_productos.describe())


        elif dataset_opcion == "Proveedores":
            st.subheader("🏭 Exploración de Proveedores")
            st.markdown("✅ Conclusiones del análisis del dataset Proveedores:\n- Hay un total de 14 proveedores registrados, todos en Argentina.\n- La mayoría se encuentran en la provincia de Buenos Aires, especialmente en el departamento capital.\n- Hay 3 proveedores repetidos por nombre, lo que sugiere sucursales o registros duplicados.\n- El dataset parece limpio, sin valores nulos, aunque podría mejorarse agregando CUIT, rubros, emails o teléfonos.")
        
            df_proveedores = pd.read_csv("Proveedores_transformado.csv")
        
            # Proveedores por provincia
            st.markdown("### 🗺️ Proveedores por Provincia")
            fig1, ax1 = plt.subplots()
            df_proveedores['State'].value_counts().plot(kind='bar', ax=ax1, color='skyblue')
            ax1.set_title("Cantidad de proveedores por provincia")
            st.pyplot(fig1)
        
            # Proveedores por ciudad
            st.markdown("### 🏙️ Proveedores por Ciudad")
            fig2, ax2 = plt.subplots()
            df_proveedores['City'].value_counts().head(10).plot(kind='bar', ax=ax2, color='coral')
            ax2.set_title("Top 10 ciudades con más proveedores")
            st.pyplot(fig2)
        
            # Duplicados por nombre
            st.markdown("### 🔍 Posibles Duplicados por Nombre")
            duplicados = df_proveedores['Nombre'].value_counts()
            duplicados = duplicados[duplicados > 1]
            st.dataframe(duplicados)


        elif dataset_opcion == "Sucursales":
            st.subheader("🏢 Exploración de Sucursales")
            st.markdown("✅ Conclusiones del análisis del dataset Sucursales:\n- La empresa tiene 31 sucursales distribuidas en 17 provincias argentinas.\n- La mayor presencia está en Buenos Aires (9 sucursales).\n- Varias localidades clave tienen más de una sucursal: CABA, Rosario, Mendoza, etc.\n- Las coordenadas permiten análisis espaciales y mapas.\n- Hay posibles redundancias en nombres de localidades (\"CABA\" y \"Ciudad de Buenos Aires\").")
        
            df_sucursales = pd.read_csv("Sucursales_transformado.csv")
        
            # Conteo por provincia
            st.markdown("### 🗺️ Cantidad de sucursales por provincia")
            fig1, ax1 = plt.subplots()
            df_sucursales["Provincia"].value_counts().plot(kind="bar", ax=ax1, color="lightblue")
            ax1.set_title("Sucursales por provincia")
            st.pyplot(fig1)
        
            # Conteo por localidad
            st.markdown("### 🏙️ Top localidades con más sucursales")
            fig2, ax2 = plt.subplots()
            df_sucursales["Localidad"].value_counts().head(10).plot(kind="bar", ax=ax2, color="lightgreen")
            ax2.set_title("Top localidades")
            st.pyplot(fig2)
        
            # Mapa de sucursales
            st.markdown("### 🌍 Mapa geográfico de sucursales")
            mapa = folium.Map(location=[df_sucursales["Latitud"].mean(), df_sucursales["Longitud"].mean()], zoom_start=5)
            for _, row in df_sucursales.iterrows():
                folium.Marker(location=[row["Latitud"], row["Longitud"]], popup=row["Sucursal"]).add_to(mapa)
            st_folium(mapa, width=700, height=400)

        elif dataset_opcion == "Ventas":
            st.subheader("💰 Exploración de Ventas")
            st.markdown("✅ Conclusiones del análisis del dataset Ventas:\n- El volumen de ventas es muy alto (más de 46.000 registros).\n- La mayoría de las ventas son de 1 a 3 unidades, con pocos casos mayores a 10.\n- Las ventas diarias son constantes, con picos estacionales.\n- Los productos más vendidos incluyen:\n    - Periféricos (mouse pads)\n    - Estuchería (mochilas y fundas)\n    - Insumos (cartuchos, limpiadores)\n- Hay una coherencia importante con los productos más comprados, lo que sugiere buena planificación de stock.")
        
            df_ventas = pd.read_csv("Venta_transformado.csv")
            df_ventas["Fecha"] = pd.to_datetime(df_ventas["Fecha"])
        
            # Ventas mensuales
            st.markdown("### 📅 Ventas mensuales")
            ventas_mensuales = df_ventas.groupby(df_ventas["Fecha"].dt.to_period("M")).size()
            ventas_mensuales.index = ventas_mensuales.index.to_timestamp()
            fig1, ax1 = plt.subplots()
            ventas_mensuales.plot(ax=ax1, color="green")
            ax1.set_title("Ventas mensuales")
            st.pyplot(fig1)
        
          # Ventas por canal
            st.markdown("### 🛍️ Ventas por canal")
            fig2, ax2 = plt.subplots()
            canales = {1: "Tienda Física", 2: "Online", 3: "Mayorista", 4: "Otros"}
            df_ventas["Canal"] = df_ventas["IdCanal"].map(canales)
            df_ventas["Canal"].value_counts().plot(kind="bar", ax=ax2, color="skyblue")
            ax2.set_title("Cantidad de ventas por canal (con nombres)")
            st.pyplot(fig2)

            # Ventas por sucursal
            st.markdown("### 🏢 Ventas por sucursal")
            fig3, ax3 = plt.subplots()
            df_sucursales = pd.read_csv("Sucursales_transformado.csv")
            sucursal_map = df_sucursales.set_index("ID")["Sucursal"].to_dict()
            df_ventas["Sucursal"] = df_ventas["IdSucursal"].map(sucursal_map)
            df_ventas["Sucursal"].value_counts().plot(kind="bar", ax=ax3, color="orange")
            ax3.set_title("Ventas por sucursal (con nombre)")
            st.pyplot(fig3)
            
            # Top productos más vendidos (con nombre)
            st.markdown("### 🏆 Top 10 productos más vendidos (por nombre)")
            df_productos = pd.read_csv("PRODUCTOS_transformado.csv")
            top_ventas = df_ventas["IdProducto"].value_counts().head(10).reset_index()
            top_ventas.columns = ["IdProducto", "Total"]
            top_ventas = top_ventas.merge(df_productos[["ID_PRODUCTO", "Concepto"]], left_on="IdProducto", right_on="ID_PRODUCTO")
            
            fig, ax = plt.subplots()
            sns.barplot(data=top_ventas, x="Total", y="Concepto", ax=ax, palette="Blues_d")
            ax.set_title("Productos más vendidos (por nombre)")
            ax.set_xlabel("Cantidad vendida")
            ax.set_ylabel("Producto")
            st.pyplot(fig)

            # Estadísticas descriptivas
            st.subheader("📋 Estadísticas descriptivas")
            st.dataframe(df_ventas.describe())


    elif menu == "Análisis cruzado":
        st.header("🔀 Análisis cruzado entre áreas")

        analisis_opcion = st.selectbox("Seleccioná el análisis cruzado a visualizar:", [
            "🛍️ Productos más vendidos vs. más comprados",
            "📍 Sucursales con más ventas vs. más gastos",
            "💸 Relación entre salario de empleados y volumen de ventas",
            "👥 Perfil de cliente vs. tipo de producto vendido",
            "🛒 Canal de venta vs. volumen/monto de ventas",
            "📈 Evolución histórica de ventas por canal",
            "📊 Proveedor con mayor volumen de compra",
            "💡 Comparar precios de compra vs. venta por producto (margen)"
        ])

        if analisis_opcion == "🛍️ Productos más vendidos vs. más comprados":
            st.markdown("### 🛍️ Productos más vendidos vs. más comprados")
            st.markdown("🔎 ¿Qué muestra el gráfico?\n- Comparación directa de la cantidad vendida vs. la cantidad comprada por producto.\n- Podés ver claramente si hay productos:\n    - Con más ventas que compras → posible falta de stock o desabastecimiento.\n    - Con más compras que ventas → posible exceso de stock o baja rotación.")

            df_ventas = pd.read_csv("Venta_transformado.csv")
            df_compras = pd.read_csv("Compra_transformada.csv")
            df_productos = pd.read_csv("PRODUCTOS_transformado.csv")

            # Agrupamos ventas y compras por producto
            ventas = df_ventas["IdProducto"].value_counts().reset_index()
            ventas.columns = ["IdProducto", "Cantidad_Vendida"]

            compras = df_compras["IdProducto"].value_counts().reset_index()
            compras.columns = ["IdProducto", "Cantidad_Comprada"]

            # Merge y agregamos nombres
            df_merge = ventas.merge(compras, on="IdProducto", how="outer").fillna(0)
            df_merge = df_merge.merge(df_productos[["ID_PRODUCTO", "Concepto"]], left_on="IdProducto", right_on="ID_PRODUCTO")

            # Top 10 productos por ventas
            top = df_merge.sort_values(by="Cantidad_Vendida", ascending=False).head(10)

            # Gráfico comparativo
            st.markdown("### 📊 Comparación de productos más vendidos y comprados")
            fig, ax = plt.subplots(figsize=(10, 6))
            bar_width = 0.4
            x = range(len(top))

            ax.bar(x, top["Cantidad_Vendida"], width=bar_width, label="Vendidos", color="blue")
            ax.bar([i + bar_width for i in x], top["Cantidad_Comprada"], width=bar_width, label="Comprados", color="orange")
            ax.set_xticks([i + bar_width/2 for i in x])
            ax.set_xticklabels(top["Concepto"], rotation=45, ha="right")
            ax.set_ylabel("Cantidad")
            ax.set_title("Productos más vendidos vs. más comprados")
            ax.legend()
            st.pyplot(fig)

        elif analisis_opcion == "📍 Sucursales con más ventas vs. más gastos":
            st.markdown("### 📍 Sucursales con más ventas vs. más gastos")
            st.markdown("🔎 ¿Qué observamos?\n- Las sucursales con mayor volumen de ventas no siempre son las que más gastan.\n- Algunas sucursales tienen gastos elevados en proporción a sus ventas, lo que podría indicar:\n    - Ineficiencia operativa\n    - Costos fijos altos\n    - Gasto en infraestructura/logística no rentable\n\n💡 Ideal para analizar rentabilidad por punto de venta.")
        
            df_ventas = pd.read_csv("Venta_transformado.csv")
            df_gastos = pd.read_csv("Gasto_transformado.csv")
            df_sucursales = pd.read_csv("Sucursales_transformado.csv")
        
            # Ventas por sucursal
            ventas_sucursal = df_ventas.groupby("IdSucursal").size().reset_index(name="Ventas")
        
            # Gastos por sucursal
            gastos_sucursal = df_gastos.groupby("IdSucursal")["Monto"].sum().reset_index(name="Gastos")
        
            # Merge con nombres de sucursales
            df_merge = ventas_sucursal.merge(gastos_sucursal, on="IdSucursal")
            sucursal_map = df_sucursales.set_index("ID")["Sucursal"].to_dict()
            df_merge["Sucursal"] = df_merge["IdSucursal"].map(sucursal_map)
        
            df_top = df_merge.sort_values(by="Ventas", ascending=False).head(10)
        
            # Gráfico comparativo
            fig, ax = plt.subplots(figsize=(10, 6))
            bar_width = 0.4
            x = range(len(df_top))
        
            ax.bar(x, df_top["Ventas"], width=bar_width, label="Ventas", color="blue")
            ax.bar([i + bar_width for i in x], df_top["Gastos"], width=bar_width, label="Gastos", color="orange")
            ax.set_xticks([i + bar_width/2 for i in x])
            ax.set_xticklabels(df_top["Sucursal"], rotation=45, ha="right")
            ax.set_ylabel("Cantidad")
            ax.set_title("Top 10 sucursales con más ventas vs. más gastos")
            ax.legend()
            st.pyplot(fig)

        elif analisis_opcion == "💸 Relación entre salario de empleados y volumen de ventas":
            st.markdown("### 💸 Relación entre salario de empleados y volumen de ventas")
            st.markdown("🔎 ¿Qué revela el gráfico?.\n- No hay una correlación directa fuerte entre salario y ventas generadas.\n- Algunos empleados con salarios medios generan altas ventas, lo cual sugiere alto rendimiento.\n- También hay empleados con salario alto y ventas bajas, lo cual puede indicar o Cargos administrativos o Antigüedad o jerarquía sin tareas comerciales directas.\n- 💡 Muy útil para evaluar productividad individual y tomar decisiones sobre incentivos o comisiones.")
        
            df_empleados = pd.read_csv("Empleados_transformados.csv")
            df_ventas = pd.read_csv("Venta_transformado.csv")
    
            ventas_empleado = df_ventas.groupby("IdEmpleado").size().reset_index(name="Ventas")
            empleados_merge = df_empleados.merge(ventas_empleado, left_on="ID_empleado", right_on="IdEmpleado", how="left").fillna(0)
            top_20 = empleados_merge.sort_values(by="Ventas", ascending=False).head(20)
    
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(data=top_20, x="Salario", y="Ventas", hue="Nombre", ax=ax)
            ax.set_title("Relación entre salario y volumen de ventas (Top 20 empleados)")
            st.pyplot(fig)
    
            # Comparador entre dos empleados
            st.markdown("### 🤝 Comparador entre empleados")
            opciones = top_20["Nombre"].tolist()
            col1, col2 = st.columns(2)
            with col1:
                emp1 = st.selectbox("Empleado 1", opciones, key="emp1")
            with col2:
                emp2 = st.selectbox("Empleado 2", opciones, key="emp2")
    
            emp_data = top_20[top_20["Nombre"].isin([emp1, emp2])]
            fig2, ax2 = plt.subplots()
            sns.barplot(data=emp_data, x="Nombre", y="Ventas", ax=ax2, palette="viridis")
            ax2.set_title("Comparación de volumen de ventas entre empleados")
            st.pyplot(fig2)

        elif analisis_opcion == "👥 Perfil de cliente vs. tipo de producto vendido":
            st.markdown("### 👥 Perfil de cliente vs. tipo de producto vendido")
            st.markdown("🔎 ¿Qué revela el gráfico?\n- Analiza qué tipo de productos prefieren distintos perfiles de clientes según edad.\n- Permite identificar patrones de consumo, segmentaciones de marketing y oportunidades de fidelización.\n\n💡 Ideal para definir campañas específicas para cada grupo etario.")
        
            df_clientes = pd.read_csv("Clientes_transformados.csv")
            df_ventas = pd.read_csv("Venta_transformado.csv")
            df_productos = pd.read_csv("PRODUCTOS_transformado.csv")
        
            # Merge para cruzar cliente + venta + producto
            df_ventas = df_ventas.merge(df_clientes, left_on="IdCliente", right_on="ID", how="left")
            df_ventas = df_ventas.merge(df_productos[["ID_PRODUCTO", "Tipo"]], left_on="IdProducto", right_on="ID_PRODUCTO", how="left")
        
            # Crear grupos etarios
            df_ventas.dropna(subset=["Edad", "Tipo"], inplace=True)
            df_ventas["Edad_grupo"] = pd.cut(df_ventas["Edad"], bins=[0, 20, 35, 50, 100], labels=["≤20", "21-35", "36-50", ">50"])
        
            # Gráfico
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.countplot(data=df_ventas, x="Tipo", hue="Edad_grupo", ax=ax)
            ax.set_title("Tipo de producto vendido según grupo etario del cliente")
            ax.set_xlabel("Tipo de producto")
            ax.set_ylabel("Cantidad de ventas")
            ax.tick_params(axis='x', rotation=45)
            st.pyplot(fig)
        
        elif analisis_opcion == "🛒 Canal de venta vs. volumen/monto de ventas":
            st.markdown("### 🛒 Canal de venta vs. volumen/monto de ventas")
            st.markdown("🔎 ¿Qué revela el gráfico?\n- Compara el volumen y la distribución de ventas por canal.\n- Permite identificar cuál canal tiene mayor actividad o ingresos.\n\n💡 Útil para ajustar estrategias comerciales y reforzar canales más rentables.")
        
            # Carga de datasets
            df_ventas = pd.read_csv("Venta_transformado.csv")
            df_productos = pd.read_csv("PRODUCTOS_transformado.csv")
            df_canal = pd.read_csv("CanalDeVenta_Tranfor.csv")
        
            # Asegurar formatos consistentes
            df_ventas["IdCanal"] = df_ventas["IdCanal"].astype(str).str.strip()
            df_canal["CODIGO"] = df_canal["CODIGO"].astype(str).str.strip()
        
            # Merge para obtener nombre de canal y precio real
            df_ventas = df_ventas.merge(df_canal, left_on="IdCanal", right_on="CODIGO", how="left")
            df_ventas = df_ventas.merge(df_productos[["ID_PRODUCTO", "Precio"]].rename(columns={"Precio": "PrecioUnitario"}), left_on="IdProducto", right_on="ID_PRODUCTO", how="left")
        
            # Agrupamos por canal
            canal_resumen = df_ventas.groupby("DESCRIPCION").agg({
                "IdVenta": "count",
                "PrecioUnitario": "sum"
            }).reset_index().rename(columns={
                "IdVenta": "Total_Vendido",
                "PrecioUnitario": "Monto_Total"
            })
        
            # Visualización combinada
            fig, ax1 = plt.subplots(figsize=(10, 6))
            sns.barplot(data=canal_resumen, x="DESCRIPCION", y="Total_Vendido", ax=ax1, color="skyblue")
            ax1.set_ylabel("Cantidad de ventas", color="skyblue")
            ax1.set_xlabel("Canal de venta")
            ax1.set_title("Volumen y monto de ventas por canal")
            ax1.tick_params(axis='y', labelcolor="skyblue")
            plt.xticks(rotation=30)
        
            # Eje secundario para monto total
            ax2 = ax1.twinx()
            sns.lineplot(data=canal_resumen, x="DESCRIPCION", y="Monto_Total", ax=ax2, color="darkblue", marker="o")
            ax2.set_ylabel("Monto total ($)", color="darkblue")
            ax2.tick_params(axis='y', labelcolor="darkblue")
        
            st.pyplot(fig)
            
        elif analisis_opcion == "📊 Proveedor con mayor volumen de compra":
            st.markdown("### 📊 Proveedor con mayor volumen de compra")
            st.markdown("🔎 ¿Qué muestra el gráfico?\n- Permite identificar cuáles proveedores concentran mayor cantidad de productos adquiridos.\n- Ayuda a tomar decisiones sobre negociación, dependencia o diversificación de proveedores.\n\n💡 Ideal para compras estratégicas y análisis de riesgo.")
        
            df_compras = pd.read_csv("Compra_transformada.csv")
            df_proveedores = pd.read_csv("Proveedores_transformado.csv")
        
            # Agrupar por proveedor
            proveedor_resumen = df_compras.groupby("IdProveedor")["Cantidad"].sum().reset_index()
            proveedor_resumen = proveedor_resumen.merge(df_proveedores, left_on="IdProveedor", right_on="IDProveedor", how="left")
            proveedor_resumen = proveedor_resumen.sort_values(by="Cantidad", ascending=False).head(10)
        
            # Gráfico
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(data=proveedor_resumen, x="Nombre", y="Cantidad", ax=ax, palette="magma")
            ax.set_title("Top 10 proveedores por volumen de compra")
            ax.set_ylabel("Cantidad total de productos comprados")
            ax.set_xlabel("Proveedor")
            ax.tick_params(axis='x', rotation=45)
            st.pyplot(fig)
            
        elif analisis_opcion == "📈 Evolución histórica de ventas por canal":
            st.markdown("### 📈 Evolución histórica de ventas por canal")
            st.markdown("🔎 ¿Qué revela el gráfico?\n- Muestra cómo evolucionaron las ventas en el tiempo según el canal de comercialización.\n- Ayuda a detectar estacionalidades, tendencias de migración entre canales, y evaluar desempeño a largo plazo.\n\n💡 Ideal para planificación comercial y campañas estacionales.")
        
            df_ventas = pd.read_csv("Venta_transformado.csv")
            df_canal = pd.read_csv("CanalDeVenta_Tranfor.csv")
        
            df_ventas["Fecha"] = pd.to_datetime(df_ventas["Fecha"])
            df_ventas["IdCanal"] = df_ventas["IdCanal"].astype(str).str.strip()
            df_canal["CODIGO"] = df_canal["CODIGO"].astype(str).str.strip()
            df_ventas = df_ventas.merge(df_canal, left_on="IdCanal", right_on="CODIGO", how="left")
        
            df_ventas["Mes"] = df_ventas["Fecha"].dt.to_period("M").dt.to_timestamp()
            resumen = df_ventas.groupby(["Mes", "DESCRIPCION"]).size().reset_index(name="Cantidad")
        
            fig = px.line(
                resumen,
                x="Mes",
                y="Cantidad",
                color="DESCRIPCION",
                markers=True,
                title="Evolución mensual de ventas por canal",
                labels={"DESCRIPCION": "Canal de Venta", "Mes": "Fecha", "Cantidad": "Cantidad de Ventas"},
            )
            st.plotly_chart(fig, use_container_width=True)

        elif analisis_opcion == "💡 Comparar precios de compra vs. venta por producto (margen)":
            st.markdown("### 💡 Comparar precios de compra vs. venta por producto (margen)")
            st.markdown("🔎 ¿Qué muestra el gráfico?\n- Compara el precio promedio de compra y venta de cada producto.\n- Muestra el margen estimado por unidad.\n\n💡 Muy útil para análisis de rentabilidad por producto y toma de decisiones comerciales.")
        
            df_ventas = pd.read_csv("Venta_transformado.csv")
            df_compras = pd.read_csv("Compra_transformada.csv")
            df_productos = pd.read_csv("PRODUCTOS_transformado.csv")
        
            # Precio promedio de compra por producto
            compra_por_prod = df_compras.groupby("IdProducto")["Precio"].mean().reset_index(name="Precio_Compra")
        
            # Precio promedio de venta por producto
            venta_por_prod = df_ventas.groupby("IdProducto")["Precio"].mean().reset_index(name="Precio_Venta")
        
            # Merge de ambos
            comparacion = compra_por_prod.merge(venta_por_prod, on="IdProducto")
            comparacion = comparacion.merge(df_productos[["ID_PRODUCTO", "Concepto"]], left_on="IdProducto", right_on="ID_PRODUCTO")
            comparacion["Margen"] = comparacion["Precio_Venta"] - comparacion["Precio_Compra"]
            comparacion = comparacion.sort_values(by="Margen", ascending=False).head(10)
        
            # Gráfico
            fig, ax = plt.subplots(figsize=(10, 6))
            comparacion.set_index("Concepto")[["Precio_Compra", "Precio_Venta"]].plot(kind="bar", ax=ax)
            ax.set_title("Comparación de precios de compra vs. venta (Top 10 por margen)")
            ax.set_ylabel("Precio promedio por unidad")
            ax.set_xlabel("Producto")
            plt.xticks(rotation=45, ha="right")
            st.pyplot(fig)

##############                       
####  ML  ####
##############
    elif menu == "Modelos de ML":
        st.header("🤖 Modelos de Machine Learning")
    
        categoria = st.selectbox("📊 Elegí una categoría de datos:", [
            "🛍️ Compras",
            "🧾 Ventas",
            "👥 Empleados",
            "🧩 Sucursales",
            "💸 Gastos",
            "📦 Productos",
            "🚚 Proveedores",
            "🌐 Canal de ventas"
        ])
        # -----------------------------
        # COMPRAS
        # -----------------------------
        if categoria == "🛍️ Compras":
            st.subheader("🛍️ Predicción de demanda de productos")
            modelo = st.selectbox("Elegí un modelo de ML:", [
                "Regresión Lineal", "Random Forest", "ARIMA (Series Temporales)"
            ])
    
            @st.cache_data
            def load_compras():
                return pd.read_csv("Compra_transformada.csv", parse_dates=["Fecha"])
            
            df = load_compras()
    
            if modelo in ["Regresión Lineal", "Random Forest"]:
                df["mes"] = df["Fecha"].dt.month
                df["año"] = df["Fecha"].dt.year
    
                features = ["mes", "año"]
                if "IdProducto" in df.columns:
                    features.append("IdProducto")
                if "IdProveedor" in df.columns:
                    features.append("IdProveedor")
    
                X = df[features]
                y = df["Cantidad"]
    
                X = pd.get_dummies(X, columns=["IdProducto", "IdProveedor"], drop_first=True)
    
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42)
    
                if modelo == "Regresión Lineal":
                    model = LinearRegression()
                    st.markdown("#### 🧠 Sobre el modelo: Regresión Lineal")
                    st.markdown("""
                    Modelo simple que busca predecir la cantidad comprada a partir de variables como mes, año, producto y proveedor.  
                    Es útil para observar tendencias generales.
                    """)
                else:
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                    st.markdown("#### 🌲 Sobre el modelo: Random Forest")
                    st.markdown("""
                    Modelo basado en árboles de decisión, más robusto ante relaciones no lineales.  
                    Mejora la precisión en escenarios más complejos como compras por proveedor y producto.
                    """)
    
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
    
                try:
                    rmse = np.sqrt(mean_squared_error(y_test, np.ravel(y_pred)))
                    st.write(f"🔍 Error cuadrático medio (RMSE): {rmse:.2f}")
                except Exception as e:
                    st.error(f"❌ Error en cálculo de RMSE: {e}")
    
                try:
                    st.markdown("#### 📊 Comparación entre valores reales y predichos")
                    chart_df = pd.DataFrame({
                        "Real": y_test.values[:50],
                        "Predicho": np.ravel(y_pred)[:50]
                    })
                    st.line_chart(chart_df)
                except Exception as e:
                    st.error(f"❌ Error en gráfico: {e}")
    
            elif modelo == "ARIMA (Series Temporales)":
                st.info("Usando solo la serie temporal agregada total por mes.")
                st.markdown("#### ⏳ Sobre el modelo: ARIMA")
                st.markdown("""
                ARIMA es un modelo estadístico para series temporales que predice la cantidad total de productos comprados mes a mes, 
                a partir del comportamiento histórico de la demanda.
                """)
    
                df_ts = df.copy()
                df_ts = df_ts.set_index("Fecha").resample("M").sum(numeric_only=True)["Cantidad"]
    
                st.line_chart(df_ts)
    
                try:
                    model = sm.tsa.ARIMA(df_ts, order=(1, 1, 1))
                    results = model.fit()
                    forecast = results.forecast(steps=6)
    
                    st.write("📈 Predicción para los próximos 6 meses:")
                    st.line_chart(forecast)
                except Exception as e:
                    st.error(f"❌ Error en modelo ARIMA: {e}")
    
        # -----------------------------
        # VENTAS
        # -----------------------------
        elif categoria == "🧾 Ventas":
            st.subheader("🧾 Análisis de ventas: predicción y detección de outliers")
    
            tarea = st.radio("¿Qué querés hacer?", [
                "🔮 Predicción de ventas futuras",
                "🚨 Detección de outliers o fraudes"
            ])
    
            @st.cache_data
            def load_ventas():
                return pd.read_csv("Venta_transformado.csv", parse_dates=["Fecha"])
            
            df = load_ventas()
    
            df["mes"] = df["Fecha"].dt.month
            df["año"] = df["Fecha"].dt.year
    
            if tarea == "🔮 Predicción de ventas futuras":
                st.markdown("#### 🔮 Predicción de ventas con Regresión Ridge")
                st.markdown("""
                Se busca predecir la cantidad vendida usando Regresión Ridge, una técnica útil cuando hay muchas variables 
                correlacionadas (producto, canal, mes, año).
                """)
    
                features = ["mes", "año", "IdProducto", "IdCanal"]
                X = df[features]
                y = df["Cantidad"]
    
                X = pd.get_dummies(X, columns=["IdProducto", "IdCanal"], drop_first=True)
    
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42)
    
                model = Ridge(alpha=1.0)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
    
                try:
                    rmse = np.sqrt(mean_squared_error(y_test, np.ravel(y_pred)))
                    st.write(f"🔍 Error cuadrático medio (RMSE): {rmse:.2f}")
                except Exception as e:
                    st.error(f"❌ Error en cálculo de RMSE: {e}")
    
                try:
                    st.markdown("#### 📊 Comparación entre valores reales y predichos")
                    chart_df = pd.DataFrame({
                        "Real": y_test.values[:50],
                        "Predicho": np.ravel(y_pred)[:50]
                    })
                    st.line_chart(chart_df)
                except Exception as e:
                    st.error(f"❌ Error en gráfico: {e}")
    
            elif tarea == "🚨 Detección de outliers o fraudes":
                st.markdown("#### 🚨 Detección de outliers con Isolation Forest")
                st.markdown("""
                Isolation Forest detecta ventas inusuales en función de precio y cantidad.  
                Los puntos anómalos podrían ser errores de carga, promociones extremas o fraudes.
                """)
    
                df_filtrado = df[["Cantidad", "Precio"]].dropna()
                modelo_iso = IsolationForest(contamination=0.02, random_state=42)
                df_filtrado["anomaly"] = modelo_iso.fit_predict(df_filtrado)
                df_filtrado["color"] = df_filtrado["anomaly"].map({1: "Normal", -1: "Outlier"})
    
                st.markdown("#### 📌 Resultados de detección")
                st.write(df_filtrado["color"].value_counts())
    
                try:
                    fig = px.scatter(df_filtrado, x="Precio", y="Cantidad", color="color",
                                     title="Detección de outliers en ventas")
                    st.plotly_chart(fig)
                except Exception as e:
                    st.error(f"❌ Error en visualización: {e}")

        # -----------------------------
        # EMPLEADOS
        # -----------------------------
        elif categoria == "👥 Empleados":
            st.subheader("👥 Análisis de productividad y rendimiento")
        
            analisis = st.radio("Seleccioná el tipo de análisis:", [
                "🔍 Clusterización por rendimiento (K-means)",
                "🧠 Clasificación de alto rendimiento (Regresión logística)"
            ])
        
            @st.cache_data
            def load_empleados():
                return pd.read_csv("Empleados_transformados.csv")
        
            df = load_empleados()
        
            if analisis == "🔍 Clusterización por rendimiento (K-means)":
                st.markdown("#### 🔍 Agrupamiento de empleados según patrones comunes")
                st.markdown("""
                Usamos **K-means**, un algoritmo de clustering no supervisado, para identificar grupos de empleados con patrones similares
                según variables como **salario**, **sector**, **cargo** y **sucursal**. Esto permite detectar posibles desequilibrios,
                como empleados con sueldos altos en sectores menos productivos.
                """)
        
                from sklearn.preprocessing import StandardScaler
                from sklearn.cluster import KMeans
        
                # Codificar variables categóricas
                df_encoded = pd.get_dummies(df[["Salario", "Sucursal", "Sector", "Cargo"]], drop_first=True)
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(df_encoded)
        
                k = st.slider("Elegí el número de clusters", 2, 6, 3)
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                clusters = kmeans.fit_predict(X_scaled)
                df["Cluster"] = clusters
        
                st.write("### Distribución de empleados por cluster")
                st.write(df["Cluster"].value_counts().sort_index())
        
                try:
                    fig = px.scatter(df, x="Salario", y="Cluster", color="Sector", hover_data=["Cargo", "Sucursal"],
                                     title="Empleados agrupados por rendimiento relativo")
                    st.plotly_chart(fig)
                except Exception as e:
                    st.error(f"\u274c Error en visualización: {e}")
        
            elif analisis == "🧠 Clasificación de alto rendimiento (Regresión logística)":
                st.markdown("#### 🧠 Clasificación de empleados con alto rendimiento")
                st.markdown("""
                En este modelo simulamos una clasificación de empleados como **alto rendimiento** si están en el percentil superior
                de salario. Se entrena una **Regresión Logística** para predecir esta condición a partir de sector, sucursal y cargo.
                """)
        
                from sklearn.linear_model import LogisticRegression
                from sklearn.metrics import classification_report, confusion_matrix
        
                # Crear variable binaria de alto rendimiento
                salario_limite = df["Salario"].quantile(0.75)
                df["alto_rendimiento"] = (df["Salario"] > salario_limite).astype(int)
        
                X = pd.get_dummies(df[["Sucursal", "Sector", "Cargo"]], drop_first=True)
                y = df["alto_rendimiento"]
        
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                model = LogisticRegression(max_iter=500)
                model.fit(X_train, y_train)
        
                y_pred = model.predict(X_test)
                report = classification_report(y_test, y_pred, output_dict=True)
                cm = confusion_matrix(y_test, y_pred)
        
                st.write("### Matriz de confusión")
                st.write(pd.DataFrame(cm, index=["No Alto", "Alto"], columns=["Predicho No", "Predicho Alto"]))
        
                st.write("### Métricas de clasificación")
                st.json({
                    "Precisión (Clase Alta)": f"{report['1']['precision']:.2f}",
                    "Recall (Clase Alta)": f"{report['1']['recall']:.2f}",
                    "F1-score (Clase Alta)": f"{report['1']['f1-score']:.2f}"
                })


# -----------------------------
# SUCURSALES
# -----------------------------
        elif categoria == "🧩 Sucursales":
            st.subheader("🧩 Sucursales")
        
            submenu = st.radio("Seleccioná el tipo de análisis:", [
                "🧹 Cluster geográfico de sucursales",
                "📊 Clasificación por volumen de ventas"
            ])
        
            @st.cache_data
            def load_sucursales():
                return pd.read_csv("Sucursales_transformado.csv")
        
            @st.cache_data
            def load_ventas():
                return pd.read_csv("Venta_transformado.csv", parse_dates=["Fecha"])
        
            if submenu == "🧹 Cluster geográfico de sucursales":
                algoritmo = st.selectbox("Elegí el algoritmo de clusterización:", ["KMeans", "DBSCAN"])
                df = load_sucursales()
                coords = df[["Latitud", "Longitud"]].dropna()
        
                st.markdown("#### 📅 Objetivo del análisis")
                st.markdown("""
                Este análisis agrupa sucursales según su ubicación geográfica. 
                Se busca identificar áreas de concentración o zonas con comportamiento similar, 
                lo cual puede ser útil para tomar decisiones logísticas, comerciales o de expansión.
                """)
        
                if algoritmo == "KMeans":
        
                    st.markdown("#### 🔍 Clustering con KMeans")
                    st.markdown("""
                    KMeans divide las sucursales en un número fijo de grupos, buscando minimizar la distancia dentro de cada cluster. 
                    Es útil para ver agrupamientos específicos según cercanía.
                    """)
        
                    k = st.slider("Seleccioná la cantidad de clusters", 2, 6, 3)
                    scaler = StandardScaler()
                    coords_scaled = scaler.fit_transform(coords)
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    df["Cluster"] = kmeans.fit_predict(coords_scaled)
        
                elif algoritmo == "DBSCAN":
                    from sklearn.cluster import DBSCAN
                    from sklearn.preprocessing import StandardScaler
        
                    st.markdown("#### 🔎 Clustering con DBSCAN")
                    st.markdown("""
                    DBSCAN encuentra agrupamientos naturales basados en la densidad de puntos, sin necesidad de indicar la cantidad de clusters. 
                    Es útil para detectar zonas aisladas o con concentración geográfica alta.
                    """)
        
                    eps = st.slider("Seleccioná el radio de agrupamiento (eps)", 0.01, 1.0, 0.2, step=0.01)
                    min_samples = st.slider("Cantidad mínima de sucursales por grupo", 2, 10, 3)
        
                    scaler = StandardScaler()
                    coords_scaled = scaler.fit_transform(coords)
                    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                    df["Cluster"] = dbscan.fit_predict(coords_scaled)
        
                st.markdown("#### 🌍 Mapa de clusters geográficos")
                try:
                    fig = px.scatter_mapbox(
                        df,
                        lat="Latitud",
                        lon="Longitud",
                        color="Cluster",
                        hover_name="Sucursal",
                        zoom=4,
                        height=600,
                        mapbox_style="open-street-map",
                        title="Distribución de sucursales por cluster geográfico"
                    )
                    st.plotly_chart(fig)
                except Exception as e:
                    st.error(f"❌ Error al generar el mapa: {e}")
        
                st.markdown("#### 🔢 Análisis final")
                st.markdown("""
                El resultado de la clusterización permite observar patrones de agrupamiento espacial entre sucursales.
                - Si las sucursales están bien agrupadas, podrían compartirse logística, recursos o estrategias regionales.
                - Las sucursales aisladas o con comportamiento atípico podrían requerir un análisis individualizado o mejoras específicas.
                - Con DBSCAN, la detección de outliers espaciales puede ayudar a identificar sucursales que no pertenecen a ningún cluster estable,
                  lo que podría indicar una oportunidad de mejora o una estrategia personalizada.
                """)
        
            elif submenu == "📊 Clasificación por volumen de ventas":
                st.markdown("#### 📊 Agrupamiento de sucursales según nivel de ventas")
                st.markdown("""
                En este análisis usamos un modelo de árbol de decisión para predecir qué categoría de ventas tiene cada sucursal: 
                **Altas**, **Medias** o **Bajas**, basándonos en sus registros históricos.
                Esto permite entender qué variables (como ventas promedio, varianza o cantidad de registros) explican mejor su desempeño.
                """)
        
                df_ventas = load_ventas()
                ventas_por_sucursal = df_ventas.groupby("IdSucursal")["Cantidad"].agg([
                    ("TotalVentas", "sum"),
                    ("PromedioVentas", "mean"),
                    ("MaxVentas", "max"),
                    ("MinVentas", "min"),
                    ("Desvio", "std"),
                    ("CantidadRegistros", "count")
                ]).reset_index()
        
                # Crear etiquetas
                q1 = ventas_por_sucursal["TotalVentas"].quantile(0.33)
                q2 = ventas_por_sucursal["TotalVentas"].quantile(0.66)
        
                def clasificar(x):
                    if x < q1:
                        return "Bajas"
                    elif x < q2:
                        return "Medias"
                    else:
                        return "Altas"
        
                ventas_por_sucursal["Categoria"] = ventas_por_sucursal["TotalVentas"].apply(clasificar)
        
                X = ventas_por_sucursal.drop(columns=["Categoria", "IdSucursal"])
                y = ventas_por_sucursal["Categoria"]
        
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                model = DecisionTreeClassifier(max_depth=3, random_state=42)
                model.fit(X_train, y_train)
        
                y_pred = model.predict(X_test)
        
                st.write("### Matriz de confusión")
                st.write(pd.DataFrame(confusion_matrix(y_test, y_pred),
                                      index=model.classes_,
                                      columns=["Pred. " + c for c in model.classes_]))
        
                st.write("### Métricas de clasificación")
                st.text(classification_report(y_test, y_pred))
        
                df = load_sucursales()
                df = df.merge(ventas_por_sucursal[["IdSucursal", "Categoria"]], left_on="ID", right_on="IdSucursal", how="left")
        
                try:
                    st.markdown("#### 🌍 Mapa de sucursales por categoría de ventas")
                    fig = px.scatter_mapbox(
                        df,
                        lat="Latitud",
                        lon="Longitud",
                        color="Categoria",
                        hover_name="Sucursal",
                        zoom=4,
                        height=600,
                        mapbox_style="open-street-map",
                        title="Sucursal agrupadas por nivel de ventas"
                    )
                    st.plotly_chart(fig)
                except Exception as e:
                    st.error(f"\u274c Error al generar el mapa de ventas: {e}")
        
                st.markdown("#### 🔢 Análisis final")
                st.markdown("""
                Gracias al modelo de árbol de decisión pudimos identificar las variables que mejor explican el rendimiento de ventas por sucursal.
                Este análisis no solo agrupa, sino que ayuda a explicar y anticipar comportamientos según patrones históricos.
                Puede ser de gran valor para tomar decisiones de inversión o asignación de recursos.
                """)
       # -----------------------------
        # GASTOS
        # -----------------------------
        elif categoria == "💸 Gastos":
            st.subheader("💸 Análisis de gastos")
        
            submenu = st.radio("Seleccioná el tipo de análisis:", [
                "📊 Análisis general de gastos",
                "🏢 Análisis por sucursal",
                "🧾 Análisis por tipo de gasto"
            ])
        
            @st.cache_data
            def load_gastos():
                return pd.read_csv("Gasto_transformado.csv", parse_dates=["Fecha"])
        
            @st.cache_data
            def load_tipos_gasto():
                return pd.read_csv("TiposDeGasto_T.csv")
        
            @st.cache_data
            def load_sucursales():
                return pd.read_csv("Sucursales_transformado.csv")
        
            df = load_gastos()
            df_tipos = load_tipos_gasto()
            df_suc = load_sucursales()
        
            if submenu == "📊 Análisis general de gastos":
                st.markdown("#### 📊 Detección general de outliers con Isolation Forest")
        
                df = df.merge(df_tipos, left_on="IdTipoGasto", right_on="IdTipoGasto", how="left")
                df = df.merge(df_suc, left_on="IdSucursal", right_on="ID", how="left")
        
                df_filtrado = df[["Monto", "Descripcion", "Sucursal"]].dropna()
                modelo_iso = IsolationForest(contamination=0.05, random_state=42)
                df_filtrado["anomaly"] = modelo_iso.fit_predict(df_filtrado[["Monto"]])
                df_filtrado["color"] = df_filtrado["anomaly"].map({1: "Normal", -1: "Atípico"})
        
                st.markdown("#### 📌 Resumen de detecciones")
                st.dataframe(df_filtrado[df_filtrado["color"] == "Atípico"].sort_values(by="Monto", ascending=False))
        
                try:
                    fig = px.scatter(df_filtrado, x="Descripcion", y="Monto", color="color", 
                                     hover_data=["Sucursal"], title="Gastos detectados como atípicos por tipo")
                    st.plotly_chart(fig)
                except Exception as e:
                    st.error(f"❌ Error al generar el gráfico: {e}")
        
            elif submenu == "🏢 Análisis por sucursal":
                st.markdown("#### 🏢 Historial de gastos por sucursal con media general")
        
                df = df.merge(df_suc, left_on="IdSucursal", right_on="ID", how="left")
                df_grouped = df.groupby(["Fecha", "Sucursal"]).agg({"Monto": "sum"}).reset_index()
        
                sucursales_disp = df_grouped["Sucursal"].dropna().unique().tolist()
                sucursales_seleccionadas = st.multiselect("Seleccioná una o más sucursales:", sucursales_disp, default=sucursales_disp)
                df_grouped = df_grouped[df_grouped["Sucursal"].isin(sucursales_seleccionadas)]
        
                promedio = df_grouped.groupby("Fecha")["Monto"].mean().reset_index(name="MediaGeneral")
                df_plot = df_grouped.merge(promedio, on="Fecha")
        
                try:
                    fig = px.line(df_plot, x="Fecha", y="Monto", color="Sucursal",
                                  title="Evolución de gastos por sucursal", labels={"Monto": "Monto gastado"})
                    fig.add_scatter(x=df_plot["Fecha"], y=df_plot["MediaGeneral"], mode="lines",
                                    name="Media General", line=dict(color="black", dash="dash"))
                    st.plotly_chart(fig)
                except Exception as e:
                    st.error(f"❌ Error en gráfico histórico: {e}")
        
            elif submenu == "🧾 Análisis por tipo de gasto":
                st.markdown("#### 🧾 Outliers dentro de cada tipo de gasto")
        
                df = df.merge(df_tipos, left_on="IdTipoGasto", right_on="IdTipoGasto", how="left")
                df = df.merge(df_suc, left_on="IdSucursal", right_on="ID", how="left")
        
                tipos = df["Descripcion"].dropna().unique()
                tipo_seleccionado = st.selectbox("Seleccioná un tipo de gasto:", tipos)
        
                df_tipo = df[df["Descripcion"] == tipo_seleccionado]
                df_tipo = df_tipo[["Monto", "Sucursal"]].dropna()

                modelo_tipo = IsolationForest(contamination=0.05, random_state=42)
                df_tipo["anomaly"] = modelo_tipo.fit_predict(df_tipo[["Monto"]])
                df_tipo["color"] = df_tipo["anomaly"].map({1: "Normal", -1: "Atípico"})
        
                st.markdown(f"#### 📊 Detección de outliers en {tipo_seleccionado}")
                st.dataframe(df_tipo[df_tipo["color"] == "Atípico"])  # Mostrar detalle con sucursal
        
                try:
                    fig = px.scatter(df_tipo, x="Sucursal", y="Monto", color="color",
                                     title=f"Outliers detectados en {tipo_seleccionado}")
                    st.plotly_chart(fig)
                except Exception as e:
                    st.error(f"❌ Error al generar el gráfico: {e}")

        # -----------------------------
        # PRODUCTOS
        # -----------------------------
        elif categoria == "📦 Productos":
            st.subheader("📦 Análisis de productos")
        
            submenu = st.radio("Seleccioná el tipo de análisis:", [
                "🤝 Recomendación de productos",
                "📈 Predicción temporal de ventas",
                "🔝 Top 10 productos por mes"
            ])
        
            @st.cache_data
            def load_ventas():
                return pd.read_csv("Venta_transformado.csv", parse_dates=["Fecha"])
        
            @st.cache_data
            def load_productos():
                return pd.read_csv("PRODUCTOS_transformado.csv")
        
            df_ventas = load_ventas()
            df_productos = load_productos()
        
            if submenu == "🤝 Recomendación de productos":
                st.markdown("#### 🤝 Sistema de recomendación basado en volumen de compra")
                st.markdown("""
                Este sistema utiliza KNN (vecinos más cercanos) para encontrar productos relacionados 
                según los patrones de compra de los clientes. 
                Recomendamos productos similares al seleccionado, basándonos en clientes que compraron ambos.
                """)
        
                top_clientes = df_ventas.groupby(["IdProducto", "IdCliente"])["Cantidad"].sum().reset_index()
                tabla_recom = top_clientes.pivot(index="IdCliente", columns="IdProducto", values="Cantidad").fillna(0)
        
                producto_ids = tabla_recom.columns.tolist()
                producto_nombres = df_productos[df_productos["ID_PRODUCTO"].isin(producto_ids)][["ID_PRODUCTO", "Concepto"]].drop_duplicates()
                producto_opciones = producto_nombres.set_index("Concepto").to_dict()["ID_PRODUCTO"]
        
                producto_nombre_sel = st.selectbox("Seleccioná un producto:", list(producto_opciones.keys()))
                producto_id_sel = producto_opciones[producto_nombre_sel]
        
                model_knn = NearestNeighbors(metric="cosine", algorithm="brute")
                model_knn.fit(tabla_recom.T.values)
        
                index = producto_ids.index(producto_id_sel)
                distancias, indices = model_knn.kneighbors([tabla_recom.T.values[index]], n_neighbors=6)
        
                st.write("### Productos recomendados:")
                for i in range(1, len(indices[0])):
                    prod_id = producto_ids[indices[0][i]]
                    descripcion = df_productos[df_productos["ID_PRODUCTO"] == prod_id]["Concepto"].values
                    st.markdown(f"- {descripcion[0] if len(descripcion) else prod_id} (similaridad: {1 - distancias[0][i]:.2f})")
        
            elif submenu == "📈 Predicción temporal de ventas":
                st.markdown("#### 📈 Predicción de ventas futuras por producto (ARIMA)")
                st.markdown("""
                Este análisis muestra la evolución histórica de las ventas de un producto y proyecta su comportamiento
                para los próximos 6 meses utilizando un modelo ARIMA.
                """)
        
                # Merge para traer nombres de productos
                df_ventas = df_ventas.merge(df_productos, left_on="IdProducto", right_on="ID_PRODUCTO", how="left")
                productos_disp = df_ventas[["IdProducto", "Concepto"]].drop_duplicates()
        
                producto_nombre = st.selectbox("Seleccioná un producto:", productos_disp["Concepto"].tolist())
                producto_id = productos_disp[productos_disp["Concepto"] == producto_nombre]["IdProducto"].values[0]
        
                df_producto = df_ventas[df_ventas["IdProducto"] == producto_id]
                df_ts = df_producto.groupby("Fecha")["Cantidad"].sum().resample("M").sum().dropna()
        
                st.line_chart(df_ts)
        
                try:
                    model = sm.tsa.ARIMA(df_ts, order=(1, 1, 1))
                    results = model.fit()
                    forecast = results.forecast(steps=6)
        
                    st.markdown("#### 📊 Predicción para los próximos 6 meses")
                    st.line_chart(forecast)
                except Exception as e:
                    st.error(f"❌ Error al generar el modelo ARIMA: {e}")

            elif submenu == "🔝 Top 10 productos por mes":
                st.markdown("#### 🔝 Top 10 productos más vendidos por mes")
            
                # Unimos productos para obtener sus nombres
                df_ventas = df_ventas.merge(df_productos, left_on="IdProducto", right_on="ID_PRODUCTO", how="left")
            
                # Extraemos año y mes
                df_ventas["Año"] = df_ventas["Fecha"].dt.year
                df_ventas["Mes"] = df_ventas["Fecha"].dt.month
                df_ventas["MesNombre"] = df_ventas["Mes"].apply(lambda x: calendar.month_name[int(x)])
            
                # Filtros año y mes
                años_disponibles = sorted(df_ventas["Año"].dropna().unique())
                año_sel = st.selectbox("Seleccioná un año:", años_disponibles)
            
                meses_disponibles = df_ventas[df_ventas["Año"] == año_sel]["MesNombre"].dropna().unique().tolist()
                mes_nombre_sel = st.selectbox("Seleccioná un mes:", sorted(meses_disponibles, key=lambda x: list(calendar.month_name).index(x)))
                mes_sel = list(calendar.month_name).index(mes_nombre_sel)
            
                # Filtrar por año y mes
                df_filtrado = df_ventas[(df_ventas["Año"] == año_sel) & (df_ventas["Mes"] == mes_sel)]
            
                # TOP 10 productos más vendidos
                top10 = df_filtrado.groupby("Concepto")["Cantidad"].sum().sort_values(ascending=False).head(10).reset_index()
            
                st.markdown("##### 📊 Top 10 productos más vendidos")
                st.dataframe(top10)
            
                # Dispersión de todos los productos clasificados
                st.markdown("##### 📈 Dispersión de productos (clasificados por cantidad vendida)")
            
                resumen = df_filtrado.groupby("Concepto")["Cantidad"].sum().reset_index()
                promedio = resumen["Cantidad"].mean()
            
                def clasificar(cantidad):
                    if cantidad > promedio:
                        return "Más vendidos"
                    elif cantidad < promedio:
                        return "Menos vendidos"
                    else:
                        return "Promedio"
            
                resumen["Clasificación"] = resumen["Cantidad"].apply(clasificar)
            
                import plotly.express as px
                fig = px.scatter(
                    resumen,
                    x="Concepto",
                    y="Cantidad",
                    color="Clasificación",
                    title=f"Dispersión de ventas por producto - {mes_nombre_sel} {año_sel}",
                    labels={"Cantidad": "Cantidad Vendida", "Concepto": "Producto"}
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig)


####################
        ## Proveedores
        ##############
        elif categoria == "🚚 Proveedores":
                    st.subheader("🚚 Proveedores")
                
                    submenu = st.radio("Seleccioná el tipo de análisis:", [
                        "💰 Top 10 proveedores por gasto",
                    ])
            
        elif submenu == "💰 Top 10 proveedores por gasto":
        st.markdown("#### 💰 Top 10 proveedores por monto total de compra")
    
        @st.cache_data
        def load_compras():
            return pd.read_csv("Compra_transformada.csv", parse_dates=["Fecha"])
    
        @st.cache_data
        def load_proveedores():
            return pd.read_csv("Proveedores_transformado.csv")
    
        df_compras = load_compras()
        df_prov = load_proveedores()
    
        # 🔧 Unir el nombre del proveedor correctamente
        df = df_compras.merge(df_prov, left_on="IdProveedor", right_on="IDProveedor", how="left")
    
        # 🧮 Calcular top 10 por monto total
        top10_prov = df.groupby("Nombre")["Monto"].sum().sort_values(ascending=False).head(10).reset_index()
        st.markdown("##### 📋 Tabla: Top 10 proveedores por monto total")
        st.dataframe(top10_prov)
    
        st.markdown("##### 📈 Evolución mensual del gasto por proveedor")
    
        # 📅 Agrupar por mes
        df["Mes"] = df["Fecha"].dt.to_period("M").dt.to_timestamp()
        gasto_mensual = df.groupby(["Mes", "Nombre"])["Monto"].sum().reset_index()
    
        # 🔍 Filtro de selección múltiple
        proveedores_disp = top10_prov["Nombre"].tolist()
        proveedores_sel = st.multiselect(
            "Seleccioná uno o más proveedores:",
            options=proveedores_disp,
            default=proveedores_disp[:3]
        )
    
        # 📊 Gráfico
        import plotly.express as px
        fig = px.line(
            gasto_mensual[gasto_mensual["Nombre"].isin(proveedores_sel)],
            x="Mes", y="Monto", color="Nombre",
            labels={"Monto": "Monto de compra", "Mes": "Mes", "Nombre": "Proveedor"},
            title="Evolución mensual del gasto por proveedor"
        )
        fig.update_layout(xaxis_title="Mes", yaxis_title="Monto total")
        st.plotly_chart(fig)

        

    
################
#### MAPA ######
################

    
    elif menu == "Mapa de sucursales y empleados":
        st.header("🗺️ Mapa de sucursales y empleados")
    
        # Cargar los datos
        sucursales_df = pd.read_csv("Sucursales_transformado.csv")
        ventas_df = pd.read_csv("Venta_transformado.csv")
        empleados_df = pd.read_csv("Empleados_transformados.csv")
        productos_df = pd.read_csv("PRODUCTOS_transformado.csv")
    
        # Limpiar columnas
        sucursales_df.columns = sucursales_df.columns.str.strip()
        empleados_df.columns = empleados_df.columns.str.strip()
        ventas_df.columns = ventas_df.columns.str.strip()
    
        # Selector de sucursales
        sucursal_seleccionada = st.selectbox("Selecciona una sucursal", ["Todas"] + list(sucursales_df["Sucursal"].unique()))
    
        # Crear mapa base
        mapa = folium.Map(location=[sucursales_df["Latitud"].mean(), sucursales_df["Longitud"].mean()], zoom_start=5)
    
        if sucursal_seleccionada == "Todas":
            for _, row in sucursales_df.iterrows():
                folium.Marker([row["Latitud"], row["Longitud"]], popup=row["Sucursal"]).add_to(mapa)
        else:
            row = sucursales_df[sucursales_df["Sucursal"] == sucursal_seleccionada].iloc[0]
            folium.Marker([row["Latitud"], row["Longitud"]], popup=row["Sucursal"]).add_to(mapa)
    
        st_folium(mapa, width=700, height=500)
    
        # Empleados por sucursal
        st.subheader("👔 Empleados por sucursal")
        empleados_por_sucursal = empleados_df.groupby("Sucursal")["ID_empleado"].count().reset_index()
        st.bar_chart(empleados_por_sucursal.set_index("Sucursal"))
    
        if sucursal_seleccionada != "Todas":
            st.subheader(f"👥 Empleados en {sucursal_seleccionada}")
            empleados_sucursal = empleados_df[empleados_df["Sucursal"] == sucursal_seleccionada]
            st.dataframe(empleados_sucursal[["Nombre", "Apellido", "Cargo"]])
    
            if not empleados_sucursal.empty:
                empleado_seleccionado = st.selectbox("Selecciona un empleado", empleados_sucursal["Nombre"].unique())
    
                # Procesamiento de ventas
                ventas_df["Fecha"] = pd.to_datetime(ventas_df["Fecha"], errors="coerce")
                ventas_df = ventas_df[ventas_df["Fecha"] >= "2015-01-01"]
                ventas_df["Ventas_totales"] = ventas_df["Precio"] * ventas_df["Cantidad"]
    
                # Agrupación de ventas por empleado
                resumen_ventas = ventas_df.groupby("IdEmpleado")["Ventas_totales"].sum().reset_index()
                resumen_ventas = resumen_ventas.merge(empleados_df, left_on="IdEmpleado", right_on="ID_empleado", how="left")
    
                ventas_filtradas = resumen_ventas[
                    (resumen_ventas["Sucursal"] == sucursal_seleccionada) &
                    (resumen_ventas["Nombre"] == empleado_seleccionado)
                ]
    
                st.subheader(f"📈 Ventas de {empleado_seleccionado} desde 2015")
                st.write(ventas_filtradas[["Nombre", "Apellido", "Ventas_totales"]])

                # Gráfico
                fig = px.bar(ventas_filtradas, x="Nombre", y="Ventas_totales", color="Sucursal",
                             title=f"Ventas de {empleado_seleccionado} en {sucursal_seleccionada}")
                st.plotly_chart(fig)
    
                # Comparación de ventas totales por empleado
                st.subheader("Comparación de ventas por empleado")
                empleado_comparar = st.selectbox("Selecciona otro empleado para comparar", empleados_sucursal["Nombre"].unique())
        
                ventas_df["Ventas_totales"] = ventas_df["Precio"] * ventas_df["Cantidad"]
        
                ventas_empleados = ventas_df.merge(empleados_df, left_on="IdEmpleado", right_on="ID_empleado", how="left")
                ventas_filtradas = ventas_empleados[
                    (ventas_empleados["Sucursal"] == sucursal_seleccionada) &
                    (ventas_empleados["Nombre"].isin([empleado_seleccionado, empleado_comparar]))
                ]
        
                resumen_comparativo = ventas_filtradas.groupby("Nombre")["Ventas_totales"].sum().reset_index()
        
                fig = px.bar(resumen_comparativo, x="Nombre", y="Ventas_totales", color="Nombre",
                             title=f"Comparación de ventas totales en {sucursal_seleccionada}")
        
                st.plotly_chart(fig)

                # Comparación de todos los empleados de la sucursal
                st.subheader("Ventas por empleado en la sucursal")
                ventas_sucursal = ventas_empleados[ventas_empleados["Sucursal"] == sucursal_seleccionada]
                resumen_sucursal = ventas_sucursal.groupby("Nombre")["Ventas_totales"].sum().reset_index().sort_values(by="Ventas_totales", ascending=False)
        
                fig_all = px.bar(resumen_sucursal, x="Nombre", y="Ventas_totales", color="Nombre",
                                 title=f"Ventas totales por empleado en {sucursal_seleccionada}")
                st.plotly_chart(fig_all)
        
    else:
        st.warning("🔒 Ingresá la clave correcta para acceder a la app")
