import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn.objects as so
import folium
from streamlit_folium import st_folium

# -----------------------------
# CONFIGURACION INICIAL
# -----------------------------
st.set_page_config(page_title="📊 DataEnterprise", page_icon="🏢", layout="wide")

# -----------------------------
# LOGIN SIMPLE
# -----------------------------
st.title("🔐 Acceso privado")

password = st.text_input("Ingresá la clave para acceder a la app:", type="password")

# Verificamos clave contra secrets
if password == st.secrets["acceso"]["clave"]:
    st.success("Acceso concedido ✅")

    # -----------------------------
    # MENU PRINCIPAL
    # -----------------------------
    menu = st.sidebar.selectbox("📂 Secciones", [
        "Inicio",
        "Análisis exploratorio",
        "Análisis cruzado",
        "Modelos de ML",
        "Mapa de sucursales",
        "Descargas"
    ])

    st.sidebar.markdown("---")
    st.sidebar.markdown("👤 Usuario: dueño")

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
            st.markdown("- Margen positivo en productos más vendidos.\n- Algunos con exceso de stock o rotación lenta.\n- Conclusión: clave ajustar precios y foco de venta.")
            st.image("graficos/productos_margen.png")

        elif dataset_opcion == "Proveedores":
            st.subheader("🏭 Exploración de Proveedores")
            st.markdown("- Dependencia de pocos proveedores clave.\n- Variabilidad en precios y frecuencia de compra.\n- Conclusión: posible mejora en condiciones de negociación.")
            st.image("graficos/proveedores_top.png")

        elif dataset_opcion == "Sucursales":
            st.subheader("🏢 Exploración de Sucursales")
            st.markdown("- Desempeño desigual entre sedes.\n- Diferencias marcadas en volumen de ventas y gastos.\n- Conclusión: requiere gestión individualizada y auditoría local.")
            st.image("graficos/sucursales_rendimiento.png")

        elif dataset_opcion == "Ventas":
            st.subheader("💰 Exploración de Ventas")
            st.markdown("- Canales online y tienda física dominan.\n- Picos de venta en ciclos mensuales.\n- Conclusión: se puede predecir estacionalidad y optimizar promociones.")
            st.image("graficos/ventas_canal.png")

    elif menu == "Análisis cruzado":
        st.header("🔀 Análisis cruzado entre áreas")
        st.info("Próximamente: visualización de los 8 análisis clave")

    elif menu == "Modelos de ML":
        st.header("🤖 Modelos de Machine Learning")
        st.info("Próximamente: predicción de ventas, segmentación, recomendaciones...")

    elif menu == "Mapa de sucursales":
        st.header("🗺️ Visualización geográfica")
        st.info("Próximamente: integración del mapa interactivo de sucursales")

    elif menu == "Descargas":
        st.header("📥 Exportación de datos y resultados")
        st.info("Próximamente: descarga de reportes, gráficos y predicciones")

else:
    st.warning("🔒 Ingresá la clave correcta para acceder a la app")
