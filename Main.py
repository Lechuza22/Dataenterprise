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
            "🔁 Tipo de gasto más frecuente por sucursal",
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

           # Asegurar tipo consistente
            df_compras["IdProveedor"] = df_compras["IdProveedor"].astype(int)
            df_proveedores["IDProveedor"] = df_proveedores["IDProveedor"].astype(int)
            
            # Agrupar compras por proveedor
            compras_por_proveedor = df_compras.groupby("IdProveedor").agg({
                "Cantidad": "sum",
                "Precio": "sum"
            }).reset_index().rename(columns={"Cantidad": "Total_Comprado", "Precio": "Monto_Total"})
            
            # Unir con nombre del proveedor
            compras_con_nombre = compras_por_proveedor.merge(
                df_proveedores[["IDProveedor", "Nombre"]],
                left_on="IdProveedor", right_on="IDProveedor", how="left"
            )
            
            # Seleccionar top 10 por monto total
            top_proveedores = compras_con_nombre.sort_values(by="Monto_Total", ascending=False).head(10)
            
              # Gráfico
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(data=proveedor_resumen, x="Nombre", y="Cantidad", ax=ax, palette="magma")
            ax.set_title("Top 10 proveedores por volumen de compra")
            ax.set_ylabel("Cantidad total de productos comprados")
            ax.set_xlabel("Proveedor")
            ax.tick_params(axis='x', rotation=45)
            st.pyplot(fig)
        


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
