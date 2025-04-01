# 📊 DataEnterprise

**DataEnterprise** es una aplicación empresarial interactiva desarrollada en **Streamlit**, diseñada para la **exploración, visualización y modelado predictivo de datos clave de una organización**. Está especialmente pensada para empresas que desean tomar decisiones estratégicas basadas en datos reales y actualizados.

---

## 🚀 ¿Qué ofrece esta app?

DataEnterprise integra múltiples herramientas de análisis de datos y Machine Learning en una sola plataforma intuitiva y segura. Entre sus funcionalidades principales se destacan:

### 🔍 Exploración de Datos (EDA)

Visualización rápida y profunda de:
- Clientes: edad, localización, comportamiento.
- Compras: productos adquiridos, volúmenes, precios.
- Empleados: salarios, cargos, sectores.
- Gastos: evolución, tipos, sucursales.
- Ventas: frecuencia, volumen, canal.
- Proveedores y productos: relaciones y análisis de stock.
- Sucursales: mapa geográfico y comparativos por provincia.

### 🔁 Análisis Cruzado

Relaciones estratégicas entre áreas clave como:
- Compras vs. Ventas.
- Sucursales con más ventas vs. más gastos.
- Salario vs. volumen de ventas por empleado.
- Perfil de clientes vs. tipo de producto.
- Comparación de canales de venta y márgenes.

### 🤖 Modelos de Machine Learning

Aplicación práctica de algoritmos para predecir, clasificar y detectar anomalías:
- **Regresión Lineal, Random Forest y ARIMA** para predicción de demanda y ventas.
- **KMeans y DBSCAN** para clusterización de sucursales y canales.
- **Isolation Forest y Huber Regressor** para control de gastos y outliers.
- **Regresión Logística y Árboles de Decisión** para segmentar empleados y sucursales.
- **KNN y Content-based filtering** para sistemas de recomendación de productos y proveedores.

### 🌍 Mapa Interactivo

Visualización geográfica de:
- Sucursales y empleados.
- Performance comercial.
- Proximidad de proveedores y análisis logístico.

---

## 🔐 Acceso seguro

Cuenta con un sistema de login básico con contraseña para proteger la información sensible de la empresa.

---

## 🧑‍💼 Público objetivo

Esta app está orientada a:
- Gerentes generales y financieros.
- Responsables de compras, ventas y logística.
- Equipos de marketing y recursos humanos.
- Analistas de datos corporativos.

---

## 🧰 Stack tecnológico

- **Python**
- **Streamlit**
- **Pandas / NumPy / Seaborn / Matplotlib / Plotly / Scikit-learn / Statsmodels**
- **Folium** para mapas interactivos
- **ARIMA / KMeans / Isolation Forest / KNN / DBSCAN** y otros algoritmos de ML

---

## 📦 Instalación rápida

1. Clonar el repositorio:
   ```bash
   git clone https://github.com/tu_usuario/dataenterprise.git
   cd dataenterprise
   ```

2. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```

3. Crear el archivo `secrets.toml` dentro del directorio `.streamlit/`:
   ```toml
   [acceso]
   clave = "35533202"
   ```

4. Ejecutar la app:
   ```bash
   streamlit run Main.py
   ```

---

## 📈 Resultado

Una plataforma unificada de análisis integral que potencia la toma de decisiones **basadas en datos concretos** y **modelos predictivos confiables**.
