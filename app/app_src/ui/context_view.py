import streamlit as st

def render_context_view():
    # ==========================================
    # 1. INTRODUCCIÓN
    # ==========================================
    st.markdown("""
        <p style="font-size: 18px; line-height: 1.6; text-align: justify; margin-bottom: 25px;">
        Esta plataforma es el resultado de un <strong>Trabajo Fin de Máster en Ciencia de Datos por la Universidad de Granada</strong>, orientado al desarrollo de un sistema automático y explicable para la clasificación de tipologías urbanas.
        <br><br> 
        El proyecto se desarrolla a partir  de la base de datos <em>"Complejos residenciales del área metropolitana de Granada"</em> [1], concebida para el estudio sociológico  de la <strong>fragmentación urbana</strong>.
        Un fenómeno urbanístico en el que el desarrollo de la ciudad no se produce de forma continua, sino mediante unidades residenciales que buscan diferenciarse y protegerse del entorno exterior.
        <br><br>
        Su origen se sitúa en la ciudad industrial de finales del siglo XIX y comienzos del XX, principalmente en Inglaterra y Estados Unidos, a través de un subtipo específico de urbanización caracterizado por un elevado nivel de aislamiento físico y social: las gated communities (comunidades cerradas).
        Sin embargo, su expansión y adaptación en diferentes contextos geográficos y culturales ha dado lugar a una amplia diversidad morfológica que motiva su estudio en profundidad.   
        </p>
        """,
    unsafe_allow_html=True)

    st.divider()

    # ==========================================
    # 2. MARCO SOCIOLÓGICO
    # ========================================== 
    st.subheader("DESCRIPCIÓN DE LA BASE DE DATOS")  
    st.markdown("""
        <style>

        /* =========================
        CONTENEDOR PRINCIPAL
        ========================= */
        .container-yellow {
            background-color: #fffbea;   /* amarillo muy pálido */
            border: 1px solid #000000;
            border-radius: 14px;
            padding: 22px;
            font-size: 18px;
            line-height: 1.6;
            text-align: justify;
        }
        .container-yellow ul,
        .container-yellow li {
            font-size: 18px;
            line-height: 1.6;
        }
        /* =========================
        GRID 2 COLUMNAS
        ========================= */
        .grid-2 {
            display: flex;
            gap: 18px;
            margin-top: 20px;
        }

        /* columnas */
        .col {
            flex: 1;
        }

        /* =========================
        CARD INTERNA
        ========================= */
        .card-box {
            background-color: rgba(255,255,255,0.75);
            border: 1px solid rgba(0,0,0,0.35);
            border-radius: 12px;
            padding: 16px;
            height: 100%;
        }

        /* título card */
        .card-title {
            font-weight: bold;
            font-size: 16px;
            margin-bottom: 8px;
        }

        /* línea separadora */
        .divider {
            border: none;
            border-top: 1px solid rgba(0,0,0,0.2);
            margin: 8px 0 12px 0;
        }

        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown("""
        <div class="container-yellow">
        La base de datos registra un censo de <b>642 complejos residenciales</b>, identificados hasta 2022, que representan el <b>13% del parque de viviendas</b> del área metropolitana de Granada.
        <br><br>
        Se define como <b>complejo residencial</b> a un desarrollo urbanístico que dispone de:
        <ul>
        <li><b>Espacios comunes.</b> Instalaciones comunitarias internas que funcionan como espacios de interacción social para sus residentes.</li>
        <li><b>Elementos de separación.</b> Mecanismos que delimitan la vida comunitaria interior del entorno urbano exterior.</li>
        </ul>
        Cada <b>complejo residencial</b> queda definido a través de variables que describen su morfología urbana (variables predictoras) y una etiqueta que determina su clasificación tipológica (variable objetivo).

        <!-- =========================
            DOS COLUMNAS
        ========================= -->
        <div class="grid-2">
            <!-- COLUMNA 1 -->
            <div class="col">
                <div class="card-box">
                    <div class="card-title">MORFOLOGÍA URBANA</div>
                    <hr class="divider">
                    Se han registrado <b>25 variables</b> relacionadas con el grado de cerramiento agrupadas en 6 bloques semánticos:
                    <ul>
                        <li><b>🧱 Aspectos estructurales:</b> estructura viaria interna y tipo de comercios.</li>
                        <li><b>📍 Distancia al núcleo urbano:</b> situación geográfica del complejo.</li>
                        <li><b>⛔ Elementos de cerramiento:</b> muros, verjas o carteles de propiedad privada.</li>
                        <li><b>🚪 Puntos de acceso:</b> forma de acceso de los residentes.</li>
                        <li><b>🚧 Uso de la vía pública:</b> tipo de uso del espacio público.</li>
                        <li><b>🛡️ Seguridad y vigilancia:</b> mecanismos de control y seguridad.</li>
                    </ul>
                </div>
            </div>
            <!-- COLUMNA 2 -->
            <div class="col">
                <div class="card-box">
                    <div class="card-title">GRADO DE CERRAMIENTO</div>
                    <hr class="divider">
                    La tipología urbana utilizada para clasificar los complejos residenciales es el <b>grado de cerramiento</b>, entendido como la forma en la que el complejo se delimita y se relaciona con el entorno urbano.
                    <ul>
                        <li>🔴 <b>Protegido:</b> Cerramiento intenso que impide el acceso de personas no residentes.</li>
                        <li>🟠 <b>Controlado:</b> Acceso supervisado por residentes.</li>
                        <li>🟣 <b>Autoaislado:</b> Aislamiento respecto al tejido urbano.</li>
                        <li>🔵 <b>Individualista:</b> Accesos privados e individualizados.</li>
                        <li>🟢 <b>Simbólico:</b> Cerramiento sutil o inexistente.</li>
                    </ul>
                </div>
            </div>
        </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()
    
    # ==========================================
    # 2. OBJETIVOS DEL PROYECTO
    # ==========================================
    st.subheader("ESTUDIO EXPERIMENTAL Y ESTRUCTURA DE LA PLATAFORMA")  
    st.markdown("""
        <style>

        /* =========================
        CONTENEDOR PRINCIPAL AZUL
        ========================= */
        .container-blue {
            background-color: #f1f6ff;   /* azul claro */
            border: 1px solid #000000;
            border-radius: 14px;
            padding: 22px;
            font-size: 18px;
            line-height: 1.6;
            text-align: justify;
        }

        /* asegurar tamaño uniforme */
        .container-blue p,
        .container-blue ul,
        .container-blue li {
            font-size: 18px;
            line-height: 1.6;
        }

        /* =========================
        GRID DOS COLUMNAS
        ========================= */
        .grid-2 {
            display: flex;
            gap: 18px;
            margin-top: 20px;
        }

        /* columnas */
        .col {
            flex: 1;
        }

        /* =========================
        CARD INTERNA
        ========================= */
        .card-box {
            background-color: rgba(255, 255, 255, 0.78);
            border: 1px solid rgba(0,0,0,0.35);
            border-radius: 12px;
            padding: 16px;
        }

        /* título */
        .card-title {
            font-weight: bold;
            font-size: 20px;
            margin-bottom: 8px;
        }

        /* separador */
        .divider {
            border: none;
            border-top: 1px solid rgba(0,0,0,0.2);
            margin: 8px 0 12px 0;
        }

        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown("""
        <div class="container-blue">
        Se ha llevado a cabo un estudio comparativo entre algoritmos de aprendizaje supervisado con diferentes niveles de complejidad e interpretabilidad: Regresión Logística, Árboles de Decisión, Support Vector Machines (SVM), Random Forest y XGBoost.

        Los resultados de la experimentación han mostrado que <b>XGBoost</b> es el algoritmo más adecuado para este problema, al alcanzar el mejor rendimiento y mantener un elevado nivel de interpretabilidad mediante el uso de <b>SHAP</b>. Esta técnica de Inteligencia Artificial Explicable (post-hoc XAI) permite traducir las decisiones del modelo a explicaciones comprensibles y coherentes con el conocimiento del dominio.

        La plataforma desarrollada funciona tanto como herramienta de análisis de las predicciones y explicaciones del modelo sobre el censo de complejos residenciales del área metropolitana de Granada como herramienta predictiva para el estudio de nuevos desarrollos urbanísticos.
        <!-- =========================
            COLUMNAS
        ========================= -->
        <div class="grid-2">
            <!-- COLUMNA 1 -->
            <div class="col">
                <div class="card-box">
                    <div class="card-title">INTERFAZ DE EXPLICABILIDAD</div>
                    <hr class="divider">
                    Permite explorar las predicciones y explicaciones SHAP de todos los complejos residenciales de la base de datos.
                    <br><br>
                    Para garantizar la validez del análisis, se ha seguido una estrategia out-of-fold, asegurando que el modelo que predice y explica cada complejo residencial no ha utilizado previamente dicha muestra durante su entrenamiento.
                </div>
            </div>
            <!-- COLUMNA 2 -->
            <div class="col">
                <div class="card-box">
                    <div class="card-title">INTERFAZ DE PREDICCIÓN</div>
                    <hr class="divider">
                    Un simulador en tiempo real que implementa el modelo XGBoost entrenado sobre toda la base de datos.
                    <br><br>
                    Permite diseñar nuevos complejos residenciales, predecir automáticamente su grado de cerramiento y analizar las variables morfológicas urbanas que influyen en dicha clasificación mediante explicaciones SHAP.
                </div>
            </div>
        </div>
        </div>
        """,
        unsafe_allow_html=True
    )