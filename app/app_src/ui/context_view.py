"""
Streamlit Context interface: Gives a global overview to the context of the project.
"""

# ==============================================================================
# IMPORTS
# ==============================================================================
import streamlit as st

def render_context_view():
    """
    Renders an aesthetic, structured interface that explains the context,
    database and objectives of the project.
    
    Parameters
    ----------
    None
    
    Returns
    -------
    None
    """
    # ==========================================
    # CONTEXT
    # ========================================== 
    st.subheader("INTRODUCCIÓN AL PROBLEMA")
    st.markdown("""
        <p style="font-size: 18px; line-height: 1.6; text-align: justify; margin-bottom: 25px;">
        La <b>fragmentación urbana</b> es un fenómeno socioespacial en el que el crecimiento de las ciudades no se produce de forma continua,
        sino mediante unidades residenciales que buscan diferenciarse y protegerse del entorno exterior. 
        Su origen se sitúa en la ciudad industrial de finales del siglo XIX y comienzos del XX, principalmente en Inglaterra y Estados Unidos, a través de un tipo específico de urbanización caracterizado por un elevado nivel de aislamiento físico y social: las <i>gated communities</i> (comunidades cerradas).
        La expansión y adaptación de este modelo a diferentes contextos geográficos y culturales ha dado lugar a una amplia diversidad morfológica que requiere de herramientas avanzadas para su clasificación y análisis.
        <br><br>
        Esta plataforma implementa un <b>sistema automático y explicable (XAI)</b> capaz de clasificar la tipología úrbana que representa el grado de cerramiento de un desarrollo urbanístico.
        El sistema ha sido entrenado y validado sobre la base de datos <em>"Complejos residenciales del área metropolitana de Granada"</em> [1], concebida para el estudio sociológico de la <b>fragmentación urbana</b> [2].   
        </p>

        <p style="font-size: 14px; color: gray; margin-top: 0px;">
        [1] Baldán Lozano, H., & Susino Arbucias, J. (2024).
        <i>Base de datos de complejos residenciales en el área metropolitana de Granada</i>.
        Universidad de Granada.
        <a href="https://hdl.handle.net/10481/105464" target="_blank">
        https://hdl.handle.net/10481/105464
        </a>
        </p>
        <p style="font-size: 14px; color: gray; margin-top: 5px;">
        [2] Baldán Lozano, H. (2025).
        <i>Formas de fragmentación urbana: los complejos residenciales en el área metropolitana de Granada</i>.
        Universidad de Granada.
        <a href="https://hdl.handle.net/10481/105368" target="_blank">
        https://hdl.handle.net/10481/105368
        </a>
        </p>
        """,
    unsafe_allow_html=True)
    st.markdown(
        """
        <div style='
            background-color: #f8fafc; 
            border-left: 4px solid #94a3b8; 
            padding: 12px 16px; 
            border-radius: 4px; 
            margin-top: 10px;
        '>
            <p style='margin: 0; font-size: 1rem; color: #475569; font-style: italic;'>
                Este proyecto ha sido desarrollado como Trabajo Fin de Máster en <i>Ciencia de Datos e Ingenieria de Computadores</i> por la Universidad de Granada.
            </p>
        </div>
        """, 
        unsafe_allow_html=True
    )
    st.divider()

    # ==========================================
    # DATABASE
    # ==========================================  
    st.subheader("DESCRIPCIÓN DE LA BASE DE DATOS")  
    st.markdown("""
        <style>
        .container-yellow {
            background-color: #eef6f0; 
            backdrop-filter: blur(6px);
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
        GRID 2 COLUMNS
        ========================= */
        .grid-2 {
            display: flex;
            gap: 18px;
            margin-top: 20px;
        }

        /* columns */
        .col {
            flex: 1;
        }

        /* =========================
        INTERNAL CARD
        ========================= */
        .card-box {
            background-color: rgba(255,255,255,0.75);
            border: 1px solid rgba(0,0,0,0.35);
            border-radius: 12px;
            padding: 16px;
            height: 100%;
        }

        /* CART TITLE*/
        .card-title {
            font-weight: bold;
            font-size: 16px;
            margin-bottom: 8px;
        }

        /* SPLIT LINE */
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
        La base de datos registra un censo de 642 <b>complejos residenciales</b>, identificados hasta 2022, que representan el 13% del parque de viviendas del área metropolitana de Granada.
        Se define como <b>complejo residencial</b> a un desarrollo urbanístico que dispone de:
        <ul>
        <li><b>Espacios comunes.</b> Instalaciones comunitarias internas que funcionan como espacios de interacción social para sus residentes.</li>
        <li><b>Elementos de separación.</b> Mecanismos que delimitan la vida comunitaria interior del entorno urbano exterior.</li>
        </ul>
        Para cada <b>complejo residencial</b> se han registrado variables que describen su morfología urbana (variables predictoras) y una etiqueta que determina su clasificación tipológica (variable objetivo).

        <div class="grid-2">
            <!-- COLUMN 1 -->
            <div class="col">
                <div class="card-box">
                    <div class="card-title">MORFOLOGÍA URBANA</div>
                    <hr class="divider">
                    Se dispone de 25 variables relacionadas con el <b>grado de cerramiento</b>, agrupadas en seis bloques semánticos:
                    <ul>
                        <li><b>🧱 Aspectos estructurales.</b> Estructura viaria interna y tipo de comercios.</li>
                        <li><b>📍 Distancia al núcleo urbano.</b> Situación geográfica del complejo residencial.</li>
                        <li><b>⛔ Elementos de cerramiento.</b> Tipos de cerramientos físicos (muros, verjas, etc).</li>
                        <li><b>🚪 Puntos de acceso.</b> Forma de acceso de los residentes.</li>
                        <li><b>🚧 Uso de la vía pública.</b> Tipo de uso del espacio viario.</li>
                        <li><b>🛡️ Seguridad y vigilancia.</b> Mecanismos o servicios de control.</li>
                    </ul>
                </div>
            </div>
            <!-- COLUMN 2 -->
            <div class="col">
                <div class="card-box">
                    <div class="card-title">GRADO DE CERRAMIENTO</div>
                    <hr class="divider">
                    La tipología urbana utilizada para clasificar los complejos residenciales es el <b>grado de cerramiento</b>, entendido como la forma en la que el complejo se delimita y se relaciona con el entorno urbano.
                    <ul>
                        <li>🔴 <b>Protegido:</b> Cerramiento intenso que impide el acceso de personas no residentes.</li>
                        <li>🟠 <b>Controlado:</b> Cerramiento intenso con acceso supervisado por residentes.</li>
                        <li>🟣 <b>Autoaislado:</b> Cerramiento definido por el aislamiento respecto al tejido urbano.</li>
                        <li>🔵 <b>Individualista:</b> Cerramiento mediante accesos privados e individualizados.</li>
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
    # EXPERIMENTAL RESULTS
    # ==========================================
    st.subheader("ESTUDIO EXPERIMENTAL Y ESTRUCTURA DE LA PLATAFORMA")  
    st.markdown("""
        <style>

        .container-blue {
            background-color: rgba(244, 247, 251, 0.92);
            backdrop-filter: blur(6px); 
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
        GRID TWO COLUMNS
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
        INTERNAL CARD
        ========================= */
        .card-box {
            background-color: rgba(255, 255, 255, 0.78);
            border: 1px solid rgba(0,0,0,0.35);
            border-radius: 12px;
            padding: 16px;
        }

        .card-title {
            font-weight: bold;
            font-size: 20px;
            margin-bottom: 8px;
        }
                
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
        Se ha diseñado e implementado un estudio experimental comparativo para evaluar algoritmos de aprendizaje supervisado con diferentes niveles de complejidad e interpretabilidad: <i>Regresión Logística, Árboles de Decisión, Support Vector Machines (SVM), Random Forest y XGBoost</i>.
        El objetivo de este estudio ha sido el análisis del compromiso (<i>trade-off</i>) entre la capacidad de generalización y la interpretabilidad del modelo.
        <br><br>
        Los resultados de la experimentación han demostrado que <b>XGBoost</b> es el algoritmo más adecuado para este problema, obteniendo el mejor rendimiento y consiguiendo un elevado nivel de interpretabilidad mediante el uso de <b>SHAP</b>. Esta técnica de Inteligencia Artificial Explicable (post-hoc XAI) permite traducir las decisiones del modelo a explicaciones comprensibles y coherentes con el conocimiento del dominio.
        <br><br>
        La plataforma desarrollada integra dos funcionalidades principales: un entorno de análisis e interpretación de las predicciones realizadas sobre el censo de complejos residenciales del área metropolitana de Granada, y una herramienta de simulación orientada a la evaluación de nuevos escenarios urbanísticos.
        
        <div class="grid-2">
            <!-- COLUMN 1 -->
            <div class="col">
                <div class="card-box">
                    <div class="card-title">🗺️ EXPLORADOR DE COMPLEJOS RESIDENCIALES</div>
                    <hr class="divider">
                    Esta interfaz permite explorar las predicciones y explicaciones SHAP de todos los complejos residenciales incluidos en la base de datos.
                    <br><br>
                    Para garantizar la validez de los resultados, se ha seguido una estrategia <i>out-of-fold</i>, asegurando que el modelo que predice y explica cada complejo residencial no ha utilizado previamente dicha muestra durante su entrenamiento.
                </div>
            </div>
            <!-- COLUMN 2 -->
            <div class="col">
                <div class="card-box">
                    <div class="card-title">🏗️ SIMULADOR DE NUEVOS ESCENARIOS</div>
                    <hr class="divider">
                    Esta interfaz actúa como un entorno de inferencia que utiliza el modelo XGBoost para estudiar el grado de cerramiento de nuevos desarrollos urbanísticos.
                    <br><br>
                    La herramienta permite diseñar nuevos complejos residenciales, predecir su grado de cerramiento y analizar el impacto de las variables morfológicas urbanas en dicha clasificación mediante la explicabilidad proporcionada por SHAP.
                </div>
            </div>
        </div>
        </div>
        """,
        unsafe_allow_html=True
    )