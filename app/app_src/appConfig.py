from dataclasses import dataclass, field
from pathlib import Path

@dataclass(frozen=True)
class AppConfig:

    ####################
    # PATHS
    #####################
    ROOT_DIR: Path = Path(__file__).resolve().parents[2]
    
    # Data folders 
    DATA_DIR: Path = ROOT_DIR / "data"
    DATA_PROCESSED_DIR: Path = DATA_DIR / "processed"
    DATASET_PATH = DATA_PROCESSED_DIR / "model_data.csv"
    DESCRIPTIVE_DATA_PATH = DATA_PROCESSED_DIR / "descriptive_data.csv"

    # Output folder
    OUTPUT_DIR: Path = ROOT_DIR /"output"

    MODELS_DIR: Path = OUTPUT_DIR / "models"
    XGBOOST_MODEL_PATH = MODELS_DIR / "XGBoost_model.pkl"
    
    SHAP_LOCAL_DIR: Path = OUTPUT_DIR / "SHAP_local"
    XGBOOST_DIR: Path = SHAP_LOCAL_DIR / "XGBoost"

    PRED_INFO_PATH = XGBOOST_DIR / "XGBoost_pred_info.csv"
    SHAP_INFO_PATH = XGBOOST_DIR / "XGBoost_shap_info.csv"
    EXPECTED_VALUE_PATH = XGBOOST_DIR / "XGBoost_expected_shap_values.csv"

    ######################
    # DATASET INFORMATION
    ######################
    ID_VARIABLE: str = "CC"
    TARGET_VARIABLE: str = "URB"
    TARGET_LABEL_MAP: dict = field(default_factory=lambda: {
            1: "Protegido",
            2: "Controlado",
            3: "Autoaislado",
            4: "Individualista",
            5: "Simbólico"
        })

    ######################
    # STYLE
    ######################    
    CLASS_STYLES: dict = field(default_factory=lambda: {
        1: {"label": "Protegido", "color": "red", "hex_code":"#ec5b5b"},
        2: {"label": "Controlado", "color": "orange", "hex_code":"#ef6c00"},
        3: {"label": "Autoaislado", "color": "purple", "hex_code": "#8e44ad"},
        4: {"label": "Individualista", "color": "blue", "hex_code": "#2980b9"},
        5: {"label": "Simbólico", "color": "green", "hex_code": "#2e7d32"}
    })    
    FEATURE_DESCRIPTION: dict = field(default_factory=lambda: {
        "CSS": "calles sin salida",
        "CFS": "calles en fondo de saco",
        "CPE": "calles peatonales",
        "LIN": "comercios internos",
        "LEX": "comercios externos",
        "DIS_1": "una ubicación geográfica aislada del núcleo urbano",
        "DIS_2": "una ubicación geográfica separada del núcleo urbano",
        "DIS_3": "una ubicación geográfica integrada en el núcleo urbano",
        "VER": "verjas",
        "MUR": "muros",
        "CAD": "cadenas",
        "BOL": "bolardos",
        "ARB": "arbustos",
        "CPP": "carteles de propiedad privada",
        "PVI": "entrada por vivienda",
        "PBL": "entrada por bloque",
        "COM": "entrada común a la urbanización",
        "COMS": "varias entradas comunes a la urbanización",
        "PPU": "uso y dominio público de la vía",
        "PRE": "dominio privado y uso público restringido de la vía",
        "PPR": "uso y dominio privado de la vía",
        "GSE": "guardia de seguridad",
        "CSE": "cámaras de seguridad",
        "BSE": "barrera de seguridad",
        "ASE": "alarma de seguridad"
    })

    ####################
    # SEED
    #####################
    SEED: int = 42


config = AppConfig()
