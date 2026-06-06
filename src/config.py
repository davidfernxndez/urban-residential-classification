from dataclasses import dataclass, field
from pathlib import Path

@dataclass(frozen=True)
class Config:

    ####################
    # PATHS
    #####################
    ROOT_DIR: Path = Path(__file__).resolve().parent.parent
    
    # Data folders 
    DATA_DIR: Path = ROOT_DIR / "data"
    DATA_RAW_DIR: Path = DATA_DIR / "raw"
    DATA_PROCESSED_DIR: Path = DATA_DIR / "processed"
    DATA_FOLDS_DIR: Path = DATA_DIR / "folds"
 
    DATASET_PATH = DATA_PROCESSED_DIR / "model_data.csv"
    
    # Output folder
    OUTPUT_DIR: Path = ROOT_DIR /"output"
      
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

    ##########################
    # NESTED CROSS-VALIDATION
    ##########################
    OUTER_SPLITS: int = 5
    INNER_SPLITS: int = 3
    OUTER_FOLD_FILENAME : str = "outer_folds.csv"
    INNER_FOLD_PREFIX: str = "inner_fold_"

    ####################
    # SEED
    #####################
    SEED: int = 42


cfg = Config()
