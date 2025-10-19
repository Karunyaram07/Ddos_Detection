import pandas as pd
from werkzeug.utils import secure_filename
import os

ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls"}
REQUIRED_COLUMNS = ["Packet_Size", "Duration", "Protocol"]  # core numeric columns
OPTIONAL_COLUMNS = ["Src_IP", "Dst_IP"]  # keep if present

def allowed_file_extension(filename: str) -> bool:
    _, ext = os.path.splitext(filename.lower())
    return ext in ALLOWED_EXTENSIONS

def read_dataset(file_path: str):
    _, ext = os.path.splitext(file_path.lower())
    if ext == ".csv":
        return pd.read_csv(file_path)
    elif ext in (".xlsx", ".xls"):
        return pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file extension: " + ext)

def extract_relevant_columns(df: pd.DataFrame):
    """
    Keep only required and optional columns present in the dataset.
    Fill numeric missing values with median.
    """
    present_cols = [c for c in REQUIRED_COLUMNS if c in df.columns]
    if not present_cols:
        raise ValueError("Dataset must have at least one required column: Packet_Size, Duration, Protocol")
    
    # Keep optional columns if present
    present_cols += [c for c in OPTIONAL_COLUMNS if c in df.columns]
    
    df = df[present_cols].copy()

    # Convert numeric required columns
    for col in REQUIRED_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(df[col].median())


    return df

def validate_dataset(file_path: str, min_rows:int=5):
    """
    High-level validate function:
    - reads file
    - extracts relevant columns
    - returns tuple (valid:bool, df_or_none, message:str)
    """
    try:
        df = read_dataset(file_path)
    except Exception as e:
        return False, None, f"Error reading file: {e}"

    try:
        df_clean = extract_relevant_columns(df)
    except Exception as e:
        return False, None, str(e)

    if df_clean.shape[0] < min_rows:
        return False, None, f"Dataset too small: {df_clean.shape[0]} rows (min {min_rows})."

    return True, df_clean, "Dataset valid."

def save_uploaded_file(flask_file, upload_folder: str) -> str:
    filename = secure_filename(flask_file.filename)
    os.makedirs(upload_folder, exist_ok=True)
    filepath = os.path.join(upload_folder, filename)
    flask_file.save(filepath)
    return filepath
