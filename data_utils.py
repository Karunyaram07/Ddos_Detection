# data_utils.py
import pandas as pd
from werkzeug.utils import secure_filename
import os

ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls"}
REQUIRED_COLUMNS = ["Src_IP", "Dst_IP", "Protocol", "Packet_Size", "Duration"]
# Columns we expect numerical for model input (you can adjust later)
NUMERIC_EXPECTED = ["Packet_Size", "Duration"]

def allowed_file_extension(filename: str) -> bool:
    _, ext = os.path.splitext(filename.lower())
    return ext in ALLOWED_EXTENSIONS

def read_dataset(file_path: str):
    """
    Read CSV or Excel and return DataFrame.
    Raises Exception on read error.
    """
    _, ext = os.path.splitext(file_path.lower())
    if ext == ".csv":
        return pd.read_csv(file_path)
    elif ext in (".xlsx", ".xls"):
        return pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file extension: " + ext)

def validate_required_columns(df: pd.DataFrame):
    """Return (True, msg) or (False, msg)"""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return False, f"Missing required columns: {', '.join(missing)}"
    return True, "All required columns present."

def check_numeric_columns(df: pd.DataFrame):
    """
    Ensure numeric columns are convertible to numeric.
    Returns (True, cleaned_df, msg) or (False, None, msg)
    """
    df_copy = df.copy()
    non_numeric = []
    for col in NUMERIC_EXPECTED:
        if col not in df_copy.columns:
            non_numeric.append(col + " (missing)")
            continue
        # try convert:
        converted = pd.to_numeric(df_copy[col], errors='coerce')
        if converted.isna().all():
            non_numeric.append(col)
        else:
            # replace column with converted (so downstream uses numeric)
            df_copy[col] = converted
    if non_numeric:
        return False, None, f"Columns not numeric or convertible: {', '.join(non_numeric)}"
    # Optionally drop rows with NaN in numeric columns (or choose another policy)
    df_copy = df_copy.dropna(subset=NUMERIC_EXPECTED)
    return True, df_copy, "Numeric checks passed."

def validate_dataset(file_path: str, min_rows:int=10):
    """
    High-level validate function:
    - reads file
    - checks required columns
    - checks numeric columns
    - returns tuple (valid:bool, df_or_none, message:str)
    """
    try:
        df = read_dataset(file_path)
    except Exception as e:
        return False, None, f"Error reading file: {e}"

    ok, msg = validate_required_columns(df)
    if not ok:
        return False, None, msg

    ok, df_clean, msg2 = check_numeric_columns(df)
    if not ok:
        return False, None, msg2

    # check number of rows
    if df_clean.shape[0] < min_rows:
        return False, None, f"Dataset too small: {df_clean.shape[0]} rows (min {min_rows})."

    return True, df_clean, "Dataset valid."

# small helper to secure and construct filepath when saving uploaded file
def save_uploaded_file(flask_file, upload_folder: str) -> str:
    """
    Save uploaded werkzeug FileStorage to upload_folder and return saved path.
    """
    filename = secure_filename(flask_file.filename)
    os.makedirs(upload_folder, exist_ok=True)
    filepath = os.path.join(upload_folder, filename)
    flask_file.save(filepath)
    return filepath
