import pandas as pd

REQUIRED_COLUMNS = [
    "date",
    "serial_number",
    "failure",
    "smart_5_raw",
    "smart_187_raw",
    "smart_188_raw",
    "smart_197_raw",
    "smart_198_raw"
]

def validate_csv(csv_path):
    try:
        df = pd.read_csv(csv_path)

        if df.empty:
            return False, "File is empty"

        missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]

        if missing:
            return False, f"Missing columns: {missing}"

        return True, "Valid"

    except Exception as e:
        return False, f"Read error: {str(e)}"