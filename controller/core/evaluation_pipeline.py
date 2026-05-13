import joblib
import pandas as pd
import re
import shutil
from controller.config import MODELS_DIR, CSV_DIR
from controller.core.logger import log
from controller.reports.csv_evaluation import evaluate_csv_dataframe, generate_email_text_report
from controller.reports.generate_pdf import create_pdf_report
from controller.emailer.sender import send_report

def extract_version(path):
    """Safely extract version number from filename."""
    match = re.search(r'_v(\d+)', path.stem)
    return int(match.group(1)) if match else 0

def load_active_model(model_type="risk_7d"):
    model_dir = MODELS_DIR / "active" / model_type
    model_files = sorted(model_dir.glob("model_v*.pkl"), key=extract_version)

    if not model_files:
        raise RuntimeError(f"No active model found for {model_type}")

    latest_model = model_files[-1]
    model = joblib.load(latest_model)
    log(f"Loaded active model: {latest_model.name}")
    return model, latest_model.name

def evaluate_csv(csv_path):
    try:
        log(f"Starting evaluation: {csv_path.name}")
        raw_df = pd.read_csv(csv_path)

        if raw_df.empty:
            log("CSV is empty. Skipping.", level="WARNING")
            return None

        # 1. Evaluate
        model, model_name = load_active_model("risk_7d")
        summary, agg = evaluate_csv_dataframe(raw_df, model)
        
        # 2. Generate detailed email text and PDF attachment
        email_body = generate_email_text_report(agg)
        pdf_bytes = create_pdf_report(agg, model_name)

        # 3. Send Email
        log("Compiling reports and dispatching email...")
        send_report(
            subject=f"HDD Risk Report - {csv_path.stem}",
            body=email_body,
            pdf_bytes=pdf_bytes,
            filename=f"{csv_path.stem}.pdf"
        )

        # 4. Final routing: Move the completed CSV to the main staging directory
        final_destination = CSV_DIR / csv_path.name
        shutil.move(str(csv_path), str(final_destination))
        log(f"Evaluation finished. Moved {csv_path.name} to CSV staging -> {CSV_DIR.name}")

        log("Evaluation pipeline completed successfully.")
        return summary

    except Exception as e:
        log(f"Evaluation pipeline FAILED: {str(e)}", level="ERROR")
        raise