import smtplib
import yaml
import os
from email.message import EmailMessage
from pathlib import Path

from controller.core.logger import log

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "email_config.yaml"

def load_email_config():
    if not CONFIG_PATH.exists():
        raise RuntimeError("email_config.yaml not found.")
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

def send_report(subject, body, pdf_bytes, filename):
    config = load_email_config()

    if not config.get("alerts", {}).get("enabled", False):
        log("Email alerts disabled in config.")
        return

    smtp_cfg = config.get("smtp", {})
    alerts_cfg = config.get("alerts", {})

    server = smtp_cfg.get("server")
    port = smtp_cfg.get("port")
    sender = smtp_cfg.get("sender")
    receiver = smtp_cfg.get("receiver")
    
    # Securely load password
    password = smtp_cfg.get("password") or os.environ.get("EMAIL_PASSWORD")

    subject_prefix = alerts_cfg.get("subject_prefix", "")

    if not all([server, port, sender, password, receiver]):
        raise RuntimeError("Incomplete SMTP configuration. Ensure EMAIL_PASSWORD env var is set.")

    msg = EmailMessage()
    msg["From"] = sender
    msg["To"] = receiver
    msg["Subject"] = f"{subject_prefix} {subject}"
    msg.set_content(body)

    msg.add_attachment(
        pdf_bytes,
        maintype="application",
        subtype="pdf",
        filename=filename
    )

    try:
        with smtplib.SMTP(server, port) as smtp:
            smtp.starttls()
            smtp.login(sender, password)
            smtp.send_message(msg)
        log(f"Email sent successfully to {receiver}")
    except smtplib.SMTPException as e:
        log(f"Email sending FAILED: {str(e)}", level="ERROR")
        raise