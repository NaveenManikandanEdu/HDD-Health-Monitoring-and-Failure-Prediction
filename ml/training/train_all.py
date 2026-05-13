from ml.training.risk_trainer import train_risk
from ml.training.health_trainer import train_health

if __name__ == "__main__":
    print("=== TRAINING MODEL-1 : RISK (7-DAY) ===")
    train_risk("data/processed")  # ALL files (bootstrap)

    print("\n=== TRAINING MODEL-2 : HEALTH ===")
    train_health("data/processed")  # ALL files

    print("\n✅ ALL MODELS TRAINED SUCCESSFULLY")
