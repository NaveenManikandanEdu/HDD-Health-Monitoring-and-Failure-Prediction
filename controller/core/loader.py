import os
import joblib


class ModelLoader:
    """
    Enterprise Model Loader

    Loads active models from:
    controller/storage/models/active/<model_name>/
    """

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ACTIVE_ROOT = os.path.join(BASE_DIR, "storage", "models", "active")

    @classmethod
    def _get_model_dir(cls, model_name: str):
        return os.path.join(cls.ACTIVE_ROOT, model_name)

    @classmethod
    def get_active_version(cls, model_name: str):
        """
        Returns latest model filename (e.g., model_v2.pkl)
        """
        model_dir = cls._get_model_dir(model_name)

        if not os.path.exists(model_dir):
            return None

        models = sorted(
            [f for f in os.listdir(model_dir) if f.endswith(".pkl")],
            key=lambda x: int(x.split("_v")[1].split(".")[0])
        )

        if not models:
            return None

        return models[-1]

    @classmethod
    def load(cls, model_name: str):
        """
        Loads latest active model
        """
        model_dir = cls._get_model_dir(model_name)
        version = cls.get_active_version(model_name)

        if not version:
            print(f"⚠️ No active model found for {model_name}")
            return None

        model_path = os.path.join(model_dir, version)

        print(f"📦 Loading Active Model: {model_name} -> {version}")

        model = joblib.load(model_path)

        # Safety: force CPU
        if hasattr(model, "params"):
            model.params["device"] = "cpu"

        return model