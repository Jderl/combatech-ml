import argparse
from pathlib import Path

from combatech_ml.core.combined_pipeline import train_all


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Combatech ML models")
    parser.add_argument(
        "--csv",
        type=str,
        default="original_files_ml/game_data_cleaned.csv",
        help="Path to training CSV",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="models/artifacts",
        help="Directory for model artifacts",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"Training CSV not found: {csv_path}")

    train_all(csv_path, args.out)
    print(f"Training complete. Artifacts saved to: {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
