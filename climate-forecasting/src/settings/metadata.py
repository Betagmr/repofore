from pathlib import Path

TEST_DATASET = "DailyDelhiClimateTest.csv"
TRAIN_DATASET = "DailyDelhiClimateTrain.csv"

DATA_PATH = Path() / "data"
TRAIN_PATH = DATA_PATH / TRAIN_DATASET
TEST_PATH = DATA_PATH / TEST_DATASET
