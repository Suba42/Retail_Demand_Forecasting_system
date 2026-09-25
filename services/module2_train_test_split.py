import pandas as pd

# ==========================================
# MODULE 2.3 - TRAIN / VALIDATION / TEST SPLIT
# ==========================================

INPUT_FILE = "module2_daily_forecasting.csv"

TRAIN_FILE = "module2_train.csv"
VALIDATION_FILE = "module2_validation.csv"
TEST_FILE = "module2_test.csv"

print("Loading daily forecasting dataset...")

df = pd.read_csv(INPUT_FILE)

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values("date").reset_index(drop=True)

print(f"Rows    : {len(df)}")
print(f"Columns : {len(df.columns)}")

print("\n==========================================")
print("MODULE 2.3 - TRAIN / VALIDATION / TEST SPLIT")
print("==========================================")

# ------------------------------------------
# 1. Dataset information
# ------------------------------------------

print("\n--- DATASET RANGE ---")

print(f"Start Date : {df['date'].min()}")
print(f"End Date   : {df['date'].max()}")
print(f"Total Days : {len(df)}")


# ------------------------------------------
# 2. Calculate split positions
# ------------------------------------------

total_rows = len(df)

train_end = int(total_rows * 0.80)
validation_end = int(total_rows * 0.90)

print("\n--- SPLIT POSITIONS ---")

print(f"Total records       : {total_rows}")
print(f"Training records    : {train_end}")
print(
    f"Validation records  : "
    f"{validation_end - train_end}"
)
print(
    f"Test records        : "
    f"{total_rows - validation_end}"
)


# ------------------------------------------
# 3. Create splits
# ------------------------------------------

train = df.iloc[:train_end].copy()

validation = df.iloc[
    train_end:validation_end
].copy()

test = df.iloc[
    validation_end:
].copy()


# ------------------------------------------
# 4. Display ranges
# ------------------------------------------

print("\n--- TRAINING SET ---")

print(f"Rows       : {len(train)}")
print(f"Start Date : {train['date'].min()}")
print(f"End Date   : {train['date'].max()}")


print("\n--- VALIDATION SET ---")

print(f"Rows       : {len(validation)}")
print(f"Start Date : {validation['date'].min()}")
print(f"End Date   : {validation['date'].max()}")


print("\n--- TEST SET ---")

print(f"Rows       : {len(test)}")
print(f"Start Date : {test['date'].min()}")
print(f"End Date   : {test['date'].max()}")


# ------------------------------------------
# 5. Check chronological order
# ------------------------------------------

print("\n--- CHRONOLOGICAL ORDER CHECK ---")

train_valid = (
    train["date"].max()
    < validation["date"].min()
)

valid_test = (
    validation["date"].max()
    < test["date"].min()
)

print(
    f"Train < Validation : {train_valid}"
)

print(
    f"Validation < Test  : {valid_test}"
)


# ------------------------------------------
# 6. Check overlapping dates
# ------------------------------------------

print("\n--- DATE OVERLAP CHECK ---")

train_validation_overlap = set(
    train["date"]
).intersection(
    set(validation["date"])
)

validation_test_overlap = set(
    validation["date"]
).intersection(
    set(test["date"])
)

print(
    f"Train / Validation overlap : "
    f"{len(train_validation_overlap)}"
)

print(
    f"Validation / Test overlap   : "
    f"{len(validation_test_overlap)}"
)


# ------------------------------------------
# 7. Save datasets
# ------------------------------------------

print("\nSaving split datasets...")

train.to_csv(
    TRAIN_FILE,
    index=False
)

validation.to_csv(
    VALIDATION_FILE,
    index=False
)

test.to_csv(
    TEST_FILE,
    index=False
)


# ------------------------------------------
# 8. Completion
# ------------------------------------------

print("\n==========================================")
print("MODULE 2.3 COMPLETED")
print("==========================================")

print("Created files:")
print(f"✓ {TRAIN_FILE}")
print(f"✓ {VALIDATION_FILE}")
print(f"✓ {TEST_FILE}")