from pathlib import Path

import pandas as pd


#build paths
BASE_DIR = Path(__file__).resolve().parent
RAW_FILE = BASE_DIR / "output" / "raw_books.csv"
CLEAN_FILE = BASE_DIR / "output" / "clean_books.csv"

#convert star rating
CONVERSION_RATE = 105.50

RATING_MAP = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5,
}

def clean_data() -> pd.DataFrame:

    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"Raw data file not found: {RAW_FILE}\n"
            "Run scrape_books.py first."
        )

    books_df = pd.read_csv(RAW_FILE)

    print("Raw dataset preview:")
    print(books_df.head())

    print("\nRaw dataset information:")
    books_df.info()

    required_columns = {
        "title",
        "price",
        "star_rating",
        "availability",
        "category",
    }

    missing_columns = required_columns.difference(books_df.columns)

    if missing_columns:
        raise ValueError(
            f"Required columns are missing: {sorted(missing_columns)}"
        )

    #remove unnecessary spaces
    books_df["title"] = books_df["title"].astype("string").str.strip()
    books_df["category"] = books_df["category"].astype("string").str.strip()

    #convert GBP price to a numeric
    books_df["price_gbp"] = pd.to_numeric(
        books_df["price"]
        .astype("string")
        .str.replace(r"[^0-9.]", "", regex=True),
        errors="coerce",
)

    #convert rating words into numbers
    books_df["rating"] = books_df["star_rating"].map(RATING_MAP)

    #convert availability text into boolean values
    books_df["in_stock"] = (
        books_df["availability"]
        .astype("string")
        .str.contains("In stock", case=False, na=False)
    )

    #drop rows where essential identifying fields are missing
    original_row_count = len(books_df)

    books_df = books_df.dropna(
        subset=["title", "category"]
    ).copy()

    books_df = books_df[
        (books_df["title"] != "")
        & (books_df["category"] != "")
    ].copy()

    dropped_rows = original_row_count - len(books_df)

    #median imputation for numeric fields
    if books_df["price_gbp"].notna().any():
        price_median = books_df["price_gbp"].median()
        books_df["price_gbp"] = books_df["price_gbp"].fillna(
            price_median
        )
    else:
        raise ValueError("No valid price values were found.")

    if books_df["rating"].notna().any():
        rating_median = books_df["rating"].median()
        books_df["rating"] = books_df["rating"].fillna(
            rating_median
        )
    else:
        raise ValueError("No valid rating values were found.")

    #round and convert rating into integer type
    books_df["rating"] = (
        books_df["rating"]
        .round()
        .clip(1, 5)
        .astype(int)
    )

    #convert GBP prices to INR using
    books_df["price_inr"] = (
        books_df["price_gbp"] * CONVERSION_RATE
    ).round(2)


    clean_books_df = books_df[
        [
            "title",
            "price_gbp",
            "price_inr",
            "rating",
            "in_stock",
            "category",
        ]
    ].copy()

    CLEAN_FILE.parent.mkdir(parents=True, exist_ok=True)
    clean_books_df.to_csv(CLEAN_FILE, index=False)

    print(f"\nRows removed because title/category was missing: {dropped_rows}")
    print(f"GBP-to-INR conversion rate used: {CONVERSION_RATE}")
    print(f"Clean dataset contains {len(clean_books_df)} rows.")

    print("\nClean dataset data types:")
    print(clean_books_df.dtypes)

    print(f"\nClean dataset saved to: {CLEAN_FILE}")

    return clean_books_df


if __name__ == "__main__":
    clean_data()