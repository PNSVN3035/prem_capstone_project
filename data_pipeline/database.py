from pathlib import Path
import sqlite3

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
CLEAN_FILE = BASE_DIR / "output" / "clean_books.csv"
DATABASE_FILE = BASE_DIR / "books.db"


def create_database() -> None:

    if not CLEAN_FILE.exists():
        raise FileNotFoundError(
            f"Clean data file not found: {CLEAN_FILE}\n"
            "Run data_cleaning.py first."
        )

    books_df = pd.read_csv(CLEAN_FILE)

    required_columns = {
        "title",
        "price_gbp",
        "price_inr",
        "rating",
        "in_stock",
        "category",
    }

    missing_columns = required_columns.difference(books_df.columns)

    if missing_columns:
        raise ValueError(
            f"Required columns are missing: {sorted(missing_columns)}"
        )

    #convert CSV values
    books_df["price_gbp"] = pd.to_numeric(
        books_df["price_gbp"],
        errors="raise",
    )

    books_df["price_inr"] = pd.to_numeric(
        books_df["price_inr"],
        errors="raise",
    )

    books_df["rating"] = pd.to_numeric(
        books_df["rating"],
        errors="raise",
    ).astype(int)

    #CSV may read boolean values as text
    if books_df["in_stock"].dtype == object:
        books_df["in_stock"] = (
            books_df["in_stock"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({"true": True, "false": False})
        )

    if books_df["in_stock"].isna().any():
        raise ValueError(
            "The in_stock column contains invalid Boolean values."
        )

    with sqlite3.connect(DATABASE_FILE) as connection:
        connection.execute("PRAGMA foreign_keys = ON;")
        cursor = connection.cursor()


        cursor.execute("DROP TABLE IF EXISTS books;")
        cursor.execute("DROP TABLE IF EXISTS categories;")

        cursor.execute(
            """
            CREATE TABLE categories (
                category_id INTEGER PRIMARY KEY AUTOINCREMENT,
                category_name TEXT NOT NULL UNIQUE
            );
            """
        )

        cursor.execute(
            """
            CREATE TABLE books (
                book_id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                price_gbp REAL NOT NULL CHECK (price_gbp >= 0),
                price_inr REAL NOT NULL CHECK (price_inr >= 0),
                rating INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
                in_stock INTEGER NOT NULL CHECK (in_stock IN (0, 1)),
                category_id INTEGER NOT NULL,
                FOREIGN KEY (category_id)
                    REFERENCES categories(category_id)
            );
            """
        )

        #insert each unique category
        category_records = [
            (category,)
            for category in sorted(
                books_df["category"].dropna().unique()
            )
        ]

        cursor.executemany(
            """
            INSERT INTO categories (category_name)
            VALUES (?);
            """,
            category_records,
        )

        #read category IDs back from the database
        cursor.execute(
            """
            SELECT category_id, category_name
            FROM categories;
            """
        )

        category_lookup = {
            category_name: category_id
            for category_id, category_name in cursor.fetchall()
        }

        #prepare all book records
        book_records = []

        for row in books_df.itertuples(index=False):
            category_id = category_lookup[row.category]

            book_records.append(
                (
                    row.title,
                    float(row.price_gbp),
                    float(row.price_inr),
                    int(row.rating),
                    int(bool(row.in_stock)),
                    category_id,
                )
            )

        cursor.executemany(
            """
            INSERT INTO books (
                title,
                price_gbp,
                price_inr,
                rating,
                in_stock,
                category_id
            )
            VALUES (?, ?, ?, ?, ?, ?);
            """,
            book_records,
        )

        connection.commit()

        category_count = cursor.execute(
            "SELECT COUNT(*) FROM categories;"
        ).fetchone()[0]

        book_count = cursor.execute(
            "SELECT COUNT(*) FROM books;"
        ).fetchone()[0]

    print(f"Database created successfully: {DATABASE_FILE}")
    print(f"Categories inserted: {category_count}")
    print(f"Books inserted: {book_count}")


if __name__ == "__main__":
    create_database()