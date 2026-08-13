from pathlib import Path
import sqlite3

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATABASE_FILE = BASE_DIR / "books.db"
OUTPUT_DIR = BASE_DIR / "output"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_queries():

    if not DATABASE_FILE.exists():
        raise FileNotFoundError(
            f"Database not found: {DATABASE_FILE}\n"
            "Run database.py first."
        )

    connection = sqlite3.connect(DATABASE_FILE)

    #query-1

    query_1 = """
    SELECT
        book_id,
        title,
        price_gbp,
        price_inr,
        rating
    FROM books
    WHERE rating >= 4;
    """

    query_1_df = pd.read_sql(query_1, connection)

    print("\nQUERY 1")
    print("Books with rating 4 or 5")
    print(query_1_df.head(10))

    query_1_df.to_csv(
        OUTPUT_DIR / "query_1.csv",
        index=False
    )

    ##query-2

    query_2 = """
    SELECT
        title,
        price_gbp,
        price_inr,
        rating
    FROM books
    ORDER BY price_gbp DESC
    LIMIT 10;
    """

    query_2_df = pd.read_sql(query_2, connection)

    print("\nQUERY 2")
    print("10 most expensive books")
    print(query_2_df)

    query_2_df.to_csv(
        OUTPUT_DIR / "query_2.csv",
        index=False
    )

    ##query-3

    query_3 = """
    SELECT DISTINCT
        rating
    FROM books
    ORDER BY rating;
    """

    query_3_df = pd.read_sql(query_3, connection)

    print("\nQUERY 3")
    print("Distinct ratings available")
    print(query_3_df)

    query_3_df.to_csv(
        OUTPUT_DIR / "query_3.csv",
        index=False
    )

    ##query-4

    query_4 = """
    SELECT
        title,
        price_gbp,
        price_inr,
        rating
    FROM books
    WHERE price_gbp BETWEEN 20 AND 40
    ORDER BY price_gbp ASC;
    """

    query_4_df = pd.read_sql(query_4, connection)

    print("\nQUERY 4")
    print("Books priced between £20 and £40")
    print(query_4_df.head(10))

    query_4_df.to_csv(
        OUTPUT_DIR / "query_4.csv",
        index=False
    )

    ##query-5

    query_5 = """
    SELECT
        title,
        rating,
        price_gbp
    FROM books
    WHERE rating IN (4, 5)
    ORDER BY rating DESC, price_gbp ASC;
    """

    query_5_df = pd.read_sql(query_5, connection)

    print("\nQUERY 5")
    print("Books with rating 4 or 5 using IN")
    print(query_5_df.head(10))

    query_5_df.to_csv(
        OUTPUT_DIR / "query_5.csv",
        index=False
    )

    ##query-6

    join_query = """
    SELECT
        b.book_id,
        b.title,
        b.price_gbp,
        b.price_inr,
        b.rating,
        b.in_stock,
        c.category_name
    FROM books AS b
    INNER JOIN categories AS c
        ON b.category_id = c.category_id
    ORDER BY c.category_name, b.rating DESC, b.title;
    """

    join_sql_df = pd.read_sql(
        join_query,
        connection
    )

    print("\nQUERY 6")
    print("Books joined with their category")
    print(join_sql_df.head(20))

    join_sql_df.to_csv(
        OUTPUT_DIR / "join_sql.csv",
        index=False
    )

    #read database tables into pandas

    books_df = pd.read_sql(
        """
        SELECT
            book_id,
            title,
            price_gbp,
            price_inr,
            rating,
            in_stock,
            category_id
        FROM books;
        """,
        connection
    )

    categories_df = pd.read_sql(
        """
        SELECT
            category_id,
            category_name
        FROM categories;
        """,
        connection
    )

    join_pandas_df = pd.merge(
        books_df,
        categories_df,
        on="category_id",
        how="inner"
    )

    join_pandas_df = join_pandas_df[
        [
            "book_id",
            "title",
            "price_gbp",
            "price_inr",
            "rating",
            "in_stock",
            "category_name",
        ]
    ]

    join_pandas_df = join_pandas_df.sort_values(
        by=[
            "category_name",
            "rating",
            "title",
        ],
        ascending=[
            True,
            False,
            True,
        ]
    ).reset_index(drop=True)

    join_sql_df = join_sql_df.reset_index(drop=True)

    print("\nPANDAS MERGE RESULT")
    print(join_pandas_df.head(20))

    #compare SQL JOIN and pandas merge

    results_match = join_sql_df.equals(
        join_pandas_df
    )

    print("\nSQL JOIN and pandas merge match:")
    print(results_match)

    join_pandas_df.to_csv(
        OUTPUT_DIR / "join_pandas.csv",
        index=False
    )

    connection.close()

    print("\nAll SQL query outputs saved successfully.")


if __name__ == "__main__":
    run_queries()