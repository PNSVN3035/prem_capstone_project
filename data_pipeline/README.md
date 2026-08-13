# Module 1 - Data Pipeline

## Overview

In this module, I built a simple data pipeline that collects book data from a public website, cleans the data, converts the prices, stores everything in a SQLite database, and then runs SQL and pandas queries on it.

For this project, I used the Books to Scrape website because it is made specifically for web scraping practice. The website contains books instead of grocery products, but the main idea is the same. In a real company like Zepto, the same type of pipeline can be used to collect product details such as price, availability, rating, and category.

The flow of this module is:

Scrape → Clean → Convert → Store → Query

## Data Collection

I used the `requests` library to access the web pages and `BeautifulSoup` to read the HTML content.

For each book, I collected:

* Title
* Price in GBP
* Star rating
* Availability
* Category

I selected multiple book categories and followed pagination so that the final dataset contained more than the minimum required number of books.

The raw scraped data is saved as:

`data_pipeline/output/raw_books.csv`

## Data Cleaning

The scraped data was not ready to use directly, so I cleaned and converted the columns into proper formats.

For the price column, I removed the currency characters and converted the value into a numeric `price_gbp` column.

The star rating was available as text such as `One`, `Two`, `Three`, `Four`, and `Five`. I mapped these values to integers from 1 to 5.

The availability column was converted into a Boolean `in_stock` column.

If a numeric value cannot be parsed correctly, I convert it to a missing value first so the program does not crash. I then use the median of the valid values to fill the missing numeric value.

For important text fields such as `title` and `category`, I drop the row if the value is missing because these fields are needed to identify and organize the book correctly.

The cleaned data is saved as:

`data_pipeline/output/clean_books.csv`

## Currency Conversion

The project requires a fixed conversion rate:

**1 GBP = 105.50 INR**

I used this exact fixed rate to calculate the INR price.

The formula is:

`price_inr = price_gbp × 105.50`

This is a project-defined conversion rate, so I did not use any external currency API.

## Database Design

I used SQLite to store the cleaned data.

I created two tables:

### categories

* `category_id`
* `category_name`

`category_id` is the primary key.

### books

* `book_id`
* `title`
* `price_gbp`
* `price_inr`
* `rating`
* `in_stock`
* `category_id`

`book_id` is the primary key.

`category_id` is a foreign key connected to the `categories` table.

I separated categories into their own table instead of storing the category name repeatedly for every book. This makes the database more organized and follows a normalized relational design.

## SQL Queries

After loading the data into SQLite, I created multiple SQL queries.

The queries demonstrate:

* SELECT
* WHERE
* ORDER BY
* LIMIT
* DISTINCT
* IN
* BETWEEN
* JOIN

Some examples include finding books with high ratings, finding the most expensive books, filtering books within a price range, and combining book information with category information.

The query results are saved in the `output` folder.

## Using pandas with SQL

I used `pd.read_sql()` to execute SQL queries and load the results directly into pandas DataFrames.

I also read the `books` and `categories` tables into pandas separately and joined them using `pd.merge()`.

The merge was done using the common `category_id` column.

After sorting both results in the same way, I compared the SQL JOIN result with the pandas merge result.

The comparison returned `True`, which shows that both methods produced the same data.

## How to Run Module - 1

The project uses one common `requirements.txt` file in the root folder for all three modules.

First install the required libraries:

```bash
pip install -r requirements.txt
```

Then run Module 1 in this order:

```bash
python3 data_pipeline/scrape_books.py
```

```bash
python3 data_pipeline/data_cleaning.py
```

```bash
python3 data_pipeline/database.py
```

```bash
python3 data_pipeline/queries.py
```

The first script collects the raw book data.

The second script cleans the data and converts the price to INR.

The third script creates the SQLite database and inserts the cleaned records.

The last script runs the SQL queries and compares the SQL JOIN result with the pandas merge result.

## Main Output Files

The main generated files are:

* `output/raw_books.csv`
* `output/clean_books.csv`
* `books.db`
* SQL query output CSV files
* SQL JOIN output
* pandas merge output

## Design Decisions

I separated the project into different Python files because it makes the pipeline easier to understand and debug.

The scraping code is kept separate from the cleaning code. The database creation is also separate from the query section.

I used SQLite because it is lightweight, easy to use with Python, and does not require a separate database server.

For numeric parsing problems, I used median imputation because it allows the pipeline to continue without losing the complete row.

For missing title or category values, I chose to remove the row because replacing these values with guessed information would not be reliable.

I also used one consolidated `requirements.txt` file in the root project folder so the dependencies for all three modules can be managed from one place.
