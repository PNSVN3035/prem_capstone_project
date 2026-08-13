import requests
from bs4 import BeautifulSoup
import pandas as pd

from pathlib import Path
from urllib.parse import urljoin

BASE_URL = "https://books.toscrape.com/"

CATEGORY_URLS = {
    "Travel": "catalogue/category/books/travel_2/index.html",
    "Mystery": "catalogue/category/books/mystery_3/index.html",
    "Historical Fiction":
        "catalogue/category/books/historical-fiction_4/index.html",
    "Sequential Art":
        "catalogue/category/books/sequential-art_5/index.html",
}

OUTPUT_DIRECTORY = Path(__file__).parent / "output"
OUTPUT_FILE = OUTPUT_DIRECTORY / "raw_books.csv"

#create empty list
books_data = []

#scraping
def scrape_category(category_name: str, category_path: str) -> None:

    current_url = urljoin(BASE_URL, category_path)

    while current_url:
        try:
            response = requests.get(current_url, timeout=20)
            response.raise_for_status()

            #fix encoding before parsing HTML
            response.encoding = "utf-8"

        except requests.RequestException as error:
            print(f"Request failed for {current_url}: {error}")
            break

        soup = BeautifulSoup(response.text, "html.parser")
        books = soup.select("article.product_pod")

        for book in books:
            title_element = book.select_one("h3 a")
            price_element = book.select_one("p.price_color")
            rating_element = book.select_one("p.star-rating")
            availability_element = book.select_one(
                "p.instock.availability"
            )

            if not all(
                [
                    title_element,
                    price_element,
                    rating_element,
                    availability_element,
                ]
            ):
                print(
                    f"Skipping one incomplete record in {category_name}."
                )
                continue

            rating_classes = rating_element.get("class", [])
            star_rating = (
                rating_classes[1]
                if len(rating_classes) > 1
                else None
            )

            books_data.append(
                {
                    "title": title_element.get("title", "").strip(),
                    "price": price_element.get_text(strip=True),
                    "star_rating": star_rating,
                    "availability": availability_element.get_text(
                        " ",
                        strip=True,
                    ),
                    "category": category_name,
                }
            )

        next_element = soup.select_one("li.next a")

        if next_element:
            next_path = next_element.get("href")
            current_url = urljoin(current_url, next_path)
        else:
            current_url = None


def main() -> None:

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    for category_name, category_path in CATEGORY_URLS.items():
        print(f"Scraping category: {category_name}")
        scrape_category(category_name, category_path)

    #create dataframe
    books_df = pd.DataFrame(books_data)

    print("\nFirst five scraped rows:")
    print(books_df.head())

    print(f"\nTotal books scraped: {len(books_df)}")
    print(
        f"Number of categories: "
        f"{books_df['category'].nunique()}"
    )

    print("\nBooks by category:")
    print(books_df["category"].value_counts())

    if len(books_df) < 60:
        raise ValueError("Not enough books were scraped.")

    if books_df["category"].nunique() < 3:
        raise ValueError("Not enough book categories were collected.")

    books_df.to_csv(OUTPUT_FILE, index=False)

    print(f"\nRaw data saved successfully to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()