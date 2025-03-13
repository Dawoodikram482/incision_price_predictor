import csv
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# Function to clean and convert price
def convert_price_to_float(price_str):
    print(f"Raw price string: {price_str}")  # Debugging line
    cleaned_price = price_str.replace('1 X €', '').strip().replace(',', '.')
    try:
        return float(cleaned_price)
    except ValueError:
        print(f"Failed to convert price, returning unmodified: {price_str}")  # Debugging
        return price_str

# Setup WebDriver
options = webdriver.ChromeOptions()
options.add_argument("--start-maximized")
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# Base URL
base_url = "https://www.henryschein.nl/nl-nl/medisch/c/browsesupplies"
driver.get(base_url)

wait = WebDriverWait(driver, 10)

# Extract categories
category_elements = wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "ul.hs-categories li.item a")))
category_links = [cat.get_attribute("href") for cat in category_elements]

all_products = []

# Loop through each category
for category_link in category_links:
    try:
        driver.get(category_link)
        time.sleep(2)

        # Extract sub-category links if available
        sub_category_elements = driver.find_elements(By.CSS_SELECTOR, "ul.hs-categories li.item a")
        sub_category_links = [sub.get_attribute("href") for sub in sub_category_elements] or [category_link]

        for sub_category_link in sub_category_links:
            driver.get(sub_category_link)
            time.sleep(3)

            product_elements = driver.find_elements(By.CSS_SELECTOR, "li.product")
            
            for product in product_elements:
                try:
                    product_name = product.find_element(By.CSS_SELECTOR, ".product-name a").text.strip()
                    price_elements = product.find_elements(By.CSS_SELECTOR, ".product-price.single-amount")
                    product_price = price_elements[0].text.strip() if price_elements else "Price not available"
                    product_price_float = convert_price_to_float(product_price)
                    
                    all_products.append([product_name, product_price_float])
                    print(f"Extracted: {product_name}, {product_price_float}")
                except Exception as e:
                    print(f"Error extracting product details: {e}")

    except Exception as e:
        print(f"Error processing category {category_link}: {e}")

# Save to CSV
try:
    with open("materials_data.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["material_name", "Price"])
        writer.writerows(all_products)
    print("Scraping complete! Data saved.")
except Exception as e:
    print(f"Error writing CSV file: {e}")

# Close driver
driver.quit()
