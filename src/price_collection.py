!pip install webdriver-manager
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support import expected_conditions as EC
import csv
import time

# Function to clean and convert the price to a float, returning original structure if failed
def convert_price_to_float(price_str):
    print(f"Raw price string: {price_str}")  # Debugging line to check the price
    # Remove the '1 X €' part and any other non-numeric characters (except commas)
    cleaned_price = price_str.replace('1 X €', '').strip()

    # Replace commas with a dot for decimal if needed
    cleaned_price = cleaned_price.replace(',', '.')

    # Try to convert to float, return the original unmodified string if conversion fails
    try:
        return float(cleaned_price)
    except ValueError:
        print(f"Failed to convert price, returning unmodified: {price_str}")  # Debugging line
        return price_str  # Return the original string if conversion fails

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
        time.sleep(2)  # Allow page to load
        
        # Extract sub-category links if available
        try:
            sub_category_elements = driver.find_elements(By.CSS_SELECTOR, "ul.hs-categories li.item a")
            sub_category_links = [sub.get_attribute("href") for sub in sub_category_elements]
        except:
            sub_category_links = []

        # If no sub-categories, add current category link to process
        if not sub_category_links:
            sub_category_links.append(category_link)

        # Loop through each sub-category (or category if no sub-category)
        for sub_category_link in sub_category_links:
            try:
                driver.get(sub_category_link)
                time.sleep(2)  # Allow page to load

                # Handle the first product separately
                try:
                    first_product = driver.find_element(By.CSS_SELECTOR, "li.product first")  # Adjust selector if needed
                    product_name = first_product.find_element(By.CSS_SELECTOR, ".product-name a").text.strip()
                    
                    try:
                        # Corrected CSS selector for price
                        product_price = first_product.find_element(By.CSS_SELECTOR, ".product-price.single-amount").text.strip()
                        product_price_float = convert_price_to_float(product_price)
                    except Exception as e:
                        product_price_float = "Price not available"  # If no price found
                        print(f"Error extracting price for first product: {e}")

                    all_products.append([product_name, product_price_float])
                    print(f"Extracted (First Product): {product_name}, {product_price_float}")

                except Exception as e:
                    print("No first-product found, skipping...")

                # Handle all other products
                other_products = driver.find_elements(By.CSS_SELECTOR, "li.product")

                for product in other_products:
                    try:
                        product_name = product.find_element(By.CSS_SELECTOR, ".product-name a").text.strip()
                        
                        try:
                            # Corrected CSS selector for price
                            product_price = product.find_element(By.CSS_SELECTOR, ".product-price.single-amount").text.strip()
                            product_price_float = convert_price_to_float(product_price)
                        except Exception as e:
                            product_price_float = "Price not available"  # If no price found
                            print(f"Error extracting price for other product: {e}")

                        all_products.append([product_name, product_price_float])
                        print(f"Extracted: {product_name}, {product_price_float}")

                    except Exception as e:
                        print(f"Error extracting product details: {e}")

            except Exception as e:
                print(f"Error processing sub-category: {e}")

    except Exception as e:
        print(f"Error processing category: {e}")

# Save to CSV
with open("../data/Materials in Protocols - IJsselland Ziekenhuis.csv", "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["material_name", "Price"])
    writer.writerows(all_products)

print("Scraping complete! Data saved.")
driver.quit()
