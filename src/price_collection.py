from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support import expected_conditions as EC

options = webdriver.ChromeOptions()
options.add_argument("--start-maximized")
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

url = "https://www.henryschein.nl/nl-nl/medisch/c/browsesupplies"
wait = WebDriverWait(driver, 10)
driver.get(url)

# category_items = driver.find_elements(By.CSS_SELECTOR, "ul.hs-categories li.item")
category_items = wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "ul.hs-categories li.item")))

for item in category_items:
    try:
        link = item.find_element(By.TAG_NAME, "a").get_attribute("href")
        print(f"Category: Link: {link}")

        # driver.get(link)
        # sub_category_items = wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "ul.hs-categories li.item")))
        
        # for sub_item in sub_category_items:
        #     try:
        #         link = sub_item.find_element(By.TAG_NAME, "a").get_attribute("href")
        #         print(f"Sub Category: Link: {link}")
        #     except Exception as e:
        #         print("error in sub category: ", e)

    except Exception as e:
        print("error in category: ", e)

driver.quit()