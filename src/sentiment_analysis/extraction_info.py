from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import csv
import time

# Configuração do WebDriver
options = webdriver.ChromeOptions()
options.add_argument('--headless')  # Execute o Chrome em modo headless
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')

driver = webdriver.Chrome(options=options)

# Navegar até o vídeo do YouTube
video_url = "https://www.youtube.com/watch?v=rPVlKOc0-rs"  # substitua pela URL do vídeo
driver.get(video_url)

# Rolar a página para carregar os comentários
time.sleep(5)  # esperar a página carregar completamente

last_height = driver.execute_script("return document.documentElement.scrollHeight")
while True:
    driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
    time.sleep(3)
    new_height = driver.execute_script("return document.documentElement.scrollHeight")
    if new_height == last_height:
        break
    last_height = new_height

# Extrair comentários
comments = driver.find_elements(By.XPATH, '//*[@id="content-text"]')

# Salvar comentários em um arquivo CSV
with open("/csv/youtube_comments.csv", "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["Comentário"])
    for comment in comments:
        writer.writerow([comment.text])

# Fechar o navegador
driver.quit()
