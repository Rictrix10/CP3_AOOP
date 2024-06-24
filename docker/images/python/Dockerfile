# Use a imagem oficial do Python como base
FROM python:3.9-slim

WORKDIR /usr/src/app

# Instale dependências do sistema
RUN apt-get update && apt-get install -y \
    wget \
    unzip \
    gnupg \
    chromium-driver \
    && rm -rf /var/lib/apt/lists/*

# Defina variáveis de ambiente para o chromedriver
ENV PATH="/usr/lib/chromium:${PATH}"
ENV CHROME_BIN="/usr/bin/chromium"

# Copie o arquivo requirements.txt para o diretório de trabalho
COPY requirements.txt .

# Instale as dependências do Python
RUN pip install --no-cache-dir -r requirements.txt

# Baixe os dados do NLTK necessários
RUN python -m nltk.downloader stopwords
RUN python -m nltk.downloader rslp

# Copie o código fonte para o diretório de trabalho
COPY . .

CMD ["python", "sentiment_analysis_model.py"]

