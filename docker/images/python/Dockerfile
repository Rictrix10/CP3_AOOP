
FROM python:3.9-slim

WORKDIR /usr/src/app


RUN apt-get update && apt-get install -y \
    wget \
    unzip \
    gnupg \
    chromium-driver \
    && rm -rf /var/lib/apt/lists/*


ENV PATH="/usr/lib/chromium:${PATH}"
ENV CHROME_BIN="/usr/bin/chromium"


COPY requirements.txt .


RUN pip install --no-cache-dir -r requirements.txt


RUN python -m nltk.downloader stopwords
RUN python -m nltk.downloader rslp


COPY . .

CMD ["python", "sentiment_analysis_model.py"]

