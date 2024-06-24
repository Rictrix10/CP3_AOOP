docker compose up --build sentiment_analysis

docker-compose run --rm sentiment_analysis /bin/bash

python extraction_info.py

python sentiment_analysis_model.py

