# API de l'assistant multi-agent, hébergée sur Hugging Face Spaces (SDK Docker, gratuit).
# Le projet est récupéré depuis GitHub au moment de la construction : pour mettre l'API à jour
# après un nouvel envoi sur GitHub, utiliser « Settings → Factory rebuild » dans le Space.
FROM python:3.12.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face exécute le conteneur avec l'utilisateur 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH
WORKDIR /home/user/app

RUN git clone --depth 1 https://github.com/kissou-alain/Assistant-Portefeuille-financier.git .
RUN pip install --no-cache-dir --user -r requirements-api.txt

EXPOSE 7860
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
