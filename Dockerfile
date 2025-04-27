# Na podstawie obrazu z Docker Hub z instalacja pythona jak nizej
FROM python:3.13

# Ustawmy katalog roboczy w kontenerze
WORKDIR /app

# Kopiujemy caly nasz folder do srodka obrazu
COPY . .

RUN mkdir "data"

## Instalujemy w obrazie wszystkie nasze zaleznosci z pyproject.toml
RUN pip install .[dev]

# Uruchamiamy testy
CMD ["python", "oml/server.py"]
