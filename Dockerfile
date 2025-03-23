# Na podstawie obrazu z Docker Hub z instalacja pythona jak nizej
FROM python:3.13

# Ustawmy katalog roboczy w kontenerze
WORKDIR /app

# Kopiujemy caly nasz folder do srodka obrazu
COPY . .

# odpalmy wirtualne srodowisko
RUN python -m venv venv
ENV PATH="/app/venv/bin:$PATH"

## Instalujemy w obrazie wszystkie nasze zaleznosci z pyproject.toml
RUN pip install .[dev]

# Uruchamiamy testy
CMD ["pytest", "tests/unit", "--cov=oml", "--cov-report=term-missing"]
