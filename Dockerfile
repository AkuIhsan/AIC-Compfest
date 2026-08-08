FROM python:3.10.11

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY sop_checker.py f_engineer.py sop_standard.csv main.py schemas.py ./
COPY aic_qc_artifact/ aic_qc_artifact/

ENV PYTHONPATH=/app
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]