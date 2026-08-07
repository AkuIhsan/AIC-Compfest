FROM python:3.10.11

WORKDIR /app

COPY aic_qc_artifact/requirements.txt aic_qc_artifact/requirements.txt
RUN pip install --no-cache-dir -r aic_qc_artifact/requirements.txt

COPY sop_checker.py f_engineer.py sop_standard.csv ./
COPY aic_qc_artifact/ aic_qc_artifact/

ENV PYTHONPATH=/app

CMD ["python", "aic_qc_artifact/smoke_test.py"]