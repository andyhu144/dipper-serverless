FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /app

RUN pip install --no-cache-dir \
    runpod \
    transformers \
    sentencepiece \
    protobuf \
    nltk

# Pre-download NLTK data only (small)
RUN python -c "import nltk; nltk.download('punkt_tab', quiet=True)"

COPY handler.py /app/handler.py

CMD ["python", "-u", "handler.py"]
