FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /app

RUN pip install --no-cache-dir \
    runpod \
    transformers \
    sentencepiece \
    protobuf \
    nltk

# Pre-download NLTK data
RUN python -c "import nltk; nltk.download('punkt_tab', quiet=True)"

# Pre-download model weights at build time so cold starts are fast
RUN python -c "from transformers import T5Tokenizer, T5ForConditionalGeneration; T5Tokenizer.from_pretrained('google/t5-v1_1-xxl'); T5ForConditionalGeneration.from_pretrained('kalpeshk2011/dipper-paraphraser-xxl')"

COPY handler.py /app/handler.py

CMD ["python", "-u", "handler.py"]
