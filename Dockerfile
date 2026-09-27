FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install --no-cache-dir -r requirements.txt psycopg[binary]
ARG RAG_MODEL=intfloat/multilingual-e5-small
ENV RAG_MODEL=${RAG_MODEL} HF_HOME=/opt/huggingface
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${RAG_MODEL}', trust_remote_code=False)"
COPY . .
RUN python generate_data.py
CMD ["uvicorn","app.main:app","--host","0.0.0.0","--port","8765"]
