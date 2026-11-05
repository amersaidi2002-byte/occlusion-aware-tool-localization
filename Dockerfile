FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir .
ENTRYPOINT ["toolloc"]
CMD ["--epochs", "16", "--output", "/results/main"]
