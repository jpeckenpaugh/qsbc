FROM python:3.14-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl zip unzip ca-certificates gnupg vim postgresql-client
RUN curl -fsSL https://deb.nodesource.com/setup_24.x | bash - \
    && apt-get install -y --no-install-recommends nodejs
RUN curl -fsSL https://opencode.ai/install | bash
RUN npm i -g bun

ENV PATH="/root/.opencode/bin:${PATH}"

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]