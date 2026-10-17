FROM nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY backend/requirements.txt /app/requirements.txt

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt

COPY backend/app /app/app

EXPOSE 8002

CMD [ \
    "python3", \
    "-m", \
    "uvicorn", \
    "app.main:app", \
    "--host", \
    "0.0.0.0", \
    "--port", \
    "8002" \
]