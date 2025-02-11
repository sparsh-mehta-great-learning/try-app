FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

ENV PYTHONUNBUFFERED=1

CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0"]
