FROM python:3.11-slim
# Working directory
WORKDIR /app

# system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# copy requirements
COPY requirements.txt .

RUN pip install --upgrade pip

# install CPU-only pytorch
RUN pip install --no-cache-dir \
    torch \
    torchvision \
    --index-url https://download.pytorch.org/whl/cpu


# install remaining packages
RUN pip install --no-cache-dir \
    -r requirements.txt

# copy requirements files project

COPY app.py .
COPY configuration ./configuration
COPY outputs ./outputs

EXPOSE 8501

CMD ["streamlit","run","app.py","--server.address=0.0.0.0"]