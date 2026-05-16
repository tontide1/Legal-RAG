FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
  build-essential \
  libpq-dev \
  gcc \
  poppler-utils \
  libgl1 \
  libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*

# Use uv for faster dependency installation
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install paddlepaddle from PaddlePaddle's official CPU index (not on PyPI)
RUN pip install paddlepaddle==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/

COPY requirements.txt ./
# Install remaining dependencies using uv with BuildKit cache mount
RUN --mount=type=cache,target=/root/.cache/uv \
  uv pip install --system -r requirements.txt

# Copy the backend folder contents into /app/backend
# This ensures that 'from backend.xxx' imports work
COPY . ./backend

EXPOSE 8000

# Run from /app so that the 'backend' folder is a package
CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
