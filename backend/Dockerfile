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

# Install CUDA-enabled PyTorch from the official PyTorch wheel index first.
# We keep this separate from the generic requirements install to avoid uv
# pulling the newest CUDA 13 stack, which is much heavier and has been flaky
# to extract in Docker Desktop on Windows.
RUN python -m pip install --no-cache-dir --upgrade pip && \
  python -m pip install --no-cache-dir \
  torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121

# Install the remaining dependencies using uv with BuildKit cache mount,
# excluding torch packages that were installed explicitly above.
RUN --mount=type=cache,target=/root/.cache/uv \
  grep -Ev "^(torch|torchvision|torchaudio)([<>=].*)?$" requirements.txt > requirements.runtime.txt && \
  uv pip install --system -r requirements.runtime.txt

# Copy the backend folder contents into /app/backend
# This ensures that 'from backend.xxx' imports work
COPY . ./backend

EXPOSE 8000

# Run from /app so that the 'backend' folder is a package
CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
