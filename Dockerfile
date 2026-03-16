# Base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install uv
RUN pip install uv

# Copy dependency files
COPY pyproject.toml .
COPY uv.lock .

# Install dependencies system-wide
RUN uv sync --frozen --no-dev

# Copy project files
COPY . .

# Remove .venv if it exists (use system Python)
ENV PATH="/app/.venv/bin:$PATH"

# Expose port
EXPOSE 8000

# Start FastAPI
CMD ["/app/.venv/bin/uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]