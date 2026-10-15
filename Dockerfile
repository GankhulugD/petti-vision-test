# Web + model in one container, same pattern as pinequest toothlings
# (Docker, port 7860). Vercel cannot run these weights.

FROM node:22-bookworm-slim AS web
WORKDIR /src
COPY apps/web/package.json apps/web/package-lock.json ./
RUN npm ci
COPY apps/web/ ./
ENV PETTI_API_URL=http://127.0.0.1:8765
ENV NEXT_TELEMETRY_DISABLED=1
RUN npm run build

FROM python:3.12-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 1000 user

COPY --from=web /usr/local/bin/node /usr/local/bin/node

WORKDIR /app
COPY apps/model/requirements.txt /app/model/requirements.txt
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && grep -v -E '^(torch|torchvision)' /app/model/requirements.txt > /tmp/req.txt \
    && pip install --no-cache-dir -r /tmp/req.txt

COPY apps/model /app/model
ENV HF_HOME=/app/model/.cache \
    PETTI_DEVICE=cpu \
    PETTI_HALF=0 \
    PETTI_PORT=8765
RUN cd /app/model && python download_models.py

COPY --from=web /src/.next/standalone /app/web
COPY --from=web /src/.next/static /app/web/.next/static
COPY --from=web /src/public /app/web/public
COPY scripts/serve.sh /app/serve.sh

RUN chmod +x /app/serve.sh \
    && mkdir -p /app/model/data \
    && chown -R user:user /app
USER user

ENV PORT=7860
EXPOSE 7860
CMD ["/app/serve.sh"]
