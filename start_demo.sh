#!/usr/bin/env bash
set -e

uvicorn app.report_assistant_server:app \
  --host 0.0.0.0 \
  --port 6006
