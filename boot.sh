#!/bin/sh
echo PORT $PORT

exec gunicorn -b :$PORT --access-logfile - --error-logfile - bts_ml:app --timeout 90