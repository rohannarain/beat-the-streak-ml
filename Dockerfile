FROM python:3.8-slim

RUN adduser flaskapp

WORKDIR /home/flaskapp

COPY app-requirements.txt ./
RUN pip install --no-cache-dir -r app-requirements.txt

COPY credentials credentials
COPY src/cloud_storage.py ./

COPY bts_ml.py config.py boot.sh ./
RUN chmod +x boot.sh

ENV FLASK_APP bts_ml.py

COPY app app

RUN chown -R flaskapp:flaskapp ./
USER flaskapp

ENTRYPOINT ["./boot.sh"]
