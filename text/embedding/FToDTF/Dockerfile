FROM python:3.5.5
COPY . /code
RUN mkdir /data && cd /code && pip3 install . && python3 -m nltk.downloader punkt
WORKDIR /data
ENTRYPOINT ["fasttext"]