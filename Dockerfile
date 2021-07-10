FROM python:3.8-buster

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
  && apt-get install -y tesseract-ocr ffmpeg \
  && rm -rf /var/lib/apt/lists/*

RUN pip install -U setuptools pip wheel
RUN pip install -U ffmpeg-python click guessit opencv-python librosa \
                   pysubs2 scikit-image jinja2 lxml tqdm pyxdameraulevenshtein \
                   textblob jinja2 pytesseract lxml annoy

RUN mkdir /code
ADD tess-data /code/tess-data
COPY cartonizer.py cowocr.py milksync.py /code/

RUN echo '#!/bin/bash\npython3 /code/cartonizer.py "$@"' > /usr/bin/cartonizer && \
    echo '#!/bin/bash\npython3 /code/cowocr.py "$@"' > /usr/bin/cowocr && \
    echo '#!/bin/bash\npython3 /code/milksync.py "$@"' > /usr/bin/milksync && \
    chmod +x /usr/bin/cartonizer /usr/bin/cowocr /usr/bin/milksync

RUN mkdir /workdir
WORKDIR /workdir