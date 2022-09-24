FROM ghcr.io/luxonis/robothub-base-app:ubuntu-depthai-main

RUN pip3 install -U numpy opencv-contrib-python-headless

ARG FILE=app.py

ADD script.py .

ADD *.blob .
ADD *.json .

ADD $FILE run.py
