FROM python:3.8-slim-buster

WORKDIR /adtest 

RUN pip3 install Flask==1.1.2

RUN pip3 install tensorflow==2.3.0

RUN pip3 install Pillow

RUN pip3 install keras

COPY . .

EXPOSE 5000

ENTRYPOINT [ "python" ]

CMD [ "main.py" ]