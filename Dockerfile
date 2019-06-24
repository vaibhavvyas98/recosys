FROM fastgenomics/pandas:0.22-p36-v3

WORKDIR /recommendersysimage
COPY . /var/www/html

RUN apk update

RUN apk upgrade

RUN	pip3 --no-cache-dir install Flask

RUN apk add build-base

RUN apk add python3-dev

RUN pip3 install Cython

RUN pip3 install scikit-learn

EXPOSE 8080

ENTRYPOINT ["python3"]
CMD ["/var/www/html/mname.py"]
