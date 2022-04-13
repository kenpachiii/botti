FROM python:3.9.9-slim-bullseye as base

# Setup env
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONFAULTHANDLER 1
ENV PATH=/home/botti/.local/bin:$PATH
ENV SES_SMTP_ENDPOINT=email-smtp.us-east-1.amazonaws.com
ENV SES_SMTP_USERNAME=AKIAURKPTFEA5VZV4WWN
ENV SES_SMTP_PASSWORD=BL6GWUudkwUAaJaUAgqjPgX5P4h6reF8WTWXcChjo8oX

# Build args
ARG GIT_TOKEN

# Prepare environment
RUN mkdir /botti \
  && apt-get update \
  && apt-get -y install sudo libatlas3-base curl libhdf5-serial-dev  \
  && apt-get clean \
  && useradd -u 1000 -G sudo -U -m -s /bin/bash botti \
  && chown botti:botti /botti \
  # Allow sudoers
  && echo "botti ALL=(ALL) NOPASSWD: /bin/chown" >> /etc/sudoers

WORKDIR /botti

# Install dependencies
FROM base as python-deps
RUN  apt-get update \
  && apt-get -y install build-essential libssl-dev git libffi-dev libgfortran5 pkg-config cmake gcc \
  && apt-get clean \
  && pip install --upgrade pip

# Install dependencies
COPY --chown=botti:botti requirements.txt /botti/
USER botti
RUN  pip install --user --no-cache-dir numpy git+https://${GIT_TOKEN}@github.com/kroitor/ccxt.pro.git#subdirectory=python

# Copy dependencies to runtime-image
FROM base as runtime-image
COPY --from=python-deps /usr/local/lib /usr/local/lib
ENV LD_LIBRARY_PATH /usr/local/lib

COPY --from=python-deps --chown=botti:botti /home/botti/.local /home/botti/.local

RUN apt-get update \
  && apt-get -y install exim4 \
  && apt-get clean   

COPY exim4.conf /etc/exim4/exim.conf.local
ADD exim.key /etc/exim4/exim.key
ADD exim.crt /etc/exim4/exim.crt

RUN /etc/init.d/exim4 restart

# ADD input.txt .

# USER botti

# # Install and execute
# COPY --chown=botti:botti . /botti/

# RUN pip install -e . --user --no-cache-dir --no-build-isolation \
#   && mkdir /botti/db/ 

# ENTRYPOINT ["botti"]

# CMD [ "python -m botti" ]


# From: hiddenleafresearch@gmail.com
# Subject: Test message
# This is a test.

# .


