FROM andres77872/ubuntu_back

WORKDIR /nn

COPY src /nn/src
COPY requirements.txt /nn

RUN pip install -r requirements.txt