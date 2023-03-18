FROM openfabric/openfabric-pyenv:0.1.9-3.8

RUN mkdir temp
WORKDIR /temp
COPY ./ignite.py /temp/
COPY ./intents.json /temp/
COPY ./main.py /temp/
COPY ./pyproject.toml /temp/
COPY ./start.sh /temp/
RUN poetry install -vvv --no-dev
CMD ["sh", "start.sh"]
