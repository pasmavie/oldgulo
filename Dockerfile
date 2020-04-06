FROM python3.7

ENV PIPENV_VENV_IN_PROJECT true

# RUN alternatives --set gcc /usr/bin/gcc48

COPY ./ ./
RUN pipenv sync && pipenv run python -m gulo.__main__
