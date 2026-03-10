FROM python:3.12-slim
WORKDIR /site
COPY . .
RUN pip install fastapi uvicorn jinja2 itsdangerous python-multipart
EXPOSE 3000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "3000"]
