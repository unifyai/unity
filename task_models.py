from pydantic import BaseModel


class Task(BaseModel):
    name: str
    description: str
    process: callable


class Primitive(BaseModel):
    name: str
    description: str
    process: callable
