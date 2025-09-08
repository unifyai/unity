import json

def bytes_to_dict(msg: bytes):
    return json.loads(msg.decode())

def dict_to_bytes(msg: dict):
    return json.dumps(msg).encode()