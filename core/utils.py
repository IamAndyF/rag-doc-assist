import hashlib

def get_preview(path):
    try:
        with open(path, "r", encoding='utf-8', errors='ignore') as f:
            return f.read(1000)
    except Exception as e:
        return "[unreadable preview]"

def hash_file(path):
    #Return an MDS hash of a file
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()