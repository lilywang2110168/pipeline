def reviews_from_file(path, splitter=None):
    with open(path) as f:
        full_text = f.read()
    if splitter:
        return full_text.split(splitter)
    return full_text
