class FileLoader():
    def load(self, path: str) -> str:
        print(f'Loading file from {path}')
        with open(path, "r") as file:
            return file.read()