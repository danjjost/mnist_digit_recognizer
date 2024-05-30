class FileWriter():
    def save(self, fileName: str, content: str):
        with open(fileName, 'w') as file:
            file.write(content)
            
        return fileName