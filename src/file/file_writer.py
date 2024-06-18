class FileWriter():
    def save(self, fileName: str, content: str):
        print(f'FileWriter - Saving population file to {fileName}')
        with open(fileName, 'w') as file:
            file.write(content)
            
        return fileName