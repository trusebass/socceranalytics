'''
unzips all the files in a folder and makes sure to name the unzipped folders just as the zipped ones
were.
'''

import zipfile as zip
import os
from pathlib import Path

FILES_PATH = "/Users/sebastiantruijens/Projects/Python/ETH/SoccerAnalytics/data/sweden_data/skillcorner/"

files = os.listdir(FILES_PATH)

#debugging
print(files)
print(type(files))
print(type(files[0]))

#extract files
for file in files:
    print(f"extracting {file}")
    print(f"foldername: {FILES_PATH + file.split('.')[0]}")
    if zip.is_zipfile(FILES_PATH + file):
        with zip.ZipFile(FILES_PATH + file) as zipref:
            zipref.extractall(FILES_PATH + file.split('.')[0])
            os.remove(FILES_PATH + file)
            
# Alternative approach using pathlib for better readability and efficiency

'''
FILES_PATH = Path(FILES_PATH)

for file in FILES_PATH.iterdir():
    if file.suffix == '.zip' and file.is_file():
        print(f"Extracting {file.name}")
        with zip.ZipFile(file) as zipref:
            extract_path = FILES_PATH / file.stem
            zipref.extractall(extract_path)
        file.unlink()  # Remove the zip file after extraction
        
'''