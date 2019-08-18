import requests
import urllib.parse
import zipfile
import io
import os

WB_URL = 'http://api.worldbank.org/v2/country/all/indicator/'
CLASS_URL = 'http://databank.worldbank.org/data/download/site-content/CLASS.xls'
INDS = ['SH.HIV.ARTC.ZS', 'SP.POP.TOTL']
OUTPUT_PATH = '/home/ben/Documents/coding_projects/growth_curves/source_data'
def getWBData(url,output_path):
    """Download data via the World Bank API, unzip the file, and extract to output_path"""
    response = requests.get(url)
    zf = zipfile.ZipFile(io.BytesIO(response.content))
    zf.extractall(path=output_path)

if __name__ == '__main__':
    for indicator in INDS:
        print(f'Downloading data for {indicator}')
        url = urllib.parse.urljoin(WB_URL, indicator) + '?downloadformat=CSV&date=2000:2017'
        getWBData(url=url, output_path=OUTPUT_PATH)
        print(f'Saving data to {OUTPUT_PATH}')
    # Get World Bank country classification data
    classExcel = requests.get(CLASS_URL)
    with open(os.path.join(OUTPUT_PATH, 'WB_Classification.xls'), 'wb') as output:
        output.write(classExcel.content)