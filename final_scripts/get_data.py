import requests
import urllib.parse
import zipfile
import io

WB_URL = 'http://api.worldbank.org/v2/country/all/indicator/'
INDS = ['SH.HIV.ARTC.ZS', 'SP.POP.TOTL']
OUTPUT_PATH = '/home/ben/Documents/coding_projects/growth_curves/source_data'
def getWBData(url,output_path):
    response = requests.get(url)
    zf = zipfile.ZipFile(io.BytesIO(response.content))
    zf.extractall(path=output_path)

if __name__ == '__main__':
    for indicator in INDS:
        print(f'Downloading data for {indicator}')
        url = urllib.parse.urljoin(WB_URL, indicator) + '?downloadformat=CSV&date=2000:2018'
        getWBData(url=url, output_path=OUTPUT_PATH)
        print(f'Saving data to {OUTPUT_PATH}')