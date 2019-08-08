import requests
import urllib.parse
import zipfile
import io

WB_URL = 'http://api.worldbank.org/v2/country/all/indicator/'
INDS = ['SH.HIV.ARTC.ZS', 'SP.POP.TOTL']
# Note: output path currently blocked by permissions error, not sure how to fix
# Without it, will just save in current directory
OUTPUT_PATH = '/home/ben.bigelow/Documents/coding_projects/growth_curves/source_data'
def getWBData(url,output_path):
    response = requests.get(url)
    zf = zipfile.ZipFile(io.BytesIO(response.content))
    zf.extractall()

if __name__ == '__main__':
    for indicator in INDS:
        print(indicator)
        url = urllib.parse.urljoin(WB_URL, indicator) + '?downloadformat=CSV&start=2000'
        getWBData(url=url, output_path=OUTPUT_PATH)