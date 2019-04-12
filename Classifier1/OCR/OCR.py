import base64
import json
from requests import Request, Session
import os, csv, glob
import collections
from bs4 import BeautifulSoup
from PIL import Image

def recognize_captcha(str_image_path):
    bin_captcha = open(str_image_path, 'rb').read()
    str_encode_file = base64.b64encode(bin_captcha).decode("utf-8")
    str_url = "https://vision.googleapis.com/v1/images:annotate?key="

    # insert your own google cloud vision api key
    str_api_key = ""

    str_headers = {'Content-Type': 'application/json'}
    str_json_data = {
        'requests': [
            {
                'image': {
                    'content': str_encode_file
                },
                'features': [
                    {
                        'type': "TEXT_DETECTION",
                        'maxResults': 10
                    }
                ]
            }
        ]
    }

    print("begin request")
    obj_session = Session()
    obj_request = Request("POST",
                          str_url + str_api_key,
                          data=json.dumps(str_json_data),
                          headers=str_headers
                          )
    obj_prepped = obj_session.prepare_request(obj_request)
    obj_response = obj_session.send(obj_prepped,
                                    verify=True,
                                    timeout=60
                                    )
    print("end request")

    if obj_response.status_code == 200:
        with open('data.json', 'w') as outfile:
            json.dump(obj_response.text, outfile)
        return obj_response.text
    else:
        return "error"

if __name__ == '__main__':
    listFile = 'filelist.csv'
    csvFile = open(listFile, 'w', newline='')
    csvWriter = csv.writer(csvFile)
    header = ['File_name', 'fraction', 'radiation', 'irradiation', 'IR', 'Gy', 'survival', 'surviving', 'colony', 'colonies', 'colony-formation', 'gy', 'dose', 'nM', 'nmol', 'mol', 'nm']
    csvWriter.writerow(header)
    for file in glob.glob(os.path.join('data/*.jpeg')):
        data = json.loads(recognize_captcha(file))
        data = data["responses"]
        new_list = []
        for i in data:
            try:
                new_list.append(i["fullTextAnnotation"]["text"])
            except:
                pass
            true_data = ",".join(new_list)
            kubetsunashi = true_data.lower()
            print(kubetsunashi)

            row = []
            row.append(os.path.basename(file))
            row.append(kubetsunashi.count('fraction'))
            row.append(true_data.count('Gy'))
            row.append(kubetsunashi.count('survival'))
            row.append(kubetsunashi.count('surviving'))
            row.append(kubetsunashi.count('gy'))
            csvWriter.writerow(row)
            print(true_data.split("\n"))