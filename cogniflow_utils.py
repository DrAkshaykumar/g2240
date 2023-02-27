import requests
import json
from time import sleep

def cogniflow_request(model_url, api_key, image_base64, image_format, attempt=3):

    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json',
        'x-api-key': api_key
    }

    data = {
        "format": image_format,
        "base64_image": image_base64
    }

    data_json = json.dumps(data)
    while attempt > 0:
        try:
            response = requests.post(model_url, headers=headers, data=data_json)
            result = response.json()
        except Exception as ex:
            attempt = attempt - 1
            if attempt > 0:
                print(f'Error trying to get cogniflow prediction endpoint. Retrying again in 3 seconds. '
                      f'Error: {str(ex)}')
                sleep(3)
            else:
                raise ex
        else:
            return result

def cogniflow_request_audio(model_url, api_key, audio_base64, audio_format, audio_text, attempt=3):

    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json',
        'x-api-key': api_key
    }

    data = {
        "format": audio_format,
        "base64_audio": audio_base64,
        "ground_truth": audio_text
    }

    data_json = json.dumps(data)
    while attempt > 0:
        try:
            response = requests.post(model_url, headers=headers, data=data_json)
            result = response.json()
        except Exception as ex:
            attempt = attempt - 1
            if attempt > 0:
                print(f'Error trying to get cogniflow prediction endpoint. Retrying again in 3 seconds. '
                      f'Error: {str(ex)}')
                sleep(3)
            else:
                raise ex
        else:
            return result

def cogniflow_request_object(model_url, api_key, image_base64, image_format, attempt=3):

    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json',
        'x-api-key': api_key
    }

    data = {
        "format": image_format,
        "base64_image": image_base64,
        "confidence_threshold": 0.2,
        "normalize_boxes": False
    }

    data_json = json.dumps(data)
    while attempt > 0:
        try:
            response = requests.post(model_url, headers=headers, data=data_json)
            result = response.json()
        except Exception as ex:
            attempt = attempt - 1
            if attempt > 0:
                print(f'Error trying to get cogniflow prediction endpoint. Retrying again in 3 seconds. '
                      f'Error: {str(ex)}')
                sleep(3)
            else:
                raise ex
        else:
            return result

