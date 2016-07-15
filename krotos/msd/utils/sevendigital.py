import py7D
import requests
import time

from krotos.secrets import SECRETS as keys
from krotos.debug import report

py7D.api_settings.oauthkey  = keys['KEY_7DIGITAL_KEY']
py7D.api_settings.secret    = keys['KEY_7DIGITAL_SECRET']
py7D.api_settings.country   = keys['KEY_7DIGITAL_COUNTRY']



def get_preview_track(track_id_7digital, write_file):
    url =       py7D.preview_url(track_id_7digital)

    try:
        response =  requests.get(url, stream=True, timeout=2.0)
    except requests.exceptions.ConnectionError as e:
        time.sleep(3)
        return False, e
    except Exception as e:
        time.sleep(3)
        return False, e

    if not response.ok: return False, response

    for block in response.iter_content(1024):
        write_file.write(block)

    return True, response
