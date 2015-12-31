import py7D

from krotos.secrets import SECRETS as keys

py7D.api_settings.oauthkey  = keys['KEY_7DIGITAL_KEY']
py7D.api_settings.secret    = keys['KEY_7DIGITAL_SECRET']
py7D.api_settings.country   = keys['KEY_7DIGITAL_COUNTRY']



def get_preview_track(track_id_7digital, file_obj):
    return py7D.preview_url(track_id_7digital)
