import requests

with open('test_audio/final.m4a', 'rb') as f:
    r = requests.post('http://localhost:8000/api/scan/audio-file', files={'file': ('final.m4a', f, 'audio/m4a')})

print('Status:', r.status_code)
if 'audio' in r.headers.get('content-type', ''):
    with open('final_redacted.wav', 'wb') as f:
        f.write(r.content)
    print('Done! Listen to final_redacted.wav')
else:
    print('Response:', r.json())