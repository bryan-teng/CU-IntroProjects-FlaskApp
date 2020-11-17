import requests

resp = requests.post("http://localhost:5000/predict", files={'file': open('/Reference/Academic Documents/Codes/IntroProject/test/plane.png', 'rb')})

print(resp.text)