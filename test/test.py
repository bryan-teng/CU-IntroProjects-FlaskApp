import requests

resp = requests.post("https://cu-intro-project.herokuapp.com/predict", files={'file': open('/Reference/Academic Documents/Codes/IntroProject/test/mdog2.jpeg', 'rb')})

print(resp.text)