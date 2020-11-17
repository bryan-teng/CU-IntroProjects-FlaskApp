### How to use:
Add your .pth file into app/, and make sure the PATH name is configured correctly in torch_utils.py

Start Flask server locally using (for Windows Powershell, for other OS please check Flask docs)
`$env:FLASK_APP = "app/main"`
`flask run`

Also, when running `test.py`, please edit your localhost port number accordingly.