import requests, getpass

client_id = input("Client ID: ").strip()
client_secret = getpass.getpass("Client Secret (hidden): ").strip()
code = input("Grant code: ").strip()

r = requests.post("https://accounts.zoho.in/oauth/v2/token", data={
    "grant_type": "authorization_code",
    "client_id": client_id,
    "client_secret": client_secret,
    "code": code,
}, timeout=30)
d = r.json()
print("Got access_token:", "access_token" in d)
print("Got refresh_token:", "refresh_token" in d)
if "refresh_token" in d:
    open("refresh_token.txt", "w").write(d["refresh_token"])
    print("Saved refresh token to refresh_token.txt")
else:
    print("Response:", {k: v for k, v in d.items() if "token" not in k})