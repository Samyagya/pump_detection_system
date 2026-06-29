import requests

stock = "Vodafone Idea"

url = (
    "https://api.gdeltproject.org/api/v2/doc/doc"
    f"?query=%22{stock}%22"
    "&mode=artlist"
    "&format=json"
    "&maxrecords=10"
)

response = requests.get(
    url,
    headers={"User-Agent": "Mozilla/5.0"}
)

print("Status:", response.status_code)

if response.status_code != 200:
    print(response.text)
    exit()

data = response.json()

for article in data["articles"]:
    print(article["title"])