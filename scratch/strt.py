from newsapi import NewsApiClient

api = NewsApiClient(api_key= "ce53ecc2c4ea4059a0c8a70765d63c6b" )

articles = api.get_everything(
    q="vodafone",
    language="en",
    sort_by="publishedAt"
)

print(articles)