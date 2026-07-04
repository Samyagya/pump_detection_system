from googlesearch import search

results = list(search("Eurotex Industries", num_results=20))

print("RESULT COUNT:", len(results))
print(results)