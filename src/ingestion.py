import json


with open("data/pubmedqa.jsonl") as f:
   records = [json.loads(line) for line in f]


documents = []

for record in records:
   for context in record["context"]["contexts"]:
       documents.append({
           "text": context,
           "pubid": record["pubid"],
       })


print(f"Created {len(documents)} documents")
print(documents[0])