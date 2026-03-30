import json

# Path to your training JSON
json_path = "../../../speech-rag/src/data/spoken_train-v1.1.json"

with open(json_path, 'r') as f:
    data = json.load(f)

print(f"Top-level keys in JSON: {list(data.keys())}")
# Ver a estrutura do json para entender como os dados estão organizados
print(f"Number of articles: {len(data['data'])}")
data['data'][0]['paragraphs'].pop(0)
print(f"{data['data'][0].keys()}")
print(data['data'][0]['title'])
print(f"Number of paragraphs in first article: {len(data['data'][0]['paragraphs'])}")
print(f"First paragraph keys: {data['data'][0]['paragraphs'][0].keys()}")
print(f"Context of first paragraph (start): {data['data'][0]['paragraphs'][0]['context'][:100]}...")
print(f"Number of sentences in context: {len([s for s in data['data'][0]['paragraphs'][0]['context'].split('.') if s.strip()])}")
print(f"Sentences: {[s.strip() for s in data['data'][0]['paragraphs'][0]['context'].split('.') if s.strip()]}")
print(f"Tamanho do contexto: {len(data['data'][0]['paragraphs'][0]['context'])} caracteres")
print(f"Number of QAs in first paragraph: {len(data['data'][0]['paragraphs'][0]['qas'])}")
print(f"First QA keys: {data['data'][0]['paragraphs'][0]['qas'][0].keys()}")
print(f"First QA question: {data['data'][0]['paragraphs'][0]['qas'][0]['question']}")
print(f"First QA ID: {data['data'][0]['paragraphs'][0]['qas'][0]['id']}")
print(f"First QA answers: {data['data'][0]['paragraphs'][0]['qas'][0]['answers']}")  # Should be empty for training set

print(f"{data['version']}")





# print("--- JSON Structure Check ---")
# article = data['data'][0]
# print(f"First Article Title: {article['title']}")

# paragraph = article['paragraphs'][0]
# print(f"First Paragraph Context (start): {paragraph['context'][:50]}...")
# print(f"First QA ID: {paragraph['qas'][0]['id']}")


# print("\n--- Checking for 'id' matching ---")
# # Does the JSON have the 0_0_0 style IDs?
# print(f"ID of 1st QA: {paragraph['qas'][0]['id']}")