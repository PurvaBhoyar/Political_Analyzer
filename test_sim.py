from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

query = "we'll give a firm reply to aatankwaad"
target1 = "Deal with cross border terrorism with a firm hand."
target2 = "आतंकवाद का कड़ाई से मुकाबला करेंगे"  # Hindi translation

query_emb = model.encode(query, convert_to_tensor=True)
t1_emb = model.encode(target1, convert_to_tensor=True)
t2_emb = model.encode(target2, convert_to_tensor=True)

print("Similarity to English target:", util.cos_sim(query_emb, t1_emb).item())
print("Similarity to Hindi target:", util.cos_sim(query_emb, t2_emb).item())
