from sentence_transformers import SentenceTransformer
model = SentenceTransformer('shibing624/text2vec-base-chinese')
model.save('hf_models/text2vec-base-chinese')


