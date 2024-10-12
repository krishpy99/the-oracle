from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class FewShotQueryClassifier:
    def __init__(self, model_name='paraphrase-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.examples = {}

    def add_category(self, category, example_queries):
        self.examples[category] = example_queries

    def classify_query(self, query):
        query_embedding = self.model.encode([query])[0]
        max_similarity = -1
        best_category = None
        
        for category, example_queries in self.examples.items():
            example_embeddings = self.model.encode(example_queries)
            similarity = cosine_similarity([query_embedding], example_embeddings)[0].max()
            if similarity > max_similarity:
                max_similarity = similarity
                best_category = category
        
        return best_category
