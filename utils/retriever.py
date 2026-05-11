import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class CatalogRetriever:
    def __init__(self, catalog_path: str):

        with open(catalog_path, "r", encoding="utf-8") as f:
            self.catalog = json.load(f)

        self.documents = []

        for item in self.catalog:

            text = " ".join([
                item.get("name", ""),
                item.get("description", ""),
                " ".join(item.get("keys", [])),
                " ".join(item.get("job_levels", [])),
                " ".join(item.get("languages", []))
            ])

            self.documents.append(text)

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            lowercase=True,
            ngram_range=(1, 2)
        )

        self.doc_vectors = self.vectorizer.fit_transform(self.documents)

    def search(self, query: str, k: int = 10):

        query_vector = self.vectorizer.transform([query])

        similarities = cosine_similarity(
            query_vector,
            self.doc_vectors
        ).flatten()

        ranked_indices = similarities.argsort()[::-1]

        results = []

        for idx in ranked_indices[:k]:

            score = float(similarities[idx])

            if score <= 0:
                continue

            item = self.catalog[idx]

            item["score"] = score

            results.append(item)

        return results