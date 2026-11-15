import ollama, math, json
from pathlib import Path


def embed_text(text: str) -> list[float]:
    response = ollama.embed(model="embeddinggemma", input=text)
    return response["embeddings"][0]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot_product = sum(x * y for x, y in zip(a, b))
    length_a = math.sqrt(sum(x**2 for x in a))
    length_b = math.sqrt(sum(y**2 for y in b))
    return dot_product / (length_a * length_b)


def build_index(chunks: list[str], batch_size: int = 8) -> list[dict]:
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")
    index = []
    total_batches = math.ceil(len(chunks) / batch_size)

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]

        response = ollama.embed(
            model="embeddinggemma",
            input=batch,
        )
        batch_number = i // batch_size + 1
        print(f"{batch_number}/{total_batches} completed")
        vectors = response["embeddings"]
        for chunk, vector in zip(batch, vectors):
            record = {
                "text": chunk,
                "embedding": vector,
            }
            index.append(record)
    return index


def save_index(index: list[dict], path: str | Path) -> None:
    file_path = Path(path)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(index, file, ensure_ascii=False)


def load_index(path: str | Path) -> list[dict]:
    file_path = Path(path)
    if file_path.is_dir():
        raise ValueError("Expected a file, got a directory")
    if not file_path.exists():
        raise FileNotFoundError(file_path)
    with open(path, "r", encoding="utf-8") as file:
        index = json.load(file)
    return index


def search_index(
    query: str, index: list[dict], top_k: int = 3
) -> list[tuple[str, float]]:
    vector_query = embed_text(query)
    results = []
    for record in index:
        text = record["text"]
        vector = record["embedding"]
        score = cosine_similarity(vector_query, vector)
        result = (text, score)
        results.append(result)
    results.sort(key=lambda item: item[1], reverse=True)
    return results[:top_k]


def hybrid_search(
    query: str, index: list[dict], keyword: str, top_k=5
) -> list[tuple[str, float]]:
    vector_query = embed_text(query)
    results = []
    for record in index:
        text = record["text"]
        vector = record["embedding"]
        score = cosine_similarity(vector_query, vector)
        if keyword in text.lower():
            score += 0.15
        result = (text, score)
        results.append(result)
    results.sort(key=lambda item: item[1], reverse=True)
    return results[:top_k]


system = "Answer the question using only the provided context. Refer to the source numbers. If the information is missing, say so"


def build_context(results: list[tuple[str, float]]) -> str:
    parts = []
    for i, (text, score) in enumerate(results, start=1):
        formatted = f"[SOURCE{i}] \n {text}"
        parts.append(formatted)
    return "\n\n".join(parts)


def generate_answer(query: str, context: str) -> str:
    system_msg = {
        "role": "system",
        "content": "Answer the question using only the provided context. Refer to the source numbers. If the information is missing, say so. "
        "Every technical claim must include a reference to the source number, e.g. [SOURCE 2].",
    }
    prompt = f"QUESTION:\n{query}\n\nCONTEXT:\n{context}"
    user_msg = {"role": "user", "content": prompt}
    messages = [system_msg, user_msg]
    response = ollama.chat(model="qwen2.5:3b", messages=messages)
    answer = response["message"]["content"]
    return answer
