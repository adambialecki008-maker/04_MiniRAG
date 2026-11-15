from minirag.search import (
    build_index,
    save_index,
    load_index,
    search_index,
    hybrid_search,
    build_context,
    generate_answer,
)
from minirag.processing import process_document
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
index_path = ROOT / "documents" / "index_full.json"


def process_index():
    documents = process_document(
        ROOT / "documents" / "520-um001_-en-e.pdf",
        chunk_size=200,
    )
    print(len(documents))
    index = build_index(documents, batch_size=8)
    save_index(index, index_path)
    print("Done and saved")


def show_index():
    index = load_index(index_path)
    print(f"Indexed chunks: {len(index)}")
    print(f"Vector dimensions: {len(index[0]['embedding'])}")


def search_question(query: str):
    index = load_index(index_path)
    results = search_index(query, index, 5)
    for position, (text, score) in enumerate(results, start=1):
        print(f"\nResult {position} | Similarity: {score:.3f}")
        print(text[:800])


def search_hybrid(query, keyword):
    index = load_index(index_path)
    results = hybrid_search(query, index, keyword, 5)
    for position, (text, score) in enumerate(results, start=1):
        print(f"\nResult {position} | Similarity: {score:.3f}")
        print(text[:800])


def search_text(query: str):
    index = load_index(index_path)
    matches = []
    for record in index:
        text = record["text"]
        if query in text.lower():
            matches.append(text)
    print("Matches:", len(matches))
    for i, text in enumerate(matches[:3], start=1):
        print(f"\nMatch {i}:")
        print(text)


def create_answer(query, keyword):
    index = load_index(index_path)
    results = hybrid_search(query, index, keyword, 5)
    context = build_context(results)
    answer = generate_answer(query, context)
    print(answer)


def show_menu():
    print("=" * 30)
    print("Mini rag fast menu")
    print("1. Build index")
    print("2. Show index")
    print("3. Search query")
    print("4. Find records with text")
    print("5. Hybrid search")
    print("6. Generate answer")
    print("X. Close")
    print("=" * 30)


def menu():
    while True:
        show_menu()
        choice = input("Chose option: ")
        match choice.lower():
            case "1":
                print("Processing index")
                process_index()
            case "2":
                show_index()
            case "3":
                query = input("Ask question: ")
                search_question(query)
            case "4":
                text = input("Write keyword: ")
                search_text(text)
            case "5":
                query = input("Ask question: ")
                keyword = input("Write keyword: ")
                results = search_hybrid(query, keyword)
            case "6":
                query = input("Ask question: ")
                keyword = input("Write keyword: ")
                create_answer(query, keyword)
            case "x":
                break


if __name__ == "__main__":
    menu()
