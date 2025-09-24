from app.rag.index import add_texts

def main():
    rows = [
        {"id":"seed1","text":"Авто-сон переносит short в temp, temp в long.","meta":{"layer":"doc"}},
        {"id":"seed2","text":"Реранкер e5-small повышает релевантность RAG.","meta":{"layer":"doc"}},
        {"id":"seed3","text":"Junior отдаёт style_directive и neuro_update уровни.","meta":{"layer":"doc"}},
    ]
    add_texts(rows); print("RAG seeded.")
if __name__ == "__main__":
    main()
