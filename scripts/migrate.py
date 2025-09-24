from app.memory.db import migrate


if __name__ == "__main__":
    migrate()
    print("DB migrated.")
