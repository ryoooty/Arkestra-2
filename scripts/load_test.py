import time

try:  # pragma: no cover - optional dependency for runtime use
    import requests
except ImportError:  # pragma: no cover - handled at runtime when script executed directly
    requests = None  # type: ignore[assignment]

URL = "http://localhost:8080/chat"


def main():
    if requests is None:
        raise RuntimeError("The 'requests' package is required to run the load test script.")
    for i in range(10):
        r = requests.post(URL, json={"user_id": "loadtest", "text": f"ping {i}"})
        print(i, r.status_code, r.json().get("text"))
        time.sleep(0.1)


if __name__ == "__main__":
    main()
