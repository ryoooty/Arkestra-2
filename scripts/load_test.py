import json
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Tuple


URL = "http://localhost:8080/chat"


def post_json(url: str, payload: Dict[str, Any]) -> Tuple[int, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            body = response.read().decode("utf-8")
            status = response.getcode()
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8")
        status = exc.code
    except urllib.error.URLError as exc:
        return 0, {"error": str(exc)}

    try:
        return status, json.loads(body)
    except json.JSONDecodeError:
        return status, body


def main() -> None:
    for i in range(10):
        status, payload = post_json(URL, {"user_id": "loadtest", "text": f"ping {i}"})
        text = payload.get("text") if isinstance(payload, dict) else None
        print(i, status, text)
        time.sleep(0.1)


if __name__ == "__main__":
    main()
