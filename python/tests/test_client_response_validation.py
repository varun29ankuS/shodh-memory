import importlib.util
import pathlib
import unittest


CLIENT_PATH = pathlib.Path(__file__).resolve().parents[1] / "shodh_memory" / "client.py"
SPEC = importlib.util.spec_from_file_location("shodh_memory_client", CLIENT_PATH)
CLIENT_MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(CLIENT_MODULE)
Memory = CLIENT_MODULE.Memory


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload
        self.ok = True
        self.status_code = 200
        self.headers = {}
        self.text = ""

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, response_payload):
        self._response_payload = response_payload

    def post(self, *args, **kwargs):
        return FakeResponse(self._response_payload)


class TestClientResponseValidation(unittest.TestCase):
    def _build_memory_like(self, payload):
        mem = Memory.__new__(Memory)
        mem._session = FakeSession(payload)
        mem.user_id = "test-user"
        mem.base_url = "http://localhost:3030"
        mem.timeout = 1
        return mem

    def test_search_returns_empty_list_when_memories_missing(self):
        mem = self._build_memory_like({})
        self.assertEqual(mem.search(query="anything"), [])

    def test_get_all_returns_empty_list_when_memories_missing(self):
        mem = self._build_memory_like({})
        self.assertEqual(mem.get_all(), [])

    def test_forget_count_methods_return_zero_when_missing_count(self):
        mem = self._build_memory_like({})
        self.assertEqual(mem.forget_by_age(30), 0)
        self.assertEqual(mem.forget_by_importance(0.5), 0)
        self.assertEqual(mem.forget_by_pattern("test"), 0)
        self.assertEqual(mem.forget_by_tags(["temp"]), 0)
        self.assertEqual(mem.forget_by_date("2024-01-01T00:00:00Z", "2024-01-31T23:59:59Z"), 0)


if __name__ == "__main__":
    unittest.main()
