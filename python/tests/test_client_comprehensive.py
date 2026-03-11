"""Comprehensive tests for shodh_memory.client

Covers all public methods, exception handling, auto-start logic,
platform detection, and edge cases. Uses unittest.mock exclusively
— no real network calls."""

import importlib.util
import os
import pathlib
import platform
import sys
import unittest
from unittest import mock

# Load the client module directly from file path to avoid import issues
CLIENT_PATH = pathlib.Path(__file__).resolve().parents[1] / "shodh_memory" / "client.py"
SPEC = importlib.util.spec_from_file_location("shodh_memory_client", CLIENT_PATH)
assert SPEC is not None and SPEC.loader is not None
CLIENT_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CLIENT_MODULE)

Memory = CLIENT_MODULE.Memory
Experience = CLIENT_MODULE.Experience
MemoryStats = CLIENT_MODULE.MemoryStats
ShodhError = CLIENT_MODULE.ShodhError
ShodhConnectionError = CLIENT_MODULE.ShodhConnectionError
ShodhAuthenticationError = CLIENT_MODULE.ShodhAuthenticationError
ShodhValidationError = CLIENT_MODULE.ShodhValidationError
ShodhNotFoundError = CLIENT_MODULE.ShodhNotFoundError
ShodhRateLimitError = CLIENT_MODULE.ShodhRateLimitError
ShodhServerError = CLIENT_MODULE.ShodhServerError
_handle_response_error = CLIENT_MODULE._handle_response_error
_default_storage_path = CLIENT_MODULE._default_storage_path


# =============================================================================
# Helpers
# =============================================================================

class FakeResponse:
    """Drop-in replacement for requests.Response."""

    def __init__(self, status_code=200, json_data=None, text="", headers=None, ok=None):
        self.status_code = status_code
        self._json_data = json_data or {}
        self.text = text
        self.headers = headers or {}
        self.ok = ok if ok is not None else (200 <= status_code < 400)

    def json(self):
        return self._json_data


def _build_memory(payload_overrides=None, session_methods=None):
    """Build a Memory instance with mocked session and no auto-start."""
    mem = Memory.__new__(Memory)
    mem.user_id = "test-user"
    mem.base_url = "http://localhost:3030"
    mem.timeout = 1
    mem.api_key = "test-key"
    mem.port = 3030
    mem.storage_path = "/tmp/test"
    mem._server_process = None

    fake_session = mock.MagicMock()
    if session_methods:
        for method_name, return_value in session_methods.items():
            getattr(fake_session, method_name).return_value = return_value
    mem._session = fake_session
    return mem


# =============================================================================
# _default_storage_path
# =============================================================================

class TestDefaultStoragePath(unittest.TestCase):
    @mock.patch("pathlib.Path.is_dir", return_value=True)
    @mock.patch("pathlib.Path.exists", return_value=True)
    def test_legacy_dir_used_if_exists(self, mock_exists, mock_is_dir):
        # Reimport to test the function (it uses the module-level constant)
        result = _default_storage_path()
        self.assertEqual(result, "shodh_memory_data")

    @mock.patch("pathlib.Path.exists", return_value=False)
    @mock.patch("platform.system", return_value="Darwin")
    def test_macos_xdg_path(self, mock_system, mock_exists):
        result = _default_storage_path()
        self.assertIn("shodh-memory", result)
        self.assertIn("Library", result)

    @mock.patch("pathlib.Path.exists", return_value=False)
    @mock.patch("platform.system", return_value="Linux")
    def test_linux_xdg_path(self, mock_system, mock_exists):
        with mock.patch.dict(os.environ, {"XDG_DATA_HOME": "/custom"}, clear=False):
            result = _default_storage_path()
            self.assertEqual(result, "/custom/shodh-memory")

    @mock.patch("pathlib.Path.exists", return_value=False)
    @mock.patch("platform.system", return_value="Windows")
    def test_windows_appdata(self, mock_system, mock_exists):
        with mock.patch.dict(os.environ, {"APPDATA": "C:\\Users\\test\\AppData"}, clear=False):
            result = _default_storage_path()
            self.assertIn("shodh-memory", result)


# =============================================================================
# _handle_response_error
# =============================================================================

class TestHandleResponseError(unittest.TestCase):
    def test_ok_response_returns_none(self):
        r = FakeResponse(200)
        self.assertIsNone(_handle_response_error(r))

    def test_401_raises_auth_error(self):
        r = FakeResponse(401, json_data={"error": "bad key"})
        with self.assertRaises(ShodhAuthenticationError):
            _handle_response_error(r)

    def test_404_raises_not_found(self):
        r = FakeResponse(404, json_data={"error": "not found"})
        with self.assertRaises(ShodhNotFoundError):
            _handle_response_error(r)

    def test_400_raises_validation(self):
        r = FakeResponse(400, json_data={"error": "invalid"})
        with self.assertRaises(ShodhValidationError):
            _handle_response_error(r)

    def test_422_raises_validation(self):
        r = FakeResponse(422, json_data={"error": "unprocessable"})
        with self.assertRaises(ShodhValidationError):
            _handle_response_error(r)

    def test_429_raises_rate_limit(self):
        r = FakeResponse(429, json_data={"error": "slow down"}, headers={"Retry-After": "30"})
        with self.assertRaises(ShodhRateLimitError) as ctx:
            _handle_response_error(r)
        self.assertEqual(ctx.exception.retry_after, 30)

    def test_429_without_retry_after(self):
        r = FakeResponse(429, json_data={"error": "slow down"})
        with self.assertRaises(ShodhRateLimitError) as ctx:
            _handle_response_error(r)
        self.assertIsNone(ctx.exception.retry_after)

    def test_500_raises_server_error(self):
        r = FakeResponse(500, json_data={"error": "internal"})
        with self.assertRaises(ShodhServerError) as ctx:
            _handle_response_error(r)
        self.assertEqual(ctx.exception.status_code, 500)

    def test_503_raises_server_error(self):
        r = FakeResponse(503, json_data={"error": "unavailable"})
        with self.assertRaises(ShodhServerError):
            _handle_response_error(r)

    def test_unknown_status_raises_shodh_error(self):
        r = FakeResponse(418, json_data={"error": "teapot"})
        with self.assertRaises(ShodhError):
            _handle_response_error(r)

    def test_json_parse_failure_uses_text(self):
        r = FakeResponse(500)
        r.text = "raw error text"
        r.json = mock.MagicMock(side_effect=Exception("not JSON"))
        with self.assertRaises(ShodhServerError):
            _handle_response_error(r)


# =============================================================================
# Experience
# =============================================================================

class TestExperience(unittest.TestCase):
    def test_to_dict_structure(self):
        exp = Experience(content="test", experience_type="learning", entities=["python"])
        d = exp.to_dict()
        self.assertEqual(d["content"], "test")
        self.assertEqual(d["experience_type"], "Learning")
        self.assertEqual(d["entities"], ["python"])
        self.assertIsNone(d["context"])
        self.assertIsNone(d["embeddings"])
        self.assertEqual(d["related_memories"], [])

    def test_default_type(self):
        exp = Experience(content="hello")
        self.assertEqual(exp.to_dict()["experience_type"], "Conversation")


# =============================================================================
# Exception classes
# =============================================================================

class TestExceptions(unittest.TestCase):
    def test_validation_error_with_field(self):
        e = ShodhValidationError("bad input", field="content")
        self.assertEqual(e.field, "content")

    def test_not_found_error_attributes(self):
        e = ShodhNotFoundError("memory", "abc-123")
        self.assertEqual(e.resource_type, "memory")
        self.assertEqual(e.resource_id, "abc-123")
        self.assertIn("not found", str(e))

    def test_rate_limit_with_retry(self):
        e = ShodhRateLimitError(retry_after=60)
        self.assertEqual(e.retry_after, 60)
        self.assertIn("60s", str(e))

    def test_rate_limit_without_retry(self):
        e = ShodhRateLimitError()
        self.assertIsNone(e.retry_after)

    def test_server_error_status(self):
        e = ShodhServerError(502, "bad gateway")
        self.assertEqual(e.status_code, 502)


# =============================================================================
# Memory.__init__ — API key validation
# =============================================================================

class TestMemoryInit(unittest.TestCase):
    def test_raises_without_api_key(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ShodhAuthenticationError):
                Memory(user_id="test", auto_start=False)

    @mock.patch.object(CLIENT_MODULE.Memory, "_ensure_server_running")
    def test_accepts_explicit_api_key(self, mock_ensure):
        mem = Memory(user_id="test", api_key="my-key", auto_start=False)
        self.assertEqual(mem.api_key, "my-key")

    @mock.patch.object(CLIENT_MODULE.Memory, "_ensure_server_running")
    def test_reads_api_key_from_env(self, mock_ensure):
        with mock.patch.dict(os.environ, {"SHODH_API_KEY": "env-key"}):
            mem = Memory(user_id="test", auto_start=False)
            self.assertEqual(mem.api_key, "env-key")

    @mock.patch.object(CLIENT_MODULE.Memory, "_ensure_server_running")
    def test_auto_start_calls_ensure_server_running(self, mock_ensure):
        Memory(user_id="test", api_key="test-key", auto_start=True)
        mock_ensure.assert_called_once()


# =============================================================================
# Memory.add
# =============================================================================

class TestMemoryAdd(unittest.TestCase):
    def test_add_success(self):
        mem = _build_memory()
        mem._session.post.return_value = FakeResponse(200, {"id": "mem-123"})
        result = mem.add("test content")
        self.assertEqual(result, "mem-123")

    def test_add_with_entities_and_tags(self):
        mem = _build_memory()
        mem._session.post.return_value = FakeResponse(200, {"id": "mem-456"})
        result = mem.add("content", entities=["python"], metadata={"tags": "a,b"})
        self.assertEqual(result, "mem-456")
        call_json = mem._session.post.call_args[1]["json"]
        self.assertIn("python", call_json["tags"])
        self.assertIn("a", call_json["tags"])

    def test_add_missing_id_raises(self):
        mem = _build_memory()
        mem._session.post.return_value = FakeResponse(200, {})
        with self.assertRaises(ShodhError):
            mem.add("test")

    def test_add_connection_error(self):
        import requests as req
        mem = _build_memory()
        mem._session.post.side_effect = req.exceptions.ConnectionError("refused")
        with self.assertRaises(ShodhConnectionError):
            mem.add("test")

    def test_add_timeout_error(self):
        import requests as req
        mem = _build_memory()
        mem._session.post.side_effect = req.exceptions.Timeout("timed out")
        with self.assertRaises(ShodhError):
            mem.add("test")


# =============================================================================
# Memory.search
# =============================================================================

class TestMemorySearch(unittest.TestCase):
    def test_search_returns_memories(self):
        mem = _build_memory()
        mem._session.post.return_value = FakeResponse(200, {"memories": [{"id": "m1"}]})
        result = mem.search(query="test")
        self.assertEqual(len(result), 1)

    def test_search_empty_response(self):
        mem = _build_memory()
        mem._session.post.return_value = FakeResponse(200, {})
        result = mem.search(query="test")
        self.assertEqual(result, [])

    def test_search_requires_query_or_embedding(self):
        mem = _build_memory()
        with self.assertRaises(ShodhValidationError):
            mem.search()

    def test_search_connection_error(self):
        import requests as req
        mem = _build_memory()
        mem._session.post.side_effect = req.exceptions.ConnectionError()
        with self.assertRaises(ShodhConnectionError):
            mem.search(query="test")


# =============================================================================
# Memory.get / get_all / delete / update / history
# =============================================================================

class TestMemoryCRUD(unittest.TestCase):
    def test_get_returns_memory(self):
        mem = _build_memory()
        mem._session.get.return_value = FakeResponse(200, {"id": "m1", "content": "hello"})
        result = mem.get("m1")
        self.assertEqual(result["id"], "m1")

    def test_get_all_returns_list(self):
        mem = _build_memory()
        mem._session.post.return_value = FakeResponse(200, {"memories": [{"id": "m1"}]})
        result = mem.get_all()
        self.assertEqual(len(result), 1)

    def test_get_all_empty_response(self):
        mem = _build_memory()
        mem._session.post.return_value = FakeResponse(200, {})
        self.assertEqual(mem.get_all(), [])

    def test_get_all_connection_error(self):
        import requests as req
        mem = _build_memory()
        mem._session.post.side_effect = req.exceptions.ConnectionError()
        with self.assertRaises(ShodhConnectionError):
            mem.get_all()

    def test_delete_success(self):
        mem = _build_memory()
        mem._session.delete.return_value = FakeResponse(200)
        mem.delete("m1")  # Should not raise

    def test_update_success(self):
        mem = _build_memory()
        mem._session.put.return_value = FakeResponse(200)
        mem.update("m1", "new content")

    def test_history_returns_events(self):
        mem = _build_memory()
        mem._session.post.return_value = FakeResponse(200, {"events": [{"type": "create"}]})
        result = mem.history()
        self.assertEqual(len(result), 1)

    def test_history_empty(self):
        mem = _build_memory()
        mem._session.post.return_value = FakeResponse(200, {})
        self.assertEqual(mem.history(), [])


# =============================================================================
# Memory.forget_* methods
# =============================================================================

class TestMemoryForget(unittest.TestCase):
    def test_forget_me(self):
        mem = _build_memory()
        mem._session.delete.return_value = FakeResponse(200)
        mem.forget_me()  # Should not raise

    def test_forget_by_age(self):
        mem = _build_memory()
        mem._session.post.return_value = FakeResponse(200, {"forgotten_count": 5})
        self.assertEqual(mem.forget_by_age(30), 5)

    def test_forget_by_age_missing_count(self):
        mem = _build_memory()
        mem._session.post.return_value = FakeResponse(200, {})
        self.assertEqual(mem.forget_by_age(30), 0)

    def test_forget_by_importance(self):
        mem = _build_memory()
        mem._session.post.return_value = FakeResponse(200, {"forgotten_count": 3})
        self.assertEqual(mem.forget_by_importance(0.5), 3)

    def test_forget_by_importance_invalid_threshold(self):
        mem = _build_memory()
        with self.assertRaises(ShodhValidationError):
            mem.forget_by_importance(1.5)
        with self.assertRaises(ShodhValidationError):
            mem.forget_by_importance(-0.1)

    def test_forget_by_pattern(self):
        mem = _build_memory()
        mem._session.post.return_value = FakeResponse(200, {"forgotten_count": 2})
        self.assertEqual(mem.forget_by_pattern("test.*"), 2)

    def test_forget_by_tags(self):
        mem = _build_memory()
        mem._session.post.return_value = FakeResponse(200, {"forgotten_count": 4})
        self.assertEqual(mem.forget_by_tags(["temp"]), 4)

    def test_forget_by_date(self):
        mem = _build_memory()
        mem._session.post.return_value = FakeResponse(200, {"forgotten_count": 1})
        self.assertEqual(mem.forget_by_date("2024-01-01T00:00:00Z", "2024-01-31T23:59:59Z"), 1)


# =============================================================================
# Memory.batch_remember
# =============================================================================

class TestBatchRemember(unittest.TestCase):
    def test_batch_success(self):
        mem = _build_memory()
        mem._session.post.return_value = FakeResponse(200, {
            "created": 2, "failed": 0, "memory_ids": ["m1", "m2"], "errors": []
        })
        result = mem.batch_remember([{"content": "a"}, {"content": "b"}])
        self.assertEqual(result["created"], 2)

    def test_batch_connection_error(self):
        import requests as req
        mem = _build_memory()
        mem._session.post.side_effect = req.exceptions.ConnectionError()
        with self.assertRaises(ShodhConnectionError):
            mem.batch_remember([{"content": "a"}])


# =============================================================================
# Memory.stats
# =============================================================================

class TestMemoryStats(unittest.TestCase):
    def test_stats_returns_dataclass(self):
        mem = _build_memory()
        mem._session.get.return_value = FakeResponse(200, {
            "total_memories": 10,
            "working_memory_count": 3,
            "session_memory_count": 5,
            "long_term_memory_count": 2,
        })
        s = mem.stats()
        self.assertIsInstance(s, MemoryStats)
        self.assertEqual(s.total_memories, 10)

    def test_stats_missing_fields_default_zero(self):
        mem = _build_memory()
        mem._session.get.return_value = FakeResponse(200, {})
        s = mem.stats()
        self.assertEqual(s.total_memories, 0)


# =============================================================================
# Memory.recall / remember / context_summary
# =============================================================================

class TestRecallRememberContextSummary(unittest.TestCase):
    def test_recall(self):
        mem = _build_memory()
        mem._session.post.return_value = FakeResponse(200, {"memories": [{"id": "m1"}]})
        result = mem.recall("test query")
        self.assertEqual(len(result), 1)

    def test_recall_timeout(self):
        import requests as req
        mem = _build_memory()
        mem._session.post.side_effect = req.exceptions.Timeout()
        with self.assertRaises(ShodhError):
            mem.recall("test")

    def test_recall_connection_error(self):
        import requests as req
        mem = _build_memory()
        mem._session.post.side_effect = req.exceptions.ConnectionError()
        with self.assertRaises(ShodhConnectionError):
            mem.recall("test query")

    def test_remember_delegates_to_add(self):
        mem = _build_memory()
        mem._session.post.return_value = FakeResponse(200, {"id": "m1"})
        result = mem.remember("content", memory_type="Decision")
        self.assertEqual(result, "m1")

    def test_context_summary(self):
        mem = _build_memory()
        # context_summary calls get_all internally
        mem._session.post.return_value = FakeResponse(200, {"memories": [
            {"memory_type": "Decision", "content": "use postgres"},
            {"memory_type": "Learning", "content": "async is fast"},
        ]})
        result = mem.context_summary()
        self.assertIn("total_memories", result)


# =============================================================================
# Memory.visualize / graph_stats / export_graph
# =============================================================================

class TestVisualizationMethods(unittest.TestCase):
    def test_visualize_returns_url(self):
        mem = _build_memory()
        url = mem.visualize(open_browser=False)
        self.assertIn("static/index.html", url)
        self.assertIn("test-user", url)

    def test_graph_stats(self):
        mem = _build_memory()
        mem._session.get.return_value = FakeResponse(200, {"nodes": 10, "edges": 20})
        result = mem.graph_stats()
        self.assertEqual(result["nodes"], 10)

    def test_export_graph(self):
        mem = _build_memory()
        r = FakeResponse(200)
        r.text = "digraph { a -> b }"
        mem._session.get.return_value = r
        result = mem.export_graph()
        self.assertIn("digraph", result)


# =============================================================================
# Platform detection and binary finding
# =============================================================================

class TestPlatformDetection(unittest.TestCase):
    def test_get_binary_name_darwin(self):
        mem = _build_memory()
        with mock.patch("platform.system", return_value="Darwin"):
            self.assertEqual(mem._get_binary_name(), "shodh-memory-darwin")

    def test_get_binary_name_linux(self):
        mem = _build_memory()
        with mock.patch("platform.system", return_value="Linux"):
            self.assertEqual(mem._get_binary_name(), "shodh-memory")

    def test_get_binary_name_windows(self):
        mem = _build_memory()
        with mock.patch("platform.system", return_value="Windows"):
            self.assertEqual(mem._get_binary_name(), "shodh-memory.exe")

    @mock.patch("pathlib.Path.exists", return_value=False)
    @mock.patch("shutil.which", return_value=None)
    def test_find_binary_not_found(self, mock_which, mock_exists):
        mem = _build_memory()
        result = mem._find_binary("shodh-memory")
        self.assertIsNone(result)

    @mock.patch("shutil.which", return_value="/usr/local/bin/shodh-memory")
    @mock.patch("pathlib.Path.exists", return_value=False)
    def test_find_binary_in_path(self, mock_exists, mock_which):
        mem = _build_memory()
        result = mem._find_binary("shodh-memory")
        self.assertIsNotNone(result)


# =============================================================================
# Server lifecycle
# =============================================================================

class TestServerLifecycle(unittest.TestCase):
    def test_stop_server_does_nothing_without_process(self):
        mem = _build_memory()
        mem._server_process = None
        mem._stop_server()  # No error

    def test_stop_server_terminates_process(self):
        mem = _build_memory()
        mock_proc = mock.MagicMock()
        mem._server_process = mock_proc
        mem._stop_server()
        mock_proc.terminate.assert_called_once()

    def test_stop_server_kills_on_timeout(self):
        import subprocess
        mem = _build_memory()
        mock_proc = mock.MagicMock()
        mock_proc.wait.side_effect = subprocess.TimeoutExpired(cmd="test", timeout=5)
        mem._server_process = mock_proc
        mem._stop_server()
        mock_proc.kill.assert_called_once()


# =============================================================================
# Connection error propagation for all methods
# =============================================================================

class TestConnectionErrors(unittest.TestCase):
    """Verify all HTTP methods properly wrap ConnectionError."""

    def _mem_with_conn_error(self, method):
        import requests as req
        mem = _build_memory()
        getattr(mem._session, method).side_effect = req.exceptions.ConnectionError()
        return mem

    def test_get_connection_error(self):
        mem = self._mem_with_conn_error("get")
        with self.assertRaises(ShodhConnectionError):
            mem.get("m1")

    def test_stats_connection_error(self):
        mem = self._mem_with_conn_error("get")
        with self.assertRaises(ShodhConnectionError):
            mem.stats()

    def test_delete_connection_error(self):
        mem = self._mem_with_conn_error("delete")
        with self.assertRaises(ShodhConnectionError):
            mem.delete("m1")

    def test_forget_me_connection_error(self):
        mem = self._mem_with_conn_error("delete")
        with self.assertRaises(ShodhConnectionError):
            mem.forget_me()

    def test_update_connection_error(self):
        mem = self._mem_with_conn_error("put")
        with self.assertRaises(ShodhConnectionError):
            mem.update("m1", "new")

    def test_graph_stats_connection_error(self):
        mem = self._mem_with_conn_error("get")
        with self.assertRaises(ShodhConnectionError):
            mem.graph_stats()

    def test_export_graph_connection_error(self):
        mem = self._mem_with_conn_error("get")
        with self.assertRaises(ShodhConnectionError):
            mem.export_graph()

    def test_history_connection_error(self):
        mem = self._mem_with_conn_error("post")
        with self.assertRaises(ShodhConnectionError):
            mem.history()

    def test_forget_by_age_connection_error(self):
        mem = self._mem_with_conn_error("post")
        with self.assertRaises(ShodhConnectionError):
            mem.forget_by_age(30)

    def test_forget_by_importance_connection_error(self):
        mem = self._mem_with_conn_error("post")
        with self.assertRaises(ShodhConnectionError):
            mem.forget_by_importance(0.5)

    def test_forget_by_pattern_connection_error(self):
        mem = self._mem_with_conn_error("post")
        with self.assertRaises(ShodhConnectionError):
            mem.forget_by_pattern("test")

    def test_forget_by_tags_connection_error(self):
        mem = self._mem_with_conn_error("post")
        with self.assertRaises(ShodhConnectionError):
            mem.forget_by_tags(["a"])

    def test_forget_by_date_connection_error(self):
        mem = self._mem_with_conn_error("post")
        with self.assertRaises(ShodhConnectionError):
            mem.forget_by_date("2024-01-01T00:00:00Z", "2024-12-31T23:59:59Z")


# =============================================================================
# _ensure_server_running
# =============================================================================

class TestEnsureServerRunning(unittest.TestCase):
    def test_server_already_running(self):
        import requests as req
        mem = _build_memory()
        with mock.patch.object(CLIENT_MODULE.requests, "get") as mock_get:
            mock_get.return_value = FakeResponse(200)
            mem._ensure_server_running()
            mock_get.assert_called_once()

    def test_server_not_running_starts_and_succeeds(self):
        import requests as req
        mem = _build_memory()
        call_count = {"n": 0}

        def health_check(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise req.exceptions.RequestException("down")
            return FakeResponse(200)

        with mock.patch.object(CLIENT_MODULE.requests, "get", side_effect=health_check):
            with mock.patch.object(mem, "_start_server"):
                with mock.patch.object(CLIENT_MODULE.time, "sleep"):
                    mem._ensure_server_running()

    def test_server_fails_to_start(self):
        import requests as req
        mem = _build_memory()
        with mock.patch.object(CLIENT_MODULE.requests, "get", side_effect=req.exceptions.RequestException("down")):
            with mock.patch.object(mem, "_start_server"):
                with mock.patch.object(CLIENT_MODULE.time, "sleep"):
                    with self.assertRaises(RuntimeError):
                        mem._ensure_server_running()


# =============================================================================
# _start_server
# =============================================================================

class TestStartServer(unittest.TestCase):
    def test_start_success(self):
        mem = _build_memory()
        mock_path = mock.MagicMock()
        mock_path.exists.return_value = True
        mock_path.__str__ = lambda self: "/usr/bin/shodh-memory"
        with mock.patch.object(mem, "_get_binary_name", return_value="shodh-memory"):
            with mock.patch.object(mem, "_find_binary", return_value=mock_path):
                with mock.patch.object(CLIENT_MODULE.subprocess, "Popen") as mock_popen:
                    with mock.patch.object(CLIENT_MODULE.atexit, "register"):
                        mem._start_server()
                        mock_popen.assert_called_once()

    def test_start_binary_not_found(self):
        mem = _build_memory()
        with mock.patch.object(mem, "_get_binary_name", return_value="shodh-memory"):
            with mock.patch.object(mem, "_find_binary", return_value=None):
                with self.assertRaises(RuntimeError):
                    mem._start_server()

    def test_start_binary_path_not_exists(self):
        mem = _build_memory()
        mock_path = mock.MagicMock()
        mock_path.exists.return_value = False
        with mock.patch.object(mem, "_get_binary_name", return_value="shodh-memory"):
            with mock.patch.object(mem, "_find_binary", return_value=mock_path):
                with self.assertRaises(RuntimeError):
                    mem._start_server()


# =============================================================================
# _find_binary — local paths
# =============================================================================

class TestFindBinaryLocal(unittest.TestCase):
    def test_found_in_bin_dir(self):
        mem = _build_memory()
        with mock.patch("pathlib.Path.exists", return_value=True):
            result = mem._find_binary("shodh-memory")
            self.assertIsNotNone(result)

    def test_found_in_cargo_target(self):
        mem = _build_memory()
        call_count = {"n": 0}

        def exists_side_effect(self_path=None):
            call_count["n"] += 1
            # First call (bin_dir) returns False, second (cargo) returns True
            return call_count["n"] >= 2

        with mock.patch("pathlib.Path.exists", side_effect=exists_side_effect):
            with mock.patch("shutil.which", return_value=None):
                result = mem._find_binary("shodh-memory")
                self.assertIsNotNone(result)


# =============================================================================
# search with query_embedding (deprecated)
# =============================================================================

class TestSearchDeprecation(unittest.TestCase):
    def test_query_embedding_warning(self):
        mem = _build_memory()
        mem._session.post.return_value = FakeResponse(200, {"memories": []})
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            mem.search(query_embedding=[0.1, 0.2, 0.3])
            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            self.assertEqual(len(deprecation_warnings), 1)

    def test_search_timeout(self):
        import requests as req
        mem = _build_memory()
        mem._session.post.side_effect = req.exceptions.Timeout()
        with self.assertRaises(ShodhError):
            mem.search(query="test")


# =============================================================================
# visualize with open_browser=True
# =============================================================================

class TestVisualizeOpenBrowser(unittest.TestCase):
    def test_opens_browser(self):
        mem = _build_memory()
        with mock.patch("webbrowser.open") as mock_open:
            url = mem.visualize(open_browser=True)
            mock_open.assert_called_once_with(url)


# =============================================================================
# context_summary — Pattern and Error branches
# =============================================================================

class TestContextSummaryBranches(unittest.TestCase):
    def test_pattern_and_error_types(self):
        mem = _build_memory()
        mem._session.post.return_value = FakeResponse(200, {"memories": [
            {"memory_type": "Pattern", "content": "singleton pattern"},
            {"memory_type": "Error", "content": "null pointer"},
            {"memory_type": "Context", "content": "project ctx"},
        ]})
        result = mem.context_summary()
        self.assertGreater(len(result["patterns"]), 0)
        self.assertGreater(len(result["errors"]), 0)
        self.assertGreater(len(result["context"]), 0)

    def test_context_summary_flags(self):
        mem = _build_memory()
        mem._session.post.return_value = FakeResponse(200, {"memories": [
            {"memory_type": "Context", "content": "ctx"},
            {"memory_type": "Decision", "content": "dec"},
            {"memory_type": "Learning", "content": "learn"},
        ]})
        result = mem.context_summary(include_context=False, include_decisions=False, include_learnings=False)
        self.assertEqual(result["context"], [])
        self.assertEqual(result["decisions"], [])
        self.assertEqual(result["learnings"], [])


# =============================================================================
# brain_state
# =============================================================================

class TestBrainState(unittest.TestCase):
    def test_brain_state_success(self):
        mem = _build_memory()
        mem._session.get.return_value = FakeResponse(200, {
            "working_memory": [], "session_memory": [], "longterm_memory": [],
            "stats": {"total_memories": 10}
        })
        result = mem.brain_state()
        self.assertEqual(result["stats"]["total_memories"], 10)

    def test_brain_state_connection_error(self):
        import requests as req
        mem = _build_memory()
        mem._session.get.side_effect = req.exceptions.ConnectionError()
        with self.assertRaises(ShodhConnectionError):
            mem.brain_state()


# =============================================================================
# Context manager support
# =============================================================================

class TestContextManager(unittest.TestCase):
    def test_enter_returns_self(self):
        mem = _build_memory()
        result = mem.__enter__()
        self.assertIs(result, mem)

    def test_exit_stops_server(self):
        mem = _build_memory()
        mock_proc = mock.MagicMock()
        mem._server_process = mock_proc
        mem.__exit__(None, None, None)
        mock_proc.terminate.assert_called_once()


# =============================================================================
# recall timeout
# =============================================================================

class TestRecallTimeout(unittest.TestCase):
    def test_recall_timeout_error(self):
        import requests as req
        mem = _build_memory()
        mem._session.post.side_effect = req.exceptions.Timeout()
        with self.assertRaises(ShodhError):
            mem.recall("test query")


# =============================================================================
# get_all additional branch
# =============================================================================

class TestGetAllExtra(unittest.TestCase):
    def test_get_all_with_importance_threshold(self):
        mem = _build_memory()
        mem._session.post.return_value = FakeResponse(200, {"memories": [{"id": "m1"}]})
        result = mem.get_all(limit=50, importance_threshold=0.5)
        self.assertEqual(len(result), 1)
        call_json = mem._session.post.call_args[1]["json"]
        self.assertEqual(call_json["importance_threshold"], 0.5)


if __name__ == "__main__":
    unittest.main()
