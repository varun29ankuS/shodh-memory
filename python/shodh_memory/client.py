"""Python client for Shodh-Memory with auto-start server

Production-grade client with:
- Typed exceptions for precise error handling
- Automatic retry with exponential backoff
- Connection pooling
- Timeout handling
"""

import atexit
import os
import platform
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# ==============================================================================
# Exception Hierarchy
# ==============================================================================

class ShodhError(Exception):
    """Base exception for all Shodh-Memory errors."""
    pass


class ShodhConnectionError(ShodhError):
    """Failed to connect to the Shodh-Memory server."""
    pass


class ShodhAuthenticationError(ShodhError):
    """Authentication failed (invalid or missing API key)."""
    pass


class ShodhValidationError(ShodhError):
    """Request validation failed (invalid input)."""

    def __init__(self, message: str, field: Optional[str] = None):
        super().__init__(message)
        self.field = field


class ShodhNotFoundError(ShodhError):
    """Requested resource (memory, user, entity) not found."""

    def __init__(self, resource_type: str, resource_id: str):
        super().__init__(f"{resource_type} not found: {resource_id}")
        self.resource_type = resource_type
        self.resource_id = resource_id


class ShodhRateLimitError(ShodhError):
    """Rate limit exceeded. Retry after the specified time."""

    def __init__(self, retry_after: Optional[int] = None):
        msg = f"Rate limit exceeded. Retry after {retry_after}s" if retry_after else "Rate limit exceeded"
        super().__init__(msg)
        self.retry_after = retry_after


class ShodhServerError(ShodhError):
    """Server-side error (5xx status codes)."""

    def __init__(self, status_code: int, message: str):
        super().__init__(f"Server error ({status_code}): {message}")
        self.status_code = status_code


def _handle_response_error(response: requests.Response, context: str = "request") -> None:
    """Convert HTTP errors to typed exceptions."""
    if response.ok:
        return

    status = response.status_code

    try:
        body = response.json()
        message = body.get("error", body.get("message", response.text))
    except Exception:
        message = response.text or f"HTTP {status}"

    if status == 401:
        raise ShodhAuthenticationError(f"Authentication failed: {message}")
    elif status == 404:
        raise ShodhNotFoundError("resource", context)
    elif status == 422 or status == 400:
        raise ShodhValidationError(message)
    elif status == 429:
        retry_after = response.headers.get("Retry-After")
        raise ShodhRateLimitError(int(retry_after) if retry_after else None)
    elif status >= 500:
        raise ShodhServerError(status, message)
    else:
        raise ShodhError(f"Request failed ({status}): {message}")


@dataclass
class Experience:
    """Experience to store in memory"""

    content: str
    experience_type: str = "conversation"  # conversation, decision, error, learning, discovery, pattern, context, task
    entities: List[str] = field(default_factory=list)
    metadata: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "experience_type": self.experience_type.capitalize(),
            "content": self.content,
            "context": None,
            "entities": self.entities,
            "metadata": self.metadata,
            "embeddings": None,
            "related_memories": [],
            "causal_chain": [],
            "outcomes": []
        }


@dataclass
class MemoryStats:
    """Memory system statistics"""

    total_memories: int
    working_memory_count: int
    session_memory_count: int
    longterm_memory_count: int


class Memory:
    """Production-grade memory client with auto-start server

    Usage:
        memory = Memory(user_id="alice")
        memory.add("I love Python", experience_type="learning")
        results = memory.search("Python programming")

    With API key authentication:
        memory = Memory(user_id="alice", api_key="your-api-key")

    Error handling:
        try:
            memory.add("test")
        except ShodhValidationError as e:
            print(f"Invalid input: {e}")
        except ShodhAuthenticationError:
            print("Invalid API key")
        except ShodhRateLimitError as e:
            time.sleep(e.retry_after or 60)
    """

    def __init__(
        self,
        user_id: str = "default",
        port: int = 3030,
        storage_path: Optional[str] = None,
        auto_start: bool = True,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
        max_retries: int = 3,
    ):
        """Initialize memory client

        Args:
            user_id: Unique identifier for this user (enables multi-user isolation)
            port: Port for the memory server (default: 3030)
            storage_path: Path to store memory data (default: ./shodh_memory_data)
            auto_start: Automatically start server if not running (default: True)
            api_key: API key for authentication (optional, uses env SHODH_API_KEY if not set)
            timeout: Request timeout in seconds (default: 30)
            max_retries: Maximum number of retry attempts (default: 3)
        """
        self.user_id = user_id
        self.port = port
        self.base_url = f"http://localhost:{port}"
        self.storage_path = storage_path or "./shodh_memory_data"
        self.api_key = api_key or os.environ.get("SHODH_API_KEY", "shodh-dev-key-change-in-production")
        self.timeout = timeout

        self._server_process = None

        # Configure session with retry logic and connection pooling
        self._session = requests.Session()

        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=0.5,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "POST", "PUT", "DELETE", "OPTIONS"],
        )

        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=10,
            pool_maxsize=20,
        )
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)

        # Set default headers
        self._session.headers.update({
            "Content-Type": "application/json",
            "X-API-Key": self.api_key,
        })

        if auto_start:
            self._ensure_server_running()

    def _ensure_server_running(self):
        """Start server if not already running"""
        # Check if server is already running
        try:
            response = requests.get(f"{self.base_url}/health", timeout=1)
            if response.status_code == 200:
                return  # Server already running
        except requests.exceptions.RequestException:
            pass

        # Start server
        print("🧠 Starting Shodh-Memory server...")
        self._start_server()

        # Wait for server to be ready
        max_retries = 30
        for i in range(max_retries):
            try:
                response = requests.get(f"{self.base_url}/health", timeout=1)
                if response.status_code == 200:
                    print("✅ Shodh-Memory server ready")
                    return
            except requests.exceptions.RequestException:
                pass
            time.sleep(0.5)

        raise RuntimeError("Failed to start Shodh-Memory server")

    def _start_server(self):
        """Start the Rust server binary"""
        # Find binary path
        binary_name = self._get_binary_name()
        binary_path = self._find_binary(binary_name)

        if not binary_path or not binary_path.exists():
            raise RuntimeError(
                f"Shodh-Memory binary not found: {binary_name}\n"
                "Please ensure the binary is built or installed correctly."
            )

        # Set environment variables
        env = os.environ.copy()
        env["SHODH_MEMORY_PORT"] = str(self.port)
        env["SHODH_MEMORY_PATH"] = self.storage_path
        env["RUST_LOG"] = "shodh_memory=info"

        # Start server process
        self._server_process = subprocess.Popen(
            [str(binary_path)],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        # Register cleanup on exit
        atexit.register(self._stop_server)

    def _get_binary_name(self) -> str:
        """Get platform-specific binary name"""
        system = platform.system().lower()
        if system == "windows":
            return "shodh-memory.exe"
        elif system == "darwin":
            return "shodh-memory-darwin"
        else:
            return "shodh-memory"

    def _find_binary(self, binary_name: str) -> Optional[Path]:
        """Find binary in common locations"""
        # Check in package directory (for pip install)
        package_dir = Path(__file__).parent.parent
        bin_dir = package_dir / "bin"

        if (bin_dir / binary_name).exists():
            return bin_dir / binary_name

        # Check in cargo target directory (for development)
        cargo_target = Path("../../target/release") / binary_name
        if cargo_target.exists():
            return cargo_target

        # Check in PATH
        import shutil
        binary_in_path = shutil.which(binary_name)
        if binary_in_path:
            return Path(binary_in_path)

        return None

    def _stop_server(self):
        """Stop the server process"""
        if self._server_process:
            self._server_process.terminate()
            try:
                self._server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._server_process.kill()

    def add(
        self,
        content: str,
        experience_type: str = "conversation",
        entities: Optional[List[str]] = None,
        metadata: Optional[Dict[str, str]] = None
    ) -> str:
        """Add a memory

        Args:
            content: The content to remember
            experience_type: Type of experience (conversation, decision, error, learning, etc.)
            entities: Named entities in the content
            metadata: Additional metadata

        Returns:
            Memory ID

        Raises:
            ShodhValidationError: If content is empty or invalid
            ShodhAuthenticationError: If API key is invalid
            ShodhRateLimitError: If rate limit is exceeded
            ShodhServerError: If server encounters an error
        """
        experience = Experience(
            content=content,
            experience_type=experience_type,
            entities=entities or [],
            metadata=metadata or {}
        )

        try:
            response = self._session.post(
                f"{self.base_url}/api/record",
                json={
                    "user_id": self.user_id,
                    "experience": experience.to_dict()
                },
                timeout=self.timeout
            )
        except requests.exceptions.ConnectionError as e:
            raise ShodhConnectionError(f"Failed to connect to server: {e}") from e
        except requests.exceptions.Timeout as e:
            raise ShodhError(f"Request timed out: {e}") from e

        _handle_response_error(response, context="add memory")
        return response.json()["memory_id"]

    def search(
        self,
        query: Optional[str] = None,
        query_embedding: Optional[List[float]] = None,
        max_results: int = 10,
        importance_threshold: Optional[float] = None
    ) -> List[dict]:
        """Search memories

        Args:
            query: Text query (keyword search)
            query_embedding: Query embedding vector for semantic search
            max_results: Maximum number of results
            importance_threshold: Minimum importance score (0.0 - 1.0)

        Returns:
            List of matching memories

        Raises:
            ShodhValidationError: If neither query nor query_embedding provided
            ShodhAuthenticationError: If API key is invalid
            ShodhRateLimitError: If rate limit is exceeded

        Examples:
            # Keyword search
            results = memory.search(query="Python programming")

            # Semantic search (user provides embedding)
            embedding = model.encode("favorite languages")
            results = memory.search(query_embedding=embedding.tolist())

            # Hybrid search (both keyword and semantic)
            results = memory.search(
                query="Python",
                query_embedding=embedding.tolist()
            )
        """
        if query is None and query_embedding is None:
            raise ShodhValidationError("Must provide either query or query_embedding", field="query")

        try:
            response = self._session.post(
                f"{self.base_url}/api/retrieve",
                json={
                    "user_id": self.user_id,
                    "query_text": query,
                    "query_embedding": query_embedding,
                    "max_results": max_results,
                    "importance_threshold": importance_threshold
                },
                timeout=self.timeout
            )
        except requests.exceptions.ConnectionError as e:
            raise ShodhConnectionError(f"Failed to connect to server: {e}") from e
        except requests.exceptions.Timeout as e:
            raise ShodhError(f"Request timed out: {e}") from e

        _handle_response_error(response, context="search")
        return response.json()["memories"]

    def stats(self) -> MemoryStats:
        """Get memory statistics

        Returns:
            MemoryStats object with counts for each memory tier

        Raises:
            ShodhAuthenticationError: If API key is invalid
        """
        try:
            response = self._session.get(
                f"{self.base_url}/api/users/{self.user_id}/stats",
                timeout=self.timeout
            )
        except requests.exceptions.ConnectionError as e:
            raise ShodhConnectionError(f"Failed to connect to server: {e}") from e

        _handle_response_error(response, context=f"stats for user {self.user_id}")
        data = response.json()
        return MemoryStats(
            total_memories=data.get("total_memories", 0),
            working_memory_count=data.get("working_memory_count", 0),
            session_memory_count=data.get("session_memory_count", 0),
            longterm_memory_count=data.get("long_term_memory_count", 0),
        )

    def get(self, memory_id: str) -> dict:
        """Get specific memory by ID

        Args:
            memory_id: UUID of the memory

        Returns:
            Memory object

        Raises:
            ShodhNotFoundError: If memory doesn't exist
            ShodhValidationError: If memory_id is invalid
        """
        try:
            response = self._session.get(
                f"{self.base_url}/api/memory/{memory_id}",
                params={"user_id": self.user_id},
                timeout=self.timeout
            )
        except requests.exceptions.ConnectionError as e:
            raise ShodhConnectionError(f"Failed to connect to server: {e}") from e

        _handle_response_error(response, context=f"memory {memory_id}")
        return response.json()

    def get_all(
        self,
        limit: int = 100,
        importance_threshold: Optional[float] = None
    ) -> List[dict]:
        """Get all memories for this user

        Args:
            limit: Maximum number of memories
            importance_threshold: Minimum importance score

        Returns:
            List of memories
        """
        try:
            response = self._session.post(
                f"{self.base_url}/api/memories",
                json={
                    "user_id": self.user_id,
                    "limit": limit,
                    "importance_threshold": importance_threshold
                },
                timeout=self.timeout
            )
        except requests.exceptions.ConnectionError as e:
            raise ShodhConnectionError(f"Failed to connect to server: {e}") from e

        _handle_response_error(response, context="get all memories")
        return response.json()["memories"]

    def update(
        self,
        memory_id: str,
        content: str,
        embeddings: Optional[List[float]] = None
    ) -> None:
        """Update existing memory

        Args:
            memory_id: UUID of memory to update
            content: New content
            embeddings: New embedding vector (optional)

        Raises:
            ShodhNotFoundError: If memory doesn't exist
            ShodhValidationError: If content is invalid
        """
        try:
            response = self._session.put(
                f"{self.base_url}/api/memory/{memory_id}",
                json={
                    "user_id": self.user_id,
                    "content": content,
                    "embeddings": embeddings
                },
                timeout=self.timeout
            )
        except requests.exceptions.ConnectionError as e:
            raise ShodhConnectionError(f"Failed to connect to server: {e}") from e

        _handle_response_error(response, context=f"update memory {memory_id}")

    def delete(self, memory_id: str) -> None:
        """Delete specific memory

        Args:
            memory_id: UUID of memory to delete

        Raises:
            ShodhNotFoundError: If memory doesn't exist
        """
        try:
            response = self._session.delete(
                f"{self.base_url}/api/memory/{memory_id}",
                params={"user_id": self.user_id},
                timeout=self.timeout
            )
        except requests.exceptions.ConnectionError as e:
            raise ShodhConnectionError(f"Failed to connect to server: {e}") from e

        _handle_response_error(response, context=f"delete memory {memory_id}")

    def history(self, memory_id: Optional[str] = None) -> List[dict]:
        """Get audit trail of memory changes

        Args:
            memory_id: Optional - filter by specific memory

        Returns:
            List of history events
        """
        try:
            response = self._session.post(
                f"{self.base_url}/api/memories/history",
                json={
                    "user_id": self.user_id,
                    "memory_id": memory_id
                },
                timeout=self.timeout
            )
        except requests.exceptions.ConnectionError as e:
            raise ShodhConnectionError(f"Failed to connect to server: {e}") from e

        _handle_response_error(response, context="history")
        return response.json()["events"]

    def forget_me(self) -> None:
        """Delete all memories for this user (GDPR compliance)

        Raises:
            ShodhAuthenticationError: If API key is invalid
        """
        try:
            response = self._session.delete(
                f"{self.base_url}/api/users/{self.user_id}",
                timeout=self.timeout
            )
        except requests.exceptions.ConnectionError as e:
            raise ShodhConnectionError(f"Failed to connect to server: {e}") from e

        _handle_response_error(response, context=f"delete user {self.user_id}")
        print(f"🧠 All memories deleted for user: {self.user_id}")

    def __enter__(self):
        """Context manager support"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup on context exit"""
        self._stop_server()
