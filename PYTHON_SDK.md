# Shodh-Memory Python SDK

Native Python bindings for shodh-memory using PyO3.

## Features

- **Zero IPC overhead**: Direct in-process function calls
- **~5-10x faster** than HTTP/REST API
- **Single process**: No server management required
- **NumPy integration**: Direct array passing (zero-copy)
- **Type hints**: Full IDE autocomplete support
- **ABI3 compatible**: Single wheel works across Python 3.8-3.12

## Installation

### From Source (Development)

```bash
# Install maturin
pip install maturin

# Build and install in development mode
cd shodh-memory
maturin develop --features python --release

# Or build wheel for distribution
maturin build --features python --release
```

### From PyPI (Once Published)

```bash
pip install shodh-memory
```

## Quick Start

```python
import shodh_memory

# Initialize memory system (fully offline)
memory = shodh_memory.MemorySystem(storage_path="./my_memory")

# Record experiences
memory_id = memory.record(
    content="Detected obstacle at (10, 20)",
    experience_type="observation",
    entities=["obstacle_1", "zone_a"]
)

# Retrieve relevant memories
results = memory.retrieve(
    query="obstacles nearby",
    max_results=5,
    mode="hybrid"
)

for mem in results:
    print(f"{mem['content']} (importance: {mem['importance']:.2f})")

# Flush before shutdown
memory.flush()
```

## ROS2 Integration

```python
import rclpy
from rclpy.node import Node
import shodh_memory

class MemoryNode(Node):
    def __init__(self):
        super().__init__('shodh_memory_node')
        self.memory = shodh_memory.MemorySystem(
            storage_path="./ros2_memory"
        )
        
        # Subscribe to observations
        self.subscription = self.create_subscription(
            String,
            '/robot/observations',
            self.observation_callback,
            10
        )
    
    def observation_callback(self, msg):
        # Record observation in memory
        self.memory.record(
            content=msg.data,
            experience_type="observation"
        )
        
        # Query related memories
        results = self.memory.retrieve(
            query=msg.data,
            max_results=3
        )
        
        # Process results...
```

## Performance

| Operation | Native (PyO3) | HTTP API | Improvement |
|-----------|---------------|----------|-------------|
| record() | 45-85ms | 50-95ms | ~5-10ms faster |
| retrieve() | 10-30ms | 15-40ms | ~5-10ms faster |
| Cached retrieve() | <5ms | 8-15ms | ~3-10ms faster |

**Key advantages**:
- No network serialization overhead
- No IPC (inter-process communication)
- Single process deployment
- Zero-copy array passing (NumPy)

## API Reference

### MemorySystem

```python
class MemorySystem:
    def __init__(
        self,
        storage_path: Optional[str] = "./shodh_data",
        user_id: Optional[str] = None
    )
```

Creates a new memory system instance.

**Args**:
- `storage_path`: Path for persistent storage (default: `./shodh_data`)
- `user_id`: Optional user ID for multi-tenant isolation

---

```python
def record(
    self,
    content: str,
    experience_type: str = "observation",
    entities: Optional[List[str]] = None,
    metadata: Optional[Dict[str, str]] = None
) -> str
```

Records a new experience.

**Args**:
- `content`: Text content of the experience
- `experience_type`: One of: `observation`, `action`, `thought`, `goal`, `outcome`, `error`, `question`
- `entities`: List of entities for graph memory
- `metadata`: Additional key-value metadata

**Returns**: Memory ID as string

---

```python
def retrieve(
    self,
    query: str,
    max_results: int = 10,
    mode: str = "hybrid",
    min_importance: Optional[float] = None
) -> List[Dict[str, Any]]
```

Retrieves memories matching a query.

**Args**:
- `query`: Query text for semantic search
- `max_results`: Maximum number of results (default: 10)
- `mode`: Retrieval mode: `semantic`, `temporal`, or `hybrid` (default: `hybrid`)
- `min_importance`: Minimum importance threshold 0.0-1.0

**Returns**: List of memory dictionaries

---

```python
def get_stats(self) -> Dict[str, int]
```

Returns memory system statistics.

**Returns**: Dictionary with keys: `working_count`, `session_count`, `total_records`, `total_retrievals`, `cache_hits`, `cache_misses`

---

```python
def flush(self) -> None
```

Flushes all data to disk. Call before shutdown.

## Building for Production

### Cross-platform wheels

```bash
# Install cross-compilation tools
pip install maturin

# Build for current platform
maturin build --release --features python

# Build for multiple Python versions (abi3)
maturin build --release --features python --interpreter python3.8 python3.9 python3.10 python3.11 python3.12

# Wheels will be in: target/wheels/
```

### Docker deployment

```dockerfile
FROM rust:1.75 as builder
WORKDIR /app
COPY . .
RUN pip install maturin
RUN maturin build --release --features python

FROM python:3.11-slim
COPY --from=builder /app/target/wheels/*.whl /tmp/
RUN pip install /tmp/*.whl
```

### Jetson Nano / ARM64

```bash
# On Jetson device
pip install maturin
maturin build --release --features python
pip install target/wheels/*.whl
```

## Troubleshooting

### Import error: `cannot import name 'shodh_memory'`

Make sure you installed with the `python` feature:
```bash
maturin develop --features python --release
```

### Performance lower than expected

1. Ensure you built with `--release` flag
2. Check if embeddings are being cached (see `get_stats()`)
3. Verify you're not in debug mode

### ONNX model not found

The ONNX model should be in `models/minilm-l6/`. Make sure:
```bash
ls models/minilm-l6/model_quantized.onnx
ls models/minilm-l6/tokenizer.json
```

## License

MIT License - see LICENSE file
