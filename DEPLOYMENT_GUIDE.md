# Shodh-Memory Deployment Guide for Robotics & Drones

## Supported Platforms

| Platform | Architecture | RAM Required | Storage |
|----------|--------------|--------------|---------|
| Raspberry Pi 4/5 | aarch64 | 2GB+ | 500MB+ |
| NVIDIA Jetson Nano/Xavier | aarch64 | 4GB+ | 500MB+ |
| Intel NUC | x86_64 | 4GB+ | 500MB+ |
| BeagleBone AI | aarch64 | 1GB+ | 500MB+ |
| Generic Linux | x86_64/aarch64 | 2GB+ | 500MB+ |

---

## Option 1: Native Deployment (Recommended for Production)

### Step 1: Cross-compile for Target Platform

**On your development machine (Windows/Linux/Mac):**

```bash
# Install Rust cross-compilation toolchain
rustup target add aarch64-unknown-linux-gnu  # For Raspberry Pi/Jetson
rustup target add armv7-unknown-linux-gnueabihf  # For 32-bit ARM

# Install cross-compilation linker
# On Ubuntu/Debian:
sudo apt install gcc-aarch64-linux-gnu

# Build release binary
cargo build --release --target aarch64-unknown-linux-gnu
```

### Step 2: Prepare ONNX Runtime for Target

Download ONNX Runtime for your target architecture:
- **aarch64**: https://github.com/microsoft/onnxruntime/releases (onnxruntime-linux-aarch64-*.tgz)
- **x86_64**: https://github.com/microsoft/onnxruntime/releases (onnxruntime-linux-x64-*.tgz)

```bash
# Extract on target device
tar -xzf onnxruntime-linux-aarch64-1.22.0.tgz
export ORT_DYLIB_PATH=/path/to/onnxruntime-linux-aarch64-1.22.0/lib/libonnxruntime.so
```

### Step 3: Deploy to Robot/Drone

```bash
# Copy binary and dependencies to target
scp target/aarch64-unknown-linux-gnu/release/shodh-memory-server robot@192.168.1.100:/home/robot/
scp -r models/ robot@192.168.1.100:/home/robot/  # MiniLM-L6 ONNX model

# Copy ONNX runtime library
scp onnxruntime-linux-aarch64-1.22.0/lib/libonnxruntime.so robot@192.168.1.100:/home/robot/lib/
```

### Step 4: Configure Environment on Robot

Create `/home/robot/shodh-memory.env`:
```bash
# Required
ORT_DYLIB_PATH=/home/robot/lib/libonnxruntime.so

# Server Configuration
SHODH_PORT=3030
SHODH_MEMORY_PATH=/home/robot/memory_data
SHODH_ENV=production

# Security (REQUIRED in production)
SHODH_API_KEYS=your-secure-api-key-here

# Performance tuning for edge devices
SHODH_MAX_USERS=10          # Limit for memory-constrained devices
SHODH_RATE_LIMIT=20         # Requests per second
SHODH_MAX_CONCURRENT=50     # Max concurrent requests

# Logging
RUST_LOG=info
```

### Step 5: Create Systemd Service

Create `/etc/systemd/system/shodh-memory.service`:
```ini
[Unit]
Description=Shodh-Memory AI Memory Server
After=network.target

[Service]
Type=simple
User=robot
EnvironmentFile=/home/robot/shodh-memory.env
ExecStart=/home/robot/shodh-memory-server
Restart=always
RestartSec=5

# Resource limits for embedded systems
MemoryMax=512M
CPUQuota=80%

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable shodh-memory
sudo systemctl start shodh-memory

# Check status
sudo systemctl status shodh-memory
```

---

## Option 2: Docker Deployment (For Jetson/NUC with Docker support)

### Dockerfile for aarch64:
```dockerfile
FROM rust:1.75-slim-bookworm AS builder
WORKDIR /app
COPY . .
RUN cargo build --release

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y libssl3 ca-certificates && rm -rf /var/lib/apt/lists/*

# Copy binary
COPY --from=builder /app/target/release/shodh-memory-server /usr/local/bin/

# Copy ONNX runtime
COPY libs/libonnxruntime.so /usr/local/lib/
ENV ORT_DYLIB_PATH=/usr/local/lib/libonnxruntime.so
ENV LD_LIBRARY_PATH=/usr/local/lib

# Copy model files
COPY models/ /app/models/

EXPOSE 3030
CMD ["shodh-memory-server"]
```

```bash
# Build for ARM64
docker buildx build --platform linux/arm64 -t shodh-memory:arm64 .

# Run on Jetson
docker run -d \
  --name shodh-memory \
  -p 3030:3030 \
  -v /data/memory:/app/memory_data \
  -e SHODH_API_KEYS=your-key \
  -e SHODH_ENV=production \
  shodh-memory:arm64
```

---

## Option 3: Python Integration (For ROS/ROS2 robots)

### Install Python Client
```bash
pip install requests
```

### Python Client Example:
```python
import requests
import json

class ShodhMemory:
    def __init__(self, host="localhost", port=3030, api_key="your-key"):
        self.base_url = f"http://{host}:{port}"
        self.headers = {"X-API-Key": api_key, "Content-Type": "application/json"}

    def record(self, user_id: str, content: str, robot_id: str = None,
               mission_id: str = None, geo_location: list = None):
        """Record a memory/experience"""
        payload = {
            "user_id": user_id,
            "experience": {
                "content": content,
                "experience_type": "Observation",
                "robot_id": robot_id,
                "mission_id": mission_id,
                "geo_location": geo_location
            }
        }
        resp = requests.post(f"{self.base_url}/record",
                            headers=self.headers, json=payload)
        return resp.json()

    def retrieve(self, user_id: str, query: str, max_results: int = 10):
        """Retrieve relevant memories"""
        payload = {
            "user_id": user_id,
            "query_text": query,
            "max_results": max_results
        }
        resp = requests.post(f"{self.base_url}/retrieve",
                            headers=self.headers, json=payload)
        return resp.json()

# Usage in ROS node
memory = ShodhMemory(host="localhost", port=3030, api_key="robot-key-123")

# Record obstacle detection
memory.record(
    user_id="robot_arm_01",
    content="Detected obstacle: large rock at position (10.5, 25.3, 0.0)",
    robot_id="robot_arm_01",
    mission_id="patrol_mission_42",
    geo_location=[37.7749, -122.4194, 0.0]
)

# Query for similar obstacles
results = memory.retrieve(
    user_id="robot_arm_01",
    query="obstacles detected during patrol",
    max_results=5
)
```

---

## ROS2 Integration Example

```python
# ros2_memory_node.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import NavSatFix
from shodh_memory_client import ShodhMemory

class MemoryNode(Node):
    def __init__(self):
        super().__init__('shodh_memory_node')

        # Initialize memory client
        self.memory = ShodhMemory(
            host=self.declare_parameter('memory_host', 'localhost').value,
            port=self.declare_parameter('memory_port', 3030).value,
            api_key=self.declare_parameter('api_key', 'robot-key').value
        )

        self.robot_id = self.declare_parameter('robot_id', 'drone_01').value
        self.mission_id = None
        self.current_position = None

        # Subscribe to events
        self.create_subscription(String, '/mission/start', self.on_mission_start, 10)
        self.create_subscription(String, '/detection/obstacle', self.on_obstacle, 10)
        self.create_subscription(NavSatFix, '/gps/fix', self.on_gps, 10)

        # Service for memory queries
        self.create_service(QueryMemory, '/memory/query', self.handle_query)

    def on_mission_start(self, msg):
        self.mission_id = msg.data
        self.memory.record(
            user_id=self.robot_id,
            content=f"Mission started: {self.mission_id}",
            robot_id=self.robot_id,
            mission_id=self.mission_id
        )

    def on_obstacle(self, msg):
        self.memory.record(
            user_id=self.robot_id,
            content=f"Obstacle detected: {msg.data}",
            robot_id=self.robot_id,
            mission_id=self.mission_id,
            geo_location=self.current_position
        )

    def on_gps(self, msg):
        self.current_position = [msg.latitude, msg.longitude, msg.altitude]

    def handle_query(self, request, response):
        results = self.memory.retrieve(
            user_id=self.robot_id,
            query=request.query,
            max_results=request.max_results
        )
        response.memories = results['memories']
        return response

def main():
    rclpy.init()
    node = MemoryNode()
    rclpy.spin(node)
    rclpy.shutdown()
```

---

## Performance Tuning for Edge Devices

### Raspberry Pi 4 (2GB RAM)
```bash
SHODH_MAX_USERS=5
SHODH_MAX_CONCURRENT=20
SHODH_RATE_LIMIT=10
```

### NVIDIA Jetson Nano (4GB RAM)
```bash
SHODH_MAX_USERS=20
SHODH_MAX_CONCURRENT=50
SHODH_RATE_LIMIT=30
```

### NVIDIA Jetson Xavier (8GB+ RAM)
```bash
SHODH_MAX_USERS=100
SHODH_MAX_CONCURRENT=100
SHODH_RATE_LIMIT=100
```

---

## Monitoring & Health Checks

### Health Endpoint
```bash
curl http://localhost:3030/health
# Returns: {"status": "healthy", "version": "0.1.0"}
```

### Prometheus Metrics
```bash
curl http://localhost:3030/metrics
# Returns Prometheus-format metrics for:
# - memory_store_duration_seconds
# - memory_retrieve_duration_seconds
# - active_users_count
# - memory_count_total
```

---

## Troubleshooting

### Issue: "ONNX Runtime not found"
```bash
# Verify ORT_DYLIB_PATH is set correctly
echo $ORT_DYLIB_PATH
ls -la $ORT_DYLIB_PATH

# Check library can be loaded
ldd /home/robot/shodh-memory-server
```

### Issue: "Out of memory on Raspberry Pi"
```bash
# Reduce memory limits
SHODH_MAX_USERS=3
SHODH_MAX_CONCURRENT=10

# Add swap space
sudo fallocate -l 1G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### Issue: "Slow embedding generation"
```bash
# Use smaller model (if available)
# Or pre-compute embeddings on powerful machine and send via API

# Check CPU governor
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
# Set to performance mode:
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

---

## Security Checklist for Production

- [ ] Set strong `SHODH_API_KEYS` (min 32 characters)
- [ ] Set `SHODH_ENV=production`
- [ ] Configure `SHODH_CORS_ORIGINS` if using web interface
- [ ] Use HTTPS with reverse proxy (nginx/caddy)
- [ ] Limit network access (firewall rules)
- [ ] Regular backups of `/memory_data` directory
- [ ] Monitor disk space (memories grow over time)
