# Shodh-Memory Dashboard

Enterprise-grade web dashboard for visualizing and testing the Shodh-Memory system.

## Features

### 1. Overview
- Real-time system health monitoring
- Active user count and statistics
- Console for debugging and activity tracking
- Quick actions for common operations

### 2. User Onboarding
- Create new users (robots, drones, devices)
- Initialize user memory systems
- Add descriptive metadata
- View initialization results

### 3. Memory Management
- Record new experiences/memories
- Set importance levels (0.0 to 1.0)
- Add contextual metadata (JSON)
- View user statistics:
  - Working memory count
  - Session memory count
  - Long-term memory count
  - Graph entities and relationships

### 4. Search & Retrieval
- Semantic search using MiniLM-L6-v2 embeddings
- Multiple search modes:
  - **Hybrid**: Combines all retrieval methods (default)
  - **Semantic Similarity**: Vector-based semantic search
  - **Temporal**: Recent memories first
  - **Causal**: Cause-and-effect relationships
- Real-time search performance metrics
- Detailed result display with scores

### 5. Graph Visualization
- D3.js-powered entity graph visualization
- Color-coded entity types:
  - Location (blue)
  - Object (green)
  - Person (orange)
  - Task (red)
  - Event (purple)
- Interactive graph statistics
- Add new entities and relationships
- Force-directed graph layout

### 6. API Testing
- Connection testing
- Configurable API URL and key
- Performance benchmarks:
  - **Record Speed**: Single record insertion performance
  - **Search Speed**: Query performance testing
  - **Bulk Insert**: Concurrent insert throughput
- Real-time test results and metrics

## Quick Start

### 1. Start the Server

```bash
cd shodh-memory
cargo run --release
```

Server starts on: http://127.0.0.1:3030

### 2. Access Dashboard

Open your browser to: http://127.0.0.1:3030

The dashboard will automatically redirect from `/` to `/static/index.html`

### 3. Configure API (if needed)

Go to the **API Testing** tab and update:
- API URL (default: http://127.0.0.1:3030)
- API Key (default: shodh-dev-key-change-in-production)

Click "Test Connection" to verify.

## Usage Examples

### Creating a New User

1. Navigate to **User Onboarding** tab
2. Enter User ID: `drone_01`
3. Add description: `Primary surveillance drone for warehouse sector A`
4. Click "Initialize User Memory"
5. View initialization results

### Recording a Memory

1. Navigate to **Memory Management** tab
2. Enter User ID: `drone_01`
3. Add experience content:
   ```
   Detected obstacle at coordinates (12.5, 45.3).
   Adjusted flight path to maintain 2m clearance.
   ```
4. Set context JSON:
   ```json
   {
     "location": "warehouse_a",
     "task": "surveillance",
     "obstacle_type": "forklift",
     "clearance_m": 2.0
   }
   ```
5. Set importance: `0.8`
6. Click "Record Memory"

### Searching Memories

1. Navigate to **Search & Retrieval** tab
2. Enter User ID: `drone_01`
3. Enter query: `obstacle avoidance in warehouse`
4. Select search mode: `Hybrid`
5. Set max results: `10`
6. Click "Search Memories"
7. View results with relevance scores

### Visualizing Entity Graph

1. Navigate to **Graph Visualization** tab
2. Enter User ID: `drone_01`
3. Click "Load Graph"
4. Interact with the force-directed graph
5. Add new entities using the form below

### Running Performance Benchmarks

1. Navigate to **API Testing** tab
2. Click benchmark buttons:
   - **Record Speed Test**: Measures single record insertion performance
   - **Search Speed Test**: Measures semantic search latency
   - **Bulk Insert Test**: Measures concurrent throughput (100 records)
3. View detailed results including:
   - Total time
   - Average time per operation
   - Throughput (ops/sec)

## Production Deployment

### Security Considerations

1. **Change API Key**: Update `shodh_config.json`:
   ```json
   {
     "api": {
       "default_api_key": "your-secure-key-here"
     }
   }
   ```

2. **Enable HTTPS**: Use reverse proxy (nginx, Caddy) for TLS termination

3. **Rate Limiting**: Already configured (2 req/sec, burst of 10)

4. **CORS**: Configure allowed origins in production:
   ```rust
   let cors = CorsLayer::new()
       .allow_origin("https://yourdomain.com".parse::<HeaderValue>().unwrap())
       .allow_methods([Method::GET, Method::POST, Method::DELETE])
       .allow_headers(Any);
   ```

### Performance Tuning

Dashboard is optimized for:
- **Low Latency**: Sub-millisecond semantic search
- **High Throughput**: Concurrent API requests
- **Memory Efficiency**: Suitable for 8-16GB RAM devices

Expected performance (on target hardware):
- Record insertion: < 5ms
- Semantic search: < 10ms (with ONNX)
- Bulk insert: > 200 records/sec

## Architecture

### Frontend Stack
- **HTML5**: Semantic structure
- **CSS3**: Modern dark theme with CSS variables
- **Vanilla JavaScript**: No framework overhead
- **D3.js v7**: Graph visualization

### Backend Integration
- **Axum**: High-performance async web framework
- **tower-http**: Static file serving and CORS
- **tower-governor**: IP-based rate limiting
- **RocksDB**: Persistent storage
- **ONNX Runtime**: ML inference for embeddings

### API Endpoints

All endpoints require `X-API-Key` header.

**Core**:
- `GET /health` - Health check
- `POST /api/record` - Record memory
- `POST /api/retrieve` - Search memories

**User Management**:
- `GET /api/users` - List all users
- `GET /api/users/{user_id}/stats` - User statistics
- `DELETE /api/users/{user_id}` - Delete user (GDPR)

**Memory CRUD**:
- `GET /api/memory/{memory_id}` - Get single memory
- `PUT /api/memory/{memory_id}` - Update memory
- `DELETE /api/memory/{memory_id}` - Delete memory
- `POST /api/memories` - Get all memories
- `POST /api/memories/history` - Get memory history

**Graph Memory**:
- `GET /api/graph/{user_id}/stats` - Graph statistics
- `POST /api/graph/entity/add` - Add entity
- `POST /api/graph/entities/all` - Get all entities
- `POST /api/graph/relationship/add` - Add relationship
- `POST /api/graph/traverse` - Traverse graph

**Advanced**:
- `POST /api/search/advanced` - Advanced search
- `POST /api/search/multimodal` - Multimodal search
- `POST /api/forget/age` - Forget by age
- `POST /api/forget/importance` - Forget by importance
- `POST /api/forget/pattern` - Forget by pattern

## Troubleshooting

### Dashboard Not Loading

1. Check server is running: `curl http://127.0.0.1:3030/health`
2. Check static files exist: `ls static/`
3. Check browser console for errors
4. Verify API key configuration

### Connection Failed

1. Verify server URL: `http://127.0.0.1:3030` (no trailing slash)
2. Check API key matches `shodh_config.json`
3. Check CORS configuration
4. Check network/firewall rules

### Search Returns No Results

1. Verify user has memories: Check **User Statistics**
2. Check ONNX model is loaded: Look for server logs
3. Try different search modes
4. Verify vector index is built

### Graph Not Rendering

1. Verify user has entities: Check **Graph Stats**
2. Check browser console for D3.js errors
3. Ensure container has dimensions
4. Add entities using the form first

## Development

### Adding New Features

1. **HTML**: Edit `static/index.html` - Add new tabs/sections
2. **CSS**: Edit `static/dashboard.css` - Style components
3. **JavaScript**: Edit `static/dashboard.js` - Add functionality
4. **Backend**: Edit `src/main.rs` - Add API endpoints

### Custom Styling

Dashboard uses CSS variables for theming. Edit `static/dashboard.css`:

```css
:root {
    --primary: #2563eb;        /* Primary color */
    --success: #10b981;        /* Success color */
    --danger: #ef4444;         /* Error color */
    --bg: #0f172a;             /* Background */
    --text: #f1f5f9;           /* Text color */
}
```

### Testing Locally

```bash
# Terminal 1: Run server
cargo run --release

# Terminal 2: Test API
curl -H "X-API-Key: shodh-dev-key-change-in-production" \
     http://127.0.0.1:3030/health

# Browser: Open dashboard
firefox http://127.0.0.1:3030
```

## License

MIT License - See project root for details

## Support

For issues, questions, or feature requests, please file an issue on the GitHub repository.
