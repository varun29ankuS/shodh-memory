// Configuration
let apiUrl = 'http://127.0.0.1:3030';
let apiKey = 'shodh-dev-key-change-in-production';

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    loadConfig();
    checkHealth();
    log('Dashboard loaded', 'info');
});

// ==================== QUICK TEST ====================

async function quickStore() {
    const userId = document.getElementById('quick-user').value.trim();
    const experience = document.getElementById('quick-experience').value.trim();

    if (!userId || !experience) {
        showResult('store-result', '❌ Please fill in both fields', 'danger');
        return;
    }

    try {
        const response = await fetch(`${apiUrl}/api/record`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-API-Key': apiKey
            },
            body: JSON.stringify({
                user_id: userId,
                experience: {
                    content: experience,
                    metadata: { source: 'quick_test' }
                }
            })
        });

        if (response.ok) {
            const data = await response.json();
            showResult('store-result', `✅ Stored! ID: ${data.memory_id}`, 'success');
            log(`Memory stored for ${userId}`, 'success');
        } else {
            const error = await response.text();
            showResult('store-result', `❌ Error: ${error}`, 'danger');
            log(`Store failed: ${error}`, 'error');
        }
    } catch (e) {
        showResult('store-result', `❌ Failed: ${e.message}`, 'danger');
        log(`Store error: ${e.message}`, 'error');
    }
}

async function quickSearch() {
    const userId = document.getElementById('quick-user').value.trim();
    const query = document.getElementById('quick-query').value.trim();
    const mode = document.getElementById('quick-mode').value;

    if (!userId || !query) {
        showResult('quick-results', '❌ Enter user ID and search query', 'danger');
        return;
    }

    const resultsDiv = document.getElementById('quick-results');
    resultsDiv.innerHTML = '<p>🔍 Searching...</p>';

    try {
        const response = await fetch(`${apiUrl}/api/retrieve`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-API-Key': apiKey
            },
            body: JSON.stringify({
                user_id: userId,
                query: query,
                limit: 5,
                mode: mode
            })
        });

        if (response.ok) {
            const data = await response.json();
            displaySearchResults(data.memories || []);
            log(`Found ${data.memories?.length || 0} memories`, 'success');
        } else {
            const error = await response.text();
            resultsDiv.innerHTML = `<p style="color: var(--danger);">❌ ${error}</p>`;
            log(`Search failed: ${error}`, 'error');
        }
    } catch (e) {
        resultsDiv.innerHTML = `<p style="color: var(--danger);">❌ ${e.message}</p>`;
        log(`Search error: ${e.message}`, 'error');
    }
}

function displaySearchResults(memories) {
    const resultsDiv = document.getElementById('quick-results');

    if (!memories || memories.length === 0) {
        resultsDiv.innerHTML = '<p style="color: var(--text-muted);">No memories found. Try storing one first!</p>';
        return;
    }

    let html = '';
    memories.forEach((mem, idx) => {
        const score = mem.score ? (mem.score * 100).toFixed(1) : 'N/A';
        html += `
            <div style="background: var(--bg-secondary); padding: 1rem; border-radius: 8px; margin-bottom: 0.5rem; border-left: 3px solid var(--primary);">
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                    <strong>#${idx + 1}</strong>
                    <span class="badge success">Score: ${score}%</span>
                </div>
                <p style="font-size: 0.875rem; color: var(--text-secondary);">${escapeHtml(mem.content || mem.experience?.content || 'No content')}</p>
                <small style="color: var(--text-muted);">ID: ${mem.memory_id || 'Unknown'}</small>
            </div>
        `;
    });
    resultsDiv.innerHTML = html;
}

// ==================== MEMORY EXPLORER ====================

async function loadAllUsers() {
    const listDiv = document.getElementById('users-list');
    listDiv.innerHTML = '<p>⏳ Loading users...</p>';

    try {
        const response = await fetch(`${apiUrl}/api/users`, {
            headers: { 'X-API-Key': apiKey }
        });

        if (response.ok) {
            const data = await response.json();
            displayUsers(data.users || []);
        } else {
            listDiv.innerHTML = '<p style="color: var(--danger);">Failed to load users</p>';
        }
    } catch (e) {
        listDiv.innerHTML = `<p style="color: var(--danger);">${e.message}</p>`;
    }
}

function displayUsers(users) {
    const listDiv = document.getElementById('users-list');

    if (users.length === 0) {
        listDiv.innerHTML = '<p style="color: var(--text-muted);">No users yet. Create one using Quick Test!</p>';
        return;
    }

    let html = '<div style="display: grid; gap: 0.5rem;">';
    users.forEach(user => {
        html += `
            <div style="background: var(--bg-secondary); padding: 0.75rem; border-radius: 8px; cursor: pointer;" onclick="document.getElementById('explorer-user').value='${user}'; loadUserMemories();">
                <strong>👤 ${user}</strong>
            </div>
        `;
    });
    html += '</div>';
    listDiv.innerHTML = html;
    log(`Loaded ${users.length} users`, 'info');
}

async function loadUserMemories() {
    const userId = document.getElementById('explorer-user').value.trim();
    const listDiv = document.getElementById('memories-list');

    if (!userId) {
        listDiv.innerHTML = '<p style="color: var(--warning);">Enter a user ID first</p>';
        return;
    }

    listDiv.innerHTML = '<p>⏳ Loading memories...</p>';

    try {
        const response = await fetch(`${apiUrl}/api/memories`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-API-Key': apiKey
            },
            body: JSON.stringify({ user_id: userId })
        });

        if (response.ok) {
            const data = await response.json();
            displayMemoriesList(data.memories || []);
        } else {
            listDiv.innerHTML = '<p style="color: var(--danger);">Failed to load memories</p>';
        }
    } catch (e) {
        listDiv.innerHTML = `<p style="color: var(--danger);">${e.message}</p>`;
    }
}

function displayMemoriesList(memories) {
    const listDiv = document.getElementById('memories-list');

    if (memories.length === 0) {
        listDiv.innerHTML = '<p style="color: var(--text-muted);">No memories stored yet.</p>';
        return;
    }

    let html = '<div style="display: grid; gap: 1rem;">';
    memories.forEach((mem, idx) => {
        const time = mem.timestamp ? new Date(mem.timestamp).toLocaleString() : 'Unknown';
        const importance = mem.experience?.importance || 0;
        html += `
            <div style="background: var(--bg-secondary); padding: 1rem; border-radius: 8px; border-left: 3px solid var(--primary);">
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                    <strong>Memory #${idx + 1}</strong>
                    <span class="badge info">Importance: ${importance.toFixed(2)}</span>
                </div>
                <p style="margin: 0.5rem 0; font-size: 0.9rem;">${escapeHtml(mem.experience?.content || 'No content')}</p>
                <div style="display: flex; justify-content: between; gap: 1rem; margin-top: 0.5rem;">
                    <small style="color: var(--text-muted);">🕐 ${time}</small>
                    <small style="color: var(--text-muted);">ID: ${mem.memory_id}</small>
                </div>
            </div>
        `;
    });
    html += '</div>';
    listDiv.innerHTML = html;
    log(`Loaded ${memories.length} memories`, 'info');
}

function clearExplorer() {
    document.getElementById('users-list').innerHTML = '<p style="color: var(--text-muted);">Click "Load Users" to see active users</p>';
    document.getElementById('memories-list').innerHTML = '<p style="color: var(--text-muted);">Enter a user ID and click "Load Memories"</p>';
}

// ==================== API TESTING ====================

function saveConfig() {
    apiUrl = document.getElementById('api-url').value.trim();
    apiKey = document.getElementById('api-key').value.trim();
    localStorage.setItem('apiUrl', apiUrl);
    localStorage.setItem('apiKey', apiKey);
    log('Configuration saved', 'success');
}

function loadConfig() {
    const savedUrl = localStorage.getItem('apiUrl');
    const savedKey = localStorage.getItem('apiKey');
    if (savedUrl) {
        apiUrl = savedUrl;
        document.getElementById('api-url').value = savedUrl;
    }
    if (savedKey) {
        apiKey = savedKey;
        document.getElementById('api-key').value = savedKey;
    }
}

async function testConnection() {
    try {
        const response = await fetch(`${apiUrl}/health`);
        if (response.ok) {
            log('✅ Connection successful', 'success');
        } else {
            log('❌ Connection failed', 'error');
        }
    } catch (e) {
        log(`❌ Connection error: ${e.message}`, 'error');
    }
}

async function testRecordSpeed() {
    const testDiv = document.getElementById('test-results');
    testDiv.innerHTML = '<p>⏳ Testing record speed...</p>';

    const start = performance.now();
    try {
        await fetch(`${apiUrl}/api/record`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-API-Key': apiKey
            },
            body: JSON.stringify({
                user_id: 'perf_test',
                experience: {
                    content: 'Performance test memory'
                }
            })
        });
        const time = (performance.now() - start).toFixed(2);
        testDiv.innerHTML = `<p style="color: var(--success);">✅ Record time: ${time}ms</p>`;
        log(`Record speed: ${time}ms`, 'success');
    } catch (e) {
        testDiv.innerHTML = `<p style="color: var(--danger);">❌ ${e.message}</p>`;
    }
}

async function testSearchSpeed() {
    const testDiv = document.getElementById('test-results');
    testDiv.innerHTML = '<p>⏳ Testing search speed...</p>';

    const start = performance.now();
    try {
        await fetch(`${apiUrl}/api/retrieve`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-API-Key': apiKey
            },
            body: JSON.stringify({
                user_id: 'perf_test',
                query: 'test',
                limit: 10
            })
        });
        const time = (performance.now() - start).toFixed(2);
        testDiv.innerHTML = `<p style="color: var(--success);">✅ Search time: ${time}ms</p>`;
        log(`Search speed: ${time}ms`, 'success');
    } catch (e) {
        testDiv.innerHTML = `<p style="color: var(--danger);">❌ ${e.message}</p>`;
    }
}

async function testBulkInsert() {
    const testDiv = document.getElementById('test-results');
    testDiv.innerHTML = '<p>⏳ Testing bulk insert (10 records)...</p>';

    const start = performance.now();
    const promises = [];
    for (let i = 0; i < 10; i++) {
        promises.push(
            fetch(`${apiUrl}/api/record`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-API-Key': apiKey
                },
                body: JSON.stringify({
                    user_id: 'bulk_test',
                    experience: {
                        content: `Bulk test memory ${i}`
                    }
                })
            })
        );
    }

    try {
        await Promise.all(promises);
        const time = (performance.now() - start).toFixed(2);
        const perRecord = (time / 10).toFixed(2);
        testDiv.innerHTML = `<p style="color: var(--success);">✅ Inserted 10 records in ${time}ms (${perRecord}ms/record)</p>`;
        log(`Bulk insert: ${time}ms total, ${perRecord}ms per record`, 'success');
    } catch (e) {
        testDiv.innerHTML = `<p style="color: var(--danger);">❌ ${e.message}</p>`;
    }
}

// ==================== HEALTH CHECK ====================

async function checkHealth() {
    const statusDot = document.getElementById('health-status');
    const statusText = document.getElementById('health-text');

    try {
        const response = await fetch(`${apiUrl}/health`);
        if (response.ok) {
            statusDot.style.background = 'var(--success)';
            statusText.textContent = 'Connected';
        } else {
            statusDot.style.background = 'var(--danger)';
            statusText.textContent = 'Error';
        }
    } catch (e) {
        statusDot.style.background = 'var(--danger)';
        statusText.textContent = 'Offline';
    }
}

// ==================== UTILITIES ====================

function showResult(elementId, message, type = 'info') {
    const element = document.getElementById(elementId);
    element.innerHTML = `<p style="color: var(--${type === 'success' ? 'success' : type === 'danger' ? 'danger' : 'info'});">${message}</p>`;
}

function log(message, level = 'info') {
    const console = document.getElementById('console');
    const time = new Date().toLocaleTimeString();
    const color = level === 'success' ? '#10b981' : level === 'error' ? '#ef4444' : '#06b6d4';
    const line = document.createElement('div');
    line.style.color = color;
    line.style.padding = '0.25rem 0';
    line.textContent = `[${time}] ${message}`;
    console.appendChild(line);
    console.scrollTop = console.scrollHeight;
}

function clearConsole() {
    document.getElementById('console').innerHTML = '';
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Auto-check health every 10 seconds
setInterval(checkHealth, 10000);

// ==================== API DOCUMENTATION ====================

function showApiCategory(category) {
    // Hide all categories
    const categories = document.querySelectorAll('.api-category');
    categories.forEach(cat => cat.style.display = 'none');

    // Show selected category
    const selected = document.getElementById(`api-${category}`);
    if (selected) {
        selected.style.display = 'block';
    }

    // Update active button
    const buttons = document.querySelectorAll('.api-nav-btn');
    buttons.forEach(btn => btn.classList.remove('active'));
    event.target.classList.add('active');
}

function showCodeTab(button, tabType) {
    // Get the parent code example
    const codeExample = button.closest('.code-example');

    // Hide all code contents in this example
    const contents = codeExample.querySelectorAll('.code-content');
    contents.forEach(content => content.style.display = 'none');

    // Show selected content
    const selected = codeExample.querySelector(`.code-content.${tabType}`);
    if (selected) {
        selected.style.display = 'block';
    }

    // Update active tab button
    const tabs = codeExample.querySelectorAll('.code-tab');
    tabs.forEach(tab => tab.classList.remove('active'));
    button.classList.add('active');
}
