const API_URL = 'http://localhost:8000';

// DOM Elements
const chatMessages = document.getElementById('chat-messages');
const queryForm = document.getElementById('query-form');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');
const apiStatus = document.getElementById('api-status');
const apiStatusText = apiStatus.querySelector('.text');
const fileUpload = document.getElementById('file-upload');
const uploadTrigger = document.getElementById('upload-trigger');
const deleteDocBtn = document.getElementById('delete-doc-btn');
const clearChatBtn = document.getElementById('clear-chat-btn');
const docNameDisplay = document.getElementById('current-doc-name');
const exampleQueriesContainer = document.getElementById('example-queries');

// State
let isProcessing = false;

// Initialize
async function init() {
    checkApiStatus();
    loadExampleQueries();
    checkDocStatus();
    
    // Interval status checks
    setInterval(checkApiStatus, 30000);
}

// Check if API is online
async function checkApiStatus() {
    try {
        const response = await fetch(`${API_URL}/health`);
        if (response.ok) {
            apiStatus.classList.add('online');
            apiStatusText.textContent = 'API: Operational';
        } else {
            setOffline();
        }
    } catch (error) {
        setOffline();
    }
}

function setOffline() {
    apiStatus.classList.remove('online');
    apiStatusText.textContent = 'API: Offline';
}

// Load example queries from API
async function loadExampleQueries() {
    try {
        const response = await fetch(`${API_URL}/examples`);
        if (response.ok) {
            const data = await response.json();
            // Combine some queries from different categories
            const queries = [
                ...data.platform_general.slice(0, 1),
                "Show AAPL performance chart",
                "What's the market sentiment for TSLA?"
            ];
            
            exampleQueriesContainer.innerHTML = '';
            queries.forEach(query => {
                const tag = document.createElement('span');
                tag.className = 'example-tag';
                tag.textContent = query;
                tag.onclick = () => {
                    userInput.value = query;
                    userInput.focus();
                };
                exampleQueriesContainer.appendChild(tag);
            });
        }
    } catch (error) {
        console.error('Error loading examples:', error);
    }
}

// Check current document status
async function checkDocStatus() {
    try {
        const response = await fetch(`${API_URL}/document/status`);
        if (response.ok) {
            const data = await response.json();
            if (data.has_document) {
                docNameDisplay.textContent = `Active: ${data.filename}`;
                deleteDocBtn.style.display = 'flex';
            } else {
                docNameDisplay.textContent = 'No knowledge base active';
                deleteDocBtn.style.display = 'none';
            }
        }
    } catch (error) {
        console.error('Error checking doc status:', error);
    }
}

// Chat functions
function addMessage(content, isBot = true, toolUsed = null, chartData = null, sentimentScore = null, sentimentLabel = null) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isBot ? 'bot-message' : 'user-message'}`;
    
    let html = content;
    if (toolUsed && toolUsed !== 'None') {
        html += `<div style="font-size: 0.75rem; color: var(--primary); margin-top: 0.5rem; opacity: 0.8; display: flex; align-items: center; gap: 0.4rem;">
            <i class="fas fa-microchip"></i> <span>Source: ${toolUsed}</span>
        </div>`;
    }
    
    if (isBot) {
        html += `<button class="copy-btn" onclick="copyToClipboard(this)" title="Copy Response">
            <i class="far fa-copy"></i>
        </button>`;
    }
    
    // Add Chart Canvas if data exists
    let canvasId = null;
    if (chartData && chartData.prices && chartData.prices.length > 0) {
        canvasId = `chart-${Date.now()}-${Math.floor(Math.random() * 1000)}`;
        html += `
        <div class="chart-container" style="margin-top: 15px; background: rgba(0,0,0,0.2); padding: 10px; border-radius: 8px; border: 1px solid var(--border-color);">
            <canvas id="${canvasId}" style="width: 100%; height: 200px;"></canvas>
        </div>`;
    }
    
    // Add Sentiment Gauge if data exists
    let gaugeId = null;
    if (sentimentScore !== null && sentimentScore !== undefined) {
        gaugeId = `gauge-${Date.now()}-${Math.floor(Math.random() * 1000)}`;
        html += `
        <div class="sentiment-container" style="margin-top: 15px; background: rgba(0,0,0,0.2); padding: 15px; border-radius: 12px; border: 1px solid var(--border-color); text-align: center;">
            <div style="font-weight: 600; margin-bottom: 10px; color: var(--text-muted); font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px;">AI Sentiment Analysis</div>
            <div style="width: 150px; height: 80px; margin: 0 auto;">
                <canvas id="${gaugeId}"></canvas>
            </div>
            <div style="margin-top: 5px; font-weight: 700; font-size: 1.1rem; color: ${sentimentScore > 0 ? '#00ffcc' : (sentimentScore < 0 ? '#ff4d4d' : '#888')}; animation: fadeIn 0.5s ease;">
                ${sentimentLabel || (sentimentScore > 0 ? 'Bullish' : (sentimentScore < 0 ? 'Bearish' : 'Neutral'))}
            </div>
        </div>`;
    }
    
    // Convert markdown-like bold to html
    html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    
    messageDiv.innerHTML = html;
    chatMessages.appendChild(messageDiv);
    
    // Render Chart or Gauge if needed
    if (canvasId) {
        setTimeout(() => renderChart(canvasId, chartData.labels, chartData.prices), 50);
    }
    if (gaugeId) {
        setTimeout(() => renderSentimentGauge(gaugeId, sentimentScore), 50);
    }
    
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function showTypingIndicator() {
    const indicator = document.createElement('div');
    indicator.className = 'typing-indicator bot-message';
    indicator.id = 'typing-indicator';
    indicator.innerHTML = `
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
    `;
    chatMessages.appendChild(indicator);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function removeTypingIndicator() {
    const indicator = document.getElementById('typing-indicator');
    if (indicator) indicator.remove();
}

// Form submission
queryForm.onsubmit = async (e) => {
    e.preventDefault();
    const query = userInput.value.trim();
    
    if (!query || isProcessing) return;
    
    userInput.value = '';
    addMessage(query, false);
    isProcessing = true;
    showTypingIndicator();
    
    try {
        const response = await fetch(`${API_URL}/query`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query })
        });
        
        removeTypingIndicator();
        
        if (response.ok) {
            const data = await response.json();
            console.log("DEBUG: Query Response Data:", data);
            addMessage(data.response, true, data.tool_used, data.chart_data, data.sentiment_score, data.sentiment_label);
        } else {
            addMessage('Sorry, I encountered an error processing that request.', true);
        }
    } catch (error) {
        removeTypingIndicator();
        addMessage('Connection error. Please make sure the backend server is running.', true);
    } finally {
        isProcessing = false;
    }
};

// Action button handlers
clearChatBtn.onclick = () => {
    chatMessages.innerHTML = `
        <div class="message bot-message">
            Terminal cleared. How can I help you with your financial research next?
        </div>
    `;
};

deleteDocBtn.onclick = async () => {
    if (!confirm('Are you sure you want to delete the current knowledge base? This cannot be undone.')) {
        return;
    }
    
    try {
        const response = await fetch(`${API_URL}/document`, {
            method: 'DELETE'
        });
        
        if (response.ok) {
            addMessage('The knowledge base has been successfully cleared.', true);
            checkDocStatus();
        } else {
            alert('Error deleting document.');
        }
    } catch (error) {
        alert('Could not connect to server for deletion.');
    }
};

// File upload handling
uploadTrigger.onclick = () => fileUpload.click();

fileUpload.onchange = async () => {
    if (!fileUpload.files.length) return;
    
    const file = fileUpload.files[0];
    if (!file.name.endsWith('.pdf')) {
        alert('Please upload a PDF document.');
        return;
    }
    
    const formData = new FormData();
    formData.append('file', file);
    
    docNameDisplay.textContent = 'Uploading...';
    
    try {
        const response = await fetch(`${API_URL}/upload`, {
            method: 'POST',
            body: formData
        });
        
        if (response.ok) {
            const data = await response.json();
            checkDocStatus();
            addMessage(`Document **${data.filename}** has been successfully uploaded and processed. I can now answer questions based on its content!`, true);
        } else {
            checkDocStatus();
            alert('Error uploading document.');
        }
    } catch (error) {
        checkDocStatus();
        alert('Could not connect to server for upload.');
    }
};

// Utility Functions
function renderChart(canvasId, labels, prices) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    
    // Create gradient
    const gradient = ctx.createLinearGradient(0, 0, 0, 200);
    gradient.addColorStop(0, 'rgba(0, 255, 204, 0.3)');
    gradient.addColorStop(1, 'rgba(0, 255, 204, 0)');

    new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Price',
                data: prices,
                borderColor: '#00ffcc',
                borderWidth: 2,
                pointRadius: 0,
                pointHoverRadius: 5,
                pointBackgroundColor: '#00ffcc',
                backgroundColor: gradient,
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    backgroundColor: 'rgba(10, 10, 15, 0.9)',
                    titleColor: '#00ffcc',
                    borderColor: 'rgba(255, 255, 255, 0.1)',
                    borderWidth: 1
                }
            },
            interaction: {
                mode: 'nearest',
                axis: 'x',
                intersect: false
            },
            scales: {
                x: {
                    display: true,
                    grid: { display: false },
                    ticks: { color: '#888', maxRotation: 0, autoSkip: true, maxTicksLimit: 5 }
                },
                y: {
                    display: true,
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#888', callback: (value) => '$' + value }
                }
            }
        }
    });
}

function renderSentimentGauge(canvasId, score) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    
    // Map -1..1 to 0..100 for display
    const normalizedValue = ((score + 1) / 2) * 100;
    const color = score > 0 ? '#00ffcc' : (score < 0 ? '#ff4d4d' : '#888');

    new Chart(ctx, {
        type: 'doughnut',
        data: {
            datasets: [{
                data: [normalizedValue, 100 - normalizedValue],
                backgroundColor: [color, 'rgba(255, 255, 255, 0.05)'],
                borderWidth: 0,
                circumference: 180,
                rotation: 270,
                borderRadius: 5
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: '85%',
            plugins: {
                tooltip: { enabled: false },
                legend: { display: false }
            }
        }
    });
}

function copyToClipboard(btn) {
    const messageContent = btn.parentElement.innerText.replace('Source:', '').split('\n')[0].trim();
    navigator.clipboard.writeText(messageContent).then(() => {
        const icon = btn.querySelector('i');
        icon.className = 'fas fa-check';
        setTimeout(() => {
            icon.className = 'far fa-copy';
        }, 2000);
    });
}

// Start Button scroll
document.getElementById('start-btn').onclick = () => {
    document.getElementById('chat').scrollIntoView({ behavior: 'smooth' });
};

// Start
init();
