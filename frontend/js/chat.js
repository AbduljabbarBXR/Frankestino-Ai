/**
 * Frankenstino AI - Chat Interface Module
 * Handles chat messages, user input, and response display
 */

class ContextManager {
    constructor() {
        this.maxContextMessages = 10;  // Configurable message limit
        this.maxContextTokens = 2000;  // Token limit (rough estimate)
        this.contextStrategy = 'recent'; // 'recent', 'summary', 'selective'
    }

    buildConversationContext(messageHistory) {
        if (!messageHistory || messageHistory.length === 0) {
            return [];
        }

        let context = [...messageHistory];

        // Apply context window management
        if (context.length > this.maxContextMessages) {
            context = this.compressContext(context);
        }

        // Convert to API format
        return context.flatMap(item => [
            { role: 'user', content: item.user },
            { role: 'assistant', content: item.assistant }
        ]);
    }

    compressContext(fullHistory) {
        // Keep most recent messages, summarize or drop older ones
        const keepRecent = this.maxContextMessages;
        const recentMessages = fullHistory.slice(-keepRecent);

        // For now, just keep recent messages
        // TODO: Implement summarization for older messages
        return recentMessages;
    }

    estimateTokenCount(messageHistory) {
        // Rough token estimation: ~4 characters per token
        let totalChars = 0;
        messageHistory.forEach(item => {
            totalChars += (item.user || '').length + (item.assistant || '').length;
        });
        return Math.ceil(totalChars / 4);
    }
}

class ChatInterface {
    constructor() {
        // Theory of Mind output boxes
        this.llmOutput = document.getElementById('llm-output');
        this.memoryOutput = document.getElementById('memory-output');
        this.messageInput = document.getElementById('message-input');
        this.sendButton = document.getElementById('send-btn');
        this.categorySelect = document.getElementById('category-select');
        this.temperatureSlider = document.getElementById('temperature-slider');
        this.temperatureValue = document.getElementById('temperature-value');
        this.streamToggle = document.getElementById('stream-toggle');

        // Metadata displays
        this.responseConfidence = document.getElementById('response-confidence');
        this.responseTime = document.getElementById('response-time');
        this.thoughtConfidence = document.getElementById('thought-confidence');
        this.thoughtNodes = document.getElementById('thought-nodes');

        this.isProcessing = false;
        this.messageHistory = [];
        this.currentCategory = '';
        this.conversationId = this.generateConversationId();

        // Initialize context manager
        this.contextManager = new ContextManager();

        this.setupEventListeners();
        this.setupConversationControls();
        this.loadConversationFromStorage();
        this.updateSendButton();
        this.updateConversationStats();
    }

    setupEventListeners() {
        // Send message on button click
        this.sendButton.addEventListener('click', () => this.sendMessage());

        // Send message on Enter key (handled by UI now)
        // Update send button state on input
        this.messageInput.addEventListener('input', () => this.updateSendButton());

        // Update temperature display
        this.temperatureSlider.addEventListener('input', () => {
            this.temperatureValue.textContent = this.temperatureSlider.value;
        });
    }

    setupModeSwitching() {
        // Mode tab event listeners
        document.getElementById('llm-mode-btn').addEventListener('click', () => this.switchMode('llm'));
        document.getElementById('memory-mode-btn').addEventListener('click', () => this.switchMode('memory'));
    }

    switchMode(mode) {
        if (this.currentMode === mode) return;

        // Update mode
        this.currentMode = mode;

        // Update UI
        document.querySelectorAll('.mode-tab').forEach(tab => tab.classList.remove('active'));
        document.querySelectorAll('.chat-section').forEach(section => section.classList.remove('active'));

        document.getElementById(`${mode}-mode-btn`).classList.add('active');
        document.getElementById(`${mode}-chat`).classList.add('active');

        // Update mode info
        this.updateModeInfo();

        console.log(`🔄 Switched to ${mode} mode`);
    }

    updateModeInfo() {
        const modeInfoEl = document.getElementById('current-mode-info');
        const modeNames = {
            llm: 'LLM Mode',
            memory: 'Memory Mode'
        };
        if (modeInfoEl) {
            modeInfoEl.textContent = `Currently using: ${modeNames[this.currentMode]}`;
        }
    }

    updateSendButton() {
        const hasText = this.messageInput.value.trim().length > 0;
        const isOnline = api.isOnline;
        this.sendButton.disabled = !hasText || !isOnline || this.isProcessing;
    }

    async sendMessage() {
        if (this.isProcessing || !api.isOnline) return;

        const message = this.messageInput.value.trim();
        if (!message) return;

        // Clear input and reset height
        this.messageInput.value = '';
        this.messageInput.style.height = 'auto';
        this.updateSendButton();

        // Set processing state
        this.isProcessing = true;
        this.updateSendButton();

        // Clear previous outputs
        this.clearOutputs();

        // Show processing state
        this.showProcessingState();

        const startTime = Date.now();

        try {
            // Prepare conversation context
            const conversationContext = this.contextManager.buildConversationContext(this.messageHistory);

            const options = {
                category: this.currentCategory || this.categorySelect.value,
                temperature: parseFloat(this.temperatureSlider.value),
                stream: this.streamToggle.checked,
                conversation_id: this.conversationId,
                conversation_messages: conversationContext
            };

            // THEORY OF MIND: Process both paths simultaneously
            // Path 1: User Input → Embedding → Memory Associations → LLM → Final Output Message (Box #1)
            // Path 2: Hidden State of LLM + Memory Vectors → Transformer Mesh → Concept Representation (Box #2)

            const [llmResponse, memoryResponse] = await Promise.allSettled([
                // Path 1: LLM conversational response
                api.sendQuery(message, options),
                // Path 2: Memory transformer internal representation
                api.sendMemoryChat(message, conversationContext)
            ]);

            // Clear processing state
            this.clearProcessingState();

            // Process LLM Response (Box #1 - What it says)
            if (llmResponse.status === 'fulfilled' && llmResponse.value.success) {
                const llmData = llmResponse.value.data;
                const responseText = llmData.answer || 'I apologize, but I couldn\'t generate a response.';
                const confidence = llmData.confidence || '--';

                this.displayLLMOutput(responseText, confidence);

                // Update response metadata
                const responseTime = Date.now() - startTime;
                if (this.responseTime) this.responseTime.textContent = `Time: ${responseTime}ms`;
            } else {
                this.displayLLMOutput('Error generating conversational response.', '--');
            }

            // Process Memory Response (Box #2 - What it thinks)
            if (memoryResponse.status === 'fulfilled' && memoryResponse.value.success) {
                const memoryData = memoryResponse.value.data;
                const thoughtStructure = this.formatThoughtStructure(memoryData);
                const nodeCount = memoryData.node_count || memoryData.nodes || '--';

                this.displayMemoryOutput(thoughtStructure, nodeCount);
            } else {
                this.displayMemoryOutput('Error processing internal thought structure.', '--');
            }

            // Store in conversation history
            this.messageHistory.push({
                user: message,
                llm_response: llmResponse.status === 'fulfilled' ? llmResponse.value.data : null,
                memory_response: memoryResponse.status === 'fulfilled' ? memoryResponse.value.data : null,
                timestamp: Date.now()
            });

            // Update conversation stats
            this.updateConversationStats();

            // Auto-store conversation in memory
            this.storeConversationInMemory();

        } catch (error) {
            console.error('Theory of Mind processing error:', error);
            this.clearProcessingState();
            this.displayLLMOutput('Error processing request.', '--');
            this.displayMemoryOutput('Error in thought processing.', '--');
        } finally {
            this.isProcessing = false;
            this.updateSendButton();
        }
    }

    clearOutputs() {
        // Clear LLM output placeholder
        if (this.llmOutput) {
            this.llmOutput.innerHTML = '<div class="output-placeholder"><div class="placeholder-icon">💬</div><p>Conversational response will appear here</p></div>';
        }

        // Clear memory output placeholder
        if (this.memoryOutput) {
            this.memoryOutput.innerHTML = '<div class="output-placeholder"><div class="placeholder-icon">🧠</div><p>Internal thought structure will appear here</p></div>';
        }

        // Reset metadata
        if (this.responseConfidence) this.responseConfidence.textContent = 'Confidence: --';
        if (this.responseTime) this.responseTime.textContent = 'Time: --';
        if (this.thoughtConfidence) this.thoughtConfidence.textContent = 'Confidence: --';
        if (this.thoughtNodes) this.thoughtNodes.textContent = 'Nodes: --';
    }

    showProcessingState() {
        if (this.llmOutput) {
            this.llmOutput.innerHTML = '<div class="processing-indicator"><div class="typing-indicator"><span></span><span></span><span></span></div><p>Generating conversational response...</p></div>';
        }
        if (this.memoryOutput) {
            this.memoryOutput.innerHTML = '<div class="processing-indicator"><div class="typing-indicator"><span></span><span></span><span></span></div><p>Processing internal thought structure...</p></div>';
        }
    }

    clearProcessingState() {
        // Remove processing indicators
        const processingIndicators = document.querySelectorAll('.processing-indicator');
        processingIndicators.forEach(indicator => indicator.remove());
    }

    displayLLMOutput(text, confidence) {
        if (!this.llmOutput) return;

        this.llmOutput.innerHTML = this.formatMessage(text);

        if (this.responseConfidence) {
            this.responseConfidence.textContent = `Confidence: ${confidence}`;
        }
    }

    displayMemoryOutput(thoughtStructure, nodeCount) {
        if (!this.memoryOutput) return;

        this.memoryOutput.innerHTML = thoughtStructure;

        if (this.thoughtNodes) {
            this.thoughtNodes.textContent = `Nodes: ${nodeCount}`;
        }
    }

    formatThoughtStructure(memoryData) {
        if (!memoryData) return '<p>No thought structure available.</p>';

        let html = '<div class="thought-structure">';

        // Show key concepts and relationships
        if (memoryData.concepts && memoryData.concepts.length > 0) {
            html += '<h5>Key Concepts:</h5><ul>';
            memoryData.concepts.slice(0, 5).forEach(concept => {
                html += `<li>${concept}</li>`;
            });
            html += '</ul>';
        }

        // Show neural connections
        if (memoryData.connections && memoryData.connections.length > 0) {
            html += '<h5>Neural Connections:</h5><ul>';
            memoryData.connections.slice(0, 3).forEach(connection => {
                html += `<li>${connection.source} → ${connection.target} (strength: ${connection.weight?.toFixed(2) || 'N/A'})</li>`;
            });
            html += '</ul>';
        }

        // Show activation patterns
        if (memoryData.activation_pattern) {
            html += `<h5>Activation Pattern:</h5><p>${memoryData.activation_pattern}</p>`;
        }

        // Show reasoning trace if available
        if (memoryData.reasoning_trace && memoryData.reasoning_trace.length > 0) {
            html += '<h5>Reasoning Steps:</h5><ol>';
            memoryData.reasoning_trace.slice(0, 3).forEach(step => {
                html += `<li>${step}</li>`;
            });
            html += '</ol>';
        }

        html += '</div>';
        return html;
    }

    getCurrentMessagesContainer() {
        return this.currentMode === 'memory' ? this.memoryMessagesContainer : this.llmMessagesContainer;
    }

    get messagesContainer() {
        return this.getCurrentMessagesContainer();
    }

    addMessage(type, content, subtype = '', sources = [], confidence = '--') {
        const container = this.getCurrentMessagesContainer();

        // Remove welcome message if this is the first real message
        if (type !== 'system' && !subtype) {
            const welcomeMessage = container.querySelector('.welcome-message');
            if (welcomeMessage) {
                welcomeMessage.remove();
            }
        }

        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}-message`;

        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';

        if (subtype === 'typing') {
            messageDiv.classList.add('typing');
            contentDiv.innerHTML = '<div class="typing-indicator"><span></span><span></span><span></span></div><em>Frankenstino AI is thinking...</em>';
        } else if (subtype === 'error') {
            contentDiv.innerHTML = `<strong>Frankenstino AI:</strong> <span class="error">${content}</span>`;
        } else {
            contentDiv.innerHTML = this.formatMessage(content);

            // Add sources if available
            if (sources && sources.length > 0) {
                const sourcesDiv = document.createElement('div');
                sourcesDiv.className = 'sources';
                sourcesDiv.innerHTML = `<strong>Sources:</strong> ${sources.join(', ')}`;
                contentDiv.appendChild(sourcesDiv);
            }

            // Add confidence for memory mode
            if (this.currentMode === 'memory' && confidence !== '--') {
                const confidenceDiv = document.createElement('div');
                confidenceDiv.className = 'confidence';
                confidenceDiv.innerHTML = `<strong>Confidence:</strong> ${confidence}`;
                contentDiv.appendChild(confidenceDiv);
            }
        }

        messageDiv.appendChild(contentDiv);
        container.appendChild(messageDiv);

        // Scroll to bottom
        this.scrollToBottom();

        return messageDiv;
    }

    updateMessageCounts() {
        this.llmMessageCount = this.messageHistory.llm.length;
        this.memoryMessageCount = this.messageHistory.memory.length;

        const llmCountEl = document.getElementById('llm-message-count');
        const memoryCountEl = document.getElementById('memory-message-count');
        const memoryConfidenceEl = document.getElementById('memory-confidence');

        if (llmCountEl) llmCountEl.textContent = `${this.llmMessageCount} messages`;
        if (memoryCountEl) memoryCountEl.textContent = `${this.memoryMessageCount} messages`;

        // Update memory confidence display
        if (memoryConfidenceEl && this.messageHistory.memory.length > 0) {
            const lastMemoryMessage = this.messageHistory.memory[this.messageHistory.memory.length - 1];
            memoryConfidenceEl.textContent = `Confidence: ${lastMemoryMessage.confidence || '--'}`;
        }
    }

    removeMessage(messageElement) {
        if (messageElement && messageElement.parentNode) {
            messageElement.parentNode.removeChild(messageElement);
        }
    }

    formatMessage(text) {
        // Basic formatting - escape HTML and convert line breaks
        return text
            .replace(/&/g, '&')
            .replace(/</g, '<')
            .replace(/>/g, '>')
            .replace(/\n/g, '<br>')
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>');
    }

    scrollToBottom() {
        this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
    }

    clearChat() {
        // Keep only the welcome message if it exists
        const welcomeMessage = this.messagesContainer.querySelector('.welcome-message');
        this.messagesContainer.innerHTML = '';
        if (welcomeMessage) {
            this.messagesContainer.appendChild(welcomeMessage.parentElement);
        }
    }

    clearChatHistory() {
        this.messageHistory = [];
    }

    showWelcomeMessage() {
        // Welcome message is now in HTML, just make sure it's visible
        const welcomeMessage = this.messagesContainer.querySelector('.welcome-message');
        if (!welcomeMessage) {
            // Recreate welcome message if it was removed
            const welcomeHTML = `
                <div class="welcome-message">
                    <div class="welcome-content">
                        <div class="welcome-icon">🧠</div>
                        <h2>Welcome to Frankenstino AI</h2>
                        <p>I'm your intelligent assistant with a hybrid memory system that learns and evolves through our conversations.</p>

                        <div class="feature-grid">
                            <div class="feature-card">
                                <div class="feature-icon">📚</div>
                                <h3>Document Memory</h3>
                                <p>Ingest documents to build personalized knowledge base</p>
                            </div>
                            <div class="feature-card">
                                <div class="feature-icon">🧲</div>
                                <h3>Evolving Memory</h3>
                                <p>Neural connections strengthen with usage patterns</p>
                            </div>
                            <div class="feature-card">
                                <div class="feature-icon">🔍</div>
                                <h3>Smart Search</h3>
                                <p>Hybrid search across hierarchical, vector, and neural layers</p>
                            </div>
                            <div class="feature-card">
                                <div class="feature-icon">⚡</div>
                                <h3>Local & Private</h3>
                                <p>Everything runs locally with complete privacy</p>
                            </div>
                        </div>

                        <div class="getting-started">
                            <h3>Getting Started</h3>
                            <ol>
                                <li><strong>Ingest Documents:</strong> Click the folder icon to add documents</li>
                                <li><strong>Ask Questions:</strong> Type questions about your documents below</li>
                                <li><strong>Explore Memory:</strong> View statistics and connection patterns</li>
                            </ol>
                        </div>
                    </div>
                </div>
            `;
            this.messagesContainer.insertAdjacentHTML('afterbegin', welcomeHTML);
        }
    }

    updateConnectionStatus(isOnline) {
        this.updateSendButton();
    }

    updateCategory(category) {
        this.currentCategory = category;
    }

    getChatHistory() {
        return this.messageHistory;
    }

    exportChat() {
        const history = this.getChatHistory();
        const exportData = {
            export_date: new Date().toISOString(),
            total_messages: history.length,
            conversations: history,
            system_info: {
                version: '1.0.0',
                model: 'Gemma-3-1B',
                memory_layers: ['hierarchical', 'vector', 'neural_mesh']
            }
        };

        const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);

        const a = document.createElement('a');
        a.href = url;
        a.download = `frankenstino_chat_${new Date().toISOString().split('T')[0]}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

        ui.showNotification('Chat exported successfully!', 'success');
    }

    generateConversationId() {
        return `conv_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }

    // ===== CONVERSATION PERSISTENCE (localStorage) =====
    saveConversationToStorage() {
        if (this.messageHistory.length === 0) return;

        try {
            const conversationData = {
                id: this.conversationId,
                messages: this.messageHistory,
                category: this.currentCategory,
                timestamp: Date.now(),
                title: this.generateConversationTitle(),
                messageCount: this.messageHistory.length,
                lastActivity: Date.now()
            };

            localStorage.setItem(`conversation_${this.conversationId}`, JSON.stringify(conversationData));
            console.log('💾 Conversation saved to localStorage:', this.conversationId);
        } catch (error) {
            console.error('❌ Failed to save conversation to localStorage:', error);
        }
    }

    loadConversationFromStorage() {
        try {
            const saved = localStorage.getItem(`conversation_${this.conversationId}`);
            if (saved) {
                const data = JSON.parse(saved);
                this.messageHistory = data.messages || [];
                this.currentCategory = data.category || '';

                // Restore chat messages to UI
                this.restoreChatMessages();
                console.log('📂 Conversation loaded from localStorage:', this.conversationId);
            }
        } catch (error) {
            console.error('❌ Failed to load conversation from localStorage:', error);
        }
    }

    restoreChatMessages() {
        // Clear current messages
        const existingMessages = this.messagesContainer.querySelectorAll('.message:not(.welcome-message)');
        existingMessages.forEach(msg => msg.remove());

        // Restore messages from history
        this.messageHistory.forEach(item => {
            this.addMessage('user', item.user);
            this.addMessage('assistant', item.assistant, 'response', item.sources || []);
        });
    }

    generateConversationTitle() {
        if (this.messageHistory.length === 0) return 'New Conversation';

        // Use first user message as title
        const firstMessage = this.messageHistory[0].user;
        if (firstMessage.length <= 50) return firstMessage;

        // Truncate long messages
        return firstMessage.substring(0, 47) + '...';
    }

    // ===== CONVERSATION CONTROLS =====
    setupConversationControls() {
        // Create conversation controls container
        const controlsContainer = document.createElement('div');
        controlsContainer.className = 'conversation-controls';
        controlsContainer.innerHTML = `
            <button id="new-conversation-btn" class="control-btn" title="Start New Conversation">
                🆕 New
            </button>
            <button id="clear-conversation-btn" class="control-btn" title="Clear Conversation History">
                🗑️ Clear
            </button>
            <button id="save-conversation-btn" class="control-btn" title="Save Conversation Locally">
                💾 Save
            </button>
            <div class="context-indicator">
                <span id="context-status">🧠 Context: Active</span>
                <span id="message-count">Messages: 0</span>
                <span id="token-count">Tokens: 0</span>
            </div>
        `;

        // Insert controls before chat messages
        const chatContainer = document.querySelector('.chat-container');
        if (chatContainer) {
            chatContainer.insertBefore(controlsContainer, this.messagesContainer);
        }

        // Setup event listeners
        document.getElementById('new-conversation-btn').addEventListener('click', () => this.newConversation());
        document.getElementById('clear-conversation-btn').addEventListener('click', () => this.clearConversation());
        document.getElementById('save-conversation-btn').addEventListener('click', () => this.saveConversationLocally());
    }

    clearConversation() {
        if (confirm('Are you sure you want to clear the conversation history? This cannot be undone.')) {
            this.messageHistory = [];
            this.clearChat();
            this.showWelcomeMessage();
            this.updateConversationStats();
            ui.showNotification('Conversation cleared', 'info');
        }
    }

    saveConversationLocally() {
        this.saveConversationToStorage();
        ui.showNotification('Conversation saved locally', 'success');
    }

    // ===== CONVERSATION STATS =====
    updateConversationStats() {
        const messageCount = this.messageHistory.length;
        const tokenCount = this.contextManager.estimateTokenCount(this.messageHistory);

        const messageEl = document.getElementById('message-count');
        const tokenEl = document.getElementById('token-count');
        const contextEl = document.getElementById('context-status');

        if (messageEl) messageEl.textContent = `Messages: ${messageCount}`;
        if (tokenEl) tokenEl.textContent = `Tokens: ${tokenCount}`;
        if (contextEl) {
            const contextStatus = messageCount > 0 ? 'Active' : 'Empty';
            contextEl.textContent = `🧠 Context: ${contextStatus}`;
        }
    }

    // ===== MEMORY OPERATION FEEDBACK =====
    showMemoryOperationFeedback(operation, success, details = {}) {
        const notification = document.createElement('div');
        notification.className = `memory-notification ${success ? 'success' : 'error'}`;

        let message = '';
        if (operation === 'store') {
            message = success
                ? `✅ Added to memory: ${details.chunks || 0} chunks stored`
                : `❌ Memory storage failed`;
        } else if (operation === 'retrieve') {
            message = success
                ? `✅ Retrieved from memory: ${details.chunks || 0} chunks found`
                : `❌ Memory retrieval failed`;
        } else if (operation === 'search') {
            message = success
                ? `✅ Memory search completed: ${details.results || 0} results`
                : `❌ Memory search failed`;
        }

        notification.innerHTML = message;

        // Add to notifications area or show as toast
        document.body.appendChild(notification);

        // Auto-remove after 3 seconds
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 3000);
    }

    async storeConversationInMemory() {
        if (this.messageHistory.length === 0) return;

        console.log('🔄 Storing conversation in memory...', {
            conversationId: this.conversationId,
            messageCount: this.messageHistory.length,
            category: this.currentCategory || 'general'
        });

        try {
            // Convert message history to API format
            const messages = [];
            this.messageHistory.forEach(item => {
                messages.push({ role: 'user', content: item.user });
                messages.push({ role: 'assistant', content: item.assistant });
            });

            console.log('📤 Calling API to store conversation...');
            const result = await api.storeConversation(this.conversationId, messages, this.currentCategory || 'general');

            // Show feedback
            this.showMemoryOperationFeedback('store', result.success, {
                chunks: Math.ceil(messages.length / 2) // Rough estimate
            });

            console.log('✅ Conversation stored successfully:', result);
        } catch (error) {
            console.error('❌ Failed to store conversation in memory:', error);
            this.showMemoryOperationFeedback('store', false);
        }
    }

    newConversation() {
        // Store current conversation before starting new one
        if (this.messageHistory.length > 0) {
            this.storeConversationInMemory();
        }

        // Generate new conversation ID
        this.conversationId = this.generateConversationId();
        this.messageHistory = [];
        this.currentCategory = '';

        // Clear chat and show welcome
        this.clearChat();
        this.showWelcomeMessage();
    }
}

// Create global chat instance
const chat = new ChatInterface();
