/**
 * Frankenstino AI - Dual Chat Interface Module
 * Handles independent chat interfaces for Memory Chat (left) and LLM Chat (right)
 */

class DualChatInterface {
    constructor() {
        // Chat mode toggle elements
        this.toggleMemoryBtn = document.getElementById('toggle-memory-btn');
        this.toggleLLMBtn = document.getElementById('toggle-llm-btn');
        this.memoryPanel = document.getElementById('memory-panel');
        this.llmPanel = document.getElementById('llm-panel');

        // Memory Chat elements
        this.memoryChatHistory = document.getElementById('memory-chat-history');
        this.memoryChatInput = document.getElementById('memory-chat-input');
        this.memoryChatSendBtn = document.getElementById('memory-chat-send-btn');
        this.memoryPersonalitySelect = document.getElementById('memory-personality-select');
        this.memoryNewChatBtn = document.getElementById('memory-new-chat-btn');
        this.memoryClearChatBtn = document.getElementById('memory-clear-chat-btn');

        // LLM Chat elements
        this.llmChatHistory = document.getElementById('chat-history');
        this.llmQueryInput = document.getElementById('llm-query-input');
        this.llmQueryBtn = document.getElementById('llm-query-btn');
        this.categorySelect = document.getElementById('category-select');
        this.temperatureSlider = document.getElementById('temperature-slider');
        this.temperatureValue = document.getElementById('temperature-value');
        this.streamToggle = document.getElementById('stream-toggle');

    // Alternate header toggle buttons (there are two header groups in the DOM)
    this.toggleMemoryBtnAlt = document.getElementById('toggle-memory-btn-llm');
    this.toggleLLMBtnAlt = document.getElementById('toggle-llm-btn-llm');

    // Header and input area elements (need to be toggled when switching modes)
    this.memoryChatHeader = document.getElementById('memory-chat-header');
    this.llmChatHeader = document.getElementById('llm-chat-header');
    this.memoryInputArea = document.getElementById('memory-input-area');
    this.llmInputArea = document.getElementById('llm-input-area');

        // State management
        this.currentMode = 'memory'; // 'memory' or 'llm'
        this.memoryChatHistoryData = [];
        this.llmChatHistoryData = [];
        this.memoryConversationId = this.generateConversationId();
        this.llmConversationId = this.generateConversationId();
        this.isMemoryProcessing = false;
        this.isLLMProcessing = false;

        this.setupEventListeners();
        this.loadCategories();
        this.updateButtonStates();
        this.updateChatMode();
    }

    setupEventListeners() {
        // Chat mode toggle events
        this.toggleMemoryBtn.addEventListener('click', () => this.switchToMemoryMode());
        this.toggleLLMBtn.addEventListener('click', () => this.switchToLLMMode());
    // Also attach listeners to the alternate toggle buttons (visible in the other header)
    if (this.toggleMemoryBtnAlt) this.toggleMemoryBtnAlt.addEventListener('click', () => this.switchToMemoryMode());
    if (this.toggleLLMBtnAlt) this.toggleLLMBtnAlt.addEventListener('click', () => this.switchToLLMMode());

        // History modal events
        document.getElementById('history-btn').addEventListener('click', () => this.showHistoryModal());

        // Memory Chat Events
        this.memoryChatSendBtn.addEventListener('click', () => this.sendMemoryMessage());
        // Use keydown for reliable Enter detection across browsers; fallback to keyCode
        this.memoryChatInput.addEventListener('keydown', (e) => {
            const isEnter = e.key === 'Enter' || e.keyCode === 13;
            if (isEnter && !e.shiftKey) {
                e.preventDefault();
                this.sendMemoryMessage();
            }
        });
        this.memoryChatInput.addEventListener('input', () => this.updateButtonStates());
        this.memoryNewChatBtn.addEventListener('click', () => this.newMemoryChat());
        this.memoryClearChatBtn.addEventListener('click', () => this.clearMemoryChat());

        // LLM Chat Events
        this.llmQueryBtn.addEventListener('click', () => this.sendLLMMessage());
        // Use keydown for reliable Enter detection across browsers; fallback to keyCode
        this.llmQueryInput.addEventListener('keydown', (e) => {
            const isEnter = e.key === 'Enter' || e.keyCode === 13;
            if (isEnter && !e.shiftKey) {
                e.preventDefault();
                this.sendLLMMessage();
            }
        });
        this.llmQueryInput.addEventListener('input', () => this.updateButtonStates());
        this.temperatureSlider.addEventListener('input', () => {
            this.temperatureValue.textContent = this.temperatureSlider.value;
        });
    }

    async sendMemoryMessage() {
        if (this.isMemoryProcessing || !api.isOnline) return;

        const message = this.memoryChatInput.value.trim();
        if (!message) return;

        // Clear input and update UI
        this.memoryChatInput.value = '';
        this.memoryChatInput.style.height = 'auto';
        this.updateButtonStates();

        // Set processing state
        this.isMemoryProcessing = true;
        this.updateButtonStates();

        // Add user message to UI
        this.addMemoryMessage('user', message);

        // Show typing indicator
        this.showMemoryTypingIndicator();

        try {
            // Build conversation history for API
            const conversationHistory = this.buildMemoryConversationHistory();

            // Send to memory chat API
            const response = await api.sendMemoryChat(message, conversationHistory);

            // Remove typing indicator
            this.removeMemoryTypingIndicator();

            if (response.success && response.data) {
                const aiResponse = response.data.response || 'I apologize, but I couldn\'t generate a response from memory.';
                const confidence = response.data.confidence || '--';
                const personality = response.data.personality || 'analytical';

                // Add AI response to UI
                this.addMemoryMessage('assistant', aiResponse, { confidence, personality });

                // Store in conversation history
                this.memoryChatHistoryData.push({
                    user: message,
                    assistant: aiResponse,
                    timestamp: Date.now(),
                    metadata: { confidence, personality }
                });

                // Auto-save conversation
                this.saveMemoryConversation();
            } else {
                this.addMemoryMessage('assistant', 'Error: Could not get response from memory system.', { error: true });
            }

        } catch (error) {
            console.error('Memory chat error:', error);
            this.removeMemoryTypingIndicator();
            this.addMemoryMessage('assistant', `Error: ${api.formatError(error)}`, { error: true });
        } finally {
            this.isMemoryProcessing = false;
            this.updateButtonStates();
        }
    }

    async sendLLMMessage() {
        if (this.isLLMProcessing || !api.isOnline) return;

        const message = this.llmQueryInput.value.trim();
        if (!message) return;

        // Clear input and update UI
        this.llmQueryInput.value = '';
        this.llmQueryInput.style.height = 'auto';
        this.updateButtonStates();

        // Set processing state
        this.isLLMProcessing = true;
        this.updateButtonStates();

        // Add user message to UI
        this.addLLMMessage('user', message);

        // Show typing indicator
        this.showLLMTypingIndicator();

        try {
            // Build conversation context
            const conversationContext = this.buildLLMConversationHistory();

            const options = {
                category: this.categorySelect.value,
                temperature: parseFloat(this.temperatureSlider.value),
                stream: this.streamToggle.checked,
                conversation_id: this.llmConversationId,
                conversation_messages: conversationContext
            };

            // Send to LLM API
            const response = await api.sendQuery(message, options);

            // Remove typing indicator
            this.removeLLMTypingIndicator();

            if (response.success && response.data) {
                const aiResponse = response.data.answer || response.data.response || 'I apologize, but I couldn\'t generate a response.';
                const confidence = response.data.confidence || '--';

                // Add AI response to UI
                this.addLLMMessage('assistant', aiResponse, { confidence });

                // Store in conversation history
                this.llmChatHistoryData.push({
                    user: message,
                    assistant: aiResponse,
                    timestamp: Date.now(),
                    metadata: { confidence, category: options.category }
                });

                // Auto-save conversation
                this.saveLLMConversation();
            } else {
                this.addLLMMessage('assistant', 'Error: Could not get response from LLM.', { error: true });
            }

        } catch (error) {
            console.error('LLM chat error:', error);
            this.removeLLMTypingIndicator();
            this.addLLMMessage('assistant', `Error: ${api.formatError(error)}`, { error: true });
        } finally {
            this.isLLMProcessing = false;
            this.updateButtonStates();
        }
    }

    addMemoryMessage(type, content, metadata = {}) {
        // Remove placeholder if it exists
        const placeholder = this.memoryChatHistory.querySelector('.chat-placeholder');
        if (placeholder) placeholder.remove();

        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}-message`;

        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';

        if (metadata.error) {
            contentDiv.innerHTML = `<strong>Memory AI:</strong> <span class="error">${content}</span>`;
        } else if (type === 'user') {
            contentDiv.innerHTML = `<strong>You:</strong> ${this.formatMessage(content)}`;
        } else {
            const personalityIcon = this.getPersonalityIcon(metadata.personality);
            contentDiv.innerHTML = `<strong>Memory AI ${personalityIcon}:</strong> ${this.formatMessage(content)}`;
            if (metadata.confidence && metadata.confidence !== '--') {
                contentDiv.innerHTML += `<div class="message-meta">Confidence: ${metadata.confidence}</div>`;
            }
        }

        messageDiv.appendChild(contentDiv);
        this.memoryChatHistory.appendChild(messageDiv);

        // Scroll to bottom
        this.memoryChatHistory.scrollTop = this.memoryChatHistory.scrollHeight;
    }

    addLLMMessage(type, content, metadata = {}) {
        // Remove placeholder if it exists
        const placeholder = this.llmChatHistory.querySelector('.chat-placeholder');
        if (placeholder) placeholder.remove();

        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}-message`;

        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';

        if (metadata.error) {
            contentDiv.innerHTML = `<strong>Frankenstino AI:</strong> <span class="error">${content}</span>`;
        } else if (type === 'user') {
            contentDiv.innerHTML = `<strong>You:</strong> ${this.formatMessage(content)}`;
        } else {
            contentDiv.innerHTML = `<strong>Frankenstino AI:</strong> ${this.formatMessage(content)}`;
            if (metadata.confidence && metadata.confidence !== '--') {
                contentDiv.innerHTML += `<div class="message-meta">Confidence: ${metadata.confidence}</div>`;
            }
        }

        messageDiv.appendChild(contentDiv);
        this.llmChatHistory.appendChild(messageDiv);

        // Scroll to bottom
        this.llmChatHistory.scrollTop = this.llmChatHistory.scrollHeight;
    }

    showMemoryTypingIndicator() {
        const typingDiv = document.createElement('div');
        typingDiv.className = 'message assistant-message typing';
        typingDiv.id = 'memory-typing-indicator';
        typingDiv.innerHTML = `
            <div class="message-content">
                <strong>Memory AI:</strong>
                <div class="typing-indicator">
                    <span></span><span></span><span></span>
                </div>
                <em>Thinking through memory...</em>
            </div>
        `;
        this.memoryChatHistory.appendChild(typingDiv);
        this.memoryChatHistory.scrollTop = this.memoryChatHistory.scrollHeight;
    }

    removeMemoryTypingIndicator() {
        const typingIndicator = document.getElementById('memory-typing-indicator');
        if (typingIndicator) typingIndicator.remove();
    }

    showLLMTypingIndicator() {
        const typingDiv = document.createElement('div');
        typingDiv.className = 'message assistant-message typing';
        typingDiv.id = 'llm-typing-indicator';
        typingDiv.innerHTML = `
            <div class="message-content">
                <strong>Frankenstino AI:</strong>
                <div class="typing-indicator">
                    <span></span><span></span><span></span>
                </div>
                <em>Generating response...</em>
            </div>
        `;
        this.llmChatHistory.appendChild(typingDiv);
        this.llmChatHistory.scrollTop = this.llmChatHistory.scrollHeight;
    }

    removeLLMTypingIndicator() {
        const typingIndicator = document.getElementById('llm-typing-indicator');
        if (typingIndicator) typingIndicator.remove();
    }

    buildMemoryConversationHistory() {
        return this.memoryChatHistoryData.flatMap(item => [
            { role: 'user', content: item.user },
            { role: 'assistant', content: item.assistant }
        ]);
    }

    buildLLMConversationHistory() {
        return this.llmChatHistoryData.flatMap(item => [
            { role: 'user', content: item.user },
            { role: 'assistant', content: item.assistant }
        ]);
    }

    newMemoryChat() {
        if (this.memoryChatHistoryData.length > 0) {
            this.saveMemoryConversation();
        }

        this.memoryChatHistoryData = [];
        this.memoryConversationId = this.generateConversationId();

        // Clear chat UI
        this.memoryChatHistory.innerHTML = `
            <div class="chat-placeholder">
                <div class="placeholder-icon">🧠</div>
                <p>Memory chat messages will appear here</p>
            </div>
        `;

        ui.showNotification('New memory chat started', 'info');
    }

    clearMemoryChat() {
        if (confirm('Are you sure you want to clear the memory chat history? This cannot be undone.')) {
            this.memoryChatHistoryData = [];

            // Safer DOM manipulation - remove all child elements individually
            while (this.memoryChatHistory.firstChild) {
                this.memoryChatHistory.removeChild(this.memoryChatHistory.firstChild);
            }

            // Add placeholder back
            const placeholder = document.createElement('div');
            placeholder.className = 'chat-placeholder';
            placeholder.innerHTML = `
                <div class="placeholder-icon">🧠</div>
                <p>Memory chat messages will appear here</p>
            `;
            this.memoryChatHistory.appendChild(placeholder);

            ui.showNotification('Memory chat cleared', 'info');
        }
    }



    updateButtonStates() {
        const hasMemoryText = this.memoryChatInput.value.trim().length > 0;
        const hasLLMText = this.llmQueryInput.value.trim().length > 0;
        const isOnline = api.isOnline;

        this.memoryChatSendBtn.disabled = !hasMemoryText || !isOnline || this.isMemoryProcessing;
        this.llmQueryBtn.disabled = !hasLLMText || !isOnline || this.isLLMProcessing;
    }

    updateConnectionStatus(isOnline) {
        this.updateButtonStates();
    }

    switchToMemoryMode() {
        this.currentMode = 'memory';
        this.updateChatMode();
    }

    switchToLLMMode() {
        this.currentMode = 'llm';
        this.updateChatMode();
    }

    updateChatMode() {
        // Update toggle button states
        this.toggleMemoryBtn.classList.toggle('active', this.currentMode === 'memory');
        this.toggleLLMBtn.classList.toggle('active', this.currentMode === 'llm');
        if (this.toggleMemoryBtnAlt) this.toggleMemoryBtnAlt.classList.toggle('active', this.currentMode === 'memory');
        if (this.toggleLLMBtnAlt) this.toggleLLMBtnAlt.classList.toggle('active', this.currentMode === 'llm');

        // Show/hide panels
        this.memoryPanel.style.display = this.currentMode === 'memory' ? 'flex' : 'none';
        this.llmPanel.style.display = this.currentMode === 'llm' ? 'flex' : 'none';

        // Show/hide headers (there are separate header blocks for each mode)
        if (this.memoryChatHeader && this.llmChatHeader) {
            this.memoryChatHeader.style.display = this.currentMode === 'memory' ? 'flex' : 'none';
            this.llmChatHeader.style.display = this.currentMode === 'llm' ? 'flex' : 'none';
        }

        // Show/hide input areas (memory vs llm input areas)
        if (this.memoryInputArea && this.llmInputArea) {
            this.memoryInputArea.style.display = this.currentMode === 'memory' ? 'block' : 'none';
            this.llmInputArea.style.display = this.currentMode === 'llm' ? 'block' : 'none';
        }

        // Update button states
        this.updateButtonStates();
    }

    async loadCategories() {
        try {
            const categories = await api.getCategories();
            this.categorySelect.innerHTML = '<option value="">All Categories</option>';

            categories.forEach(category => {
                const option = document.createElement('option');
                option.value = category.name;
                option.textContent = `${category.name} (${category.count})`;
                this.categorySelect.appendChild(option);
            });
        } catch (error) {
            console.warn('Failed to load categories:', error);
        }
    }

    formatMessage(text) {
        // Basic formatting
        return text
            .replace(/&/g, '&')
            .replace(/</g, '<')
            .replace(/>/g, '>')
            .replace(/\n/g, '<br>')
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>');
    }

    getPersonalityIcon(personality) {
        const icons = {
            'analytical': '🧠',
            'creative': '🎨',
            'helpful': '🤝'
        };
        return icons[personality] || '🧠';
    }

    generateConversationId() {
        return `conv_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }

    generateConversationTitle(firstMessage) {
        if (!firstMessage) return 'New Chat';

        // Clean and process the message
        const cleanMessage = firstMessage.trim();

        // Extract key words and create a meaningful title
        const words = cleanMessage.split(/\s+/).filter(word => word.length > 2);

        // Look for question words or common patterns
        const questionWords = ['what', 'how', 'why', 'when', 'where', 'who', 'which', 'can', 'could', 'would', 'should', 'help', 'explain', 'tell'];
        const firstWord = words[0]?.toLowerCase();

        if (questionWords.includes(firstWord)) {
            // For questions, take first 4-6 words
            const titleWords = words.slice(0, Math.min(6, words.length));
            let title = titleWords.join(' ');
            if (title.length > 40) {
                title = title.substring(0, 37) + '...';
            }
            // Capitalize first letter
            return title.charAt(0).toUpperCase() + title.slice(1);
        } else {
            // For statements, extract key nouns/topics
            const importantWords = words.filter(word =>
                word.length > 3 && // Longer words
                !['with', 'from', 'about', 'this', 'that', 'these', 'those', 'have', 'been', 'were', 'will', 'would', 'could', 'should'].includes(word.toLowerCase())
            );

            if (importantWords.length >= 2) {
                const titleWords = importantWords.slice(0, 4);
                let title = titleWords.join(' ');
                if (title.length > 40) {
                    title = title.substring(0, 37) + '...';
                }
                return title.charAt(0).toUpperCase() + title.slice(1);
            } else {
                // Fallback to first part of message
                let title = cleanMessage.substring(0, 40);
                if (cleanMessage.length > 40) {
                    title += '...';
                }
                return title.charAt(0).toUpperCase() + title.slice(1);
            }
        }
    }

    saveMemoryConversation() {
        if (this.memoryChatHistoryData.length === 0) return;

        try {
            const firstMessage = this.memoryChatHistoryData[0]?.user || '';
            const title = this.generateConversationTitle(firstMessage);

            const conversationData = {
                id: this.memoryConversationId,
                messages: this.memoryChatHistoryData,
                timestamp: Date.now(),
                title: title,
                messageCount: this.memoryChatHistoryData.length,
                type: 'memory_chat'
            };

            localStorage.setItem(`memory_conversation_${this.memoryConversationId}`, JSON.stringify(conversationData));
            console.log('💾 Memory conversation saved:', title);
        } catch (error) {
            console.error('❌ Failed to save memory conversation:', error);
        }
    }

    saveLLMConversation() {
        if (this.llmChatHistoryData.length === 0) return;

        try {
            const firstMessage = this.llmChatHistoryData[0]?.user || '';
            const title = this.generateConversationTitle(firstMessage);

            const conversationData = {
                id: this.llmConversationId,
                messages: this.llmChatHistoryData,
                timestamp: Date.now(),
                title: title,
                messageCount: this.llmChatHistoryData.length,
                type: 'llm_chat'
            };

            localStorage.setItem(`llm_conversation_${this.llmConversationId}`, JSON.stringify(conversationData));
            console.log('💾 LLM conversation saved:', title);
        } catch (error) {
            console.error('❌ Failed to save LLM conversation:', error);
        }
    }

    showHistoryModal() {
        const modal = document.getElementById('history-modal');
        modal.style.display = 'flex';

        // Load and display conversations
        this.loadConversations();

        // Setup modal event listeners
        const closeBtn = modal.querySelector('.modal-close');
        const historyTabs = modal.querySelectorAll('.history-tab-btn');

        // Remove existing listeners to avoid duplicates
        closeBtn.removeEventListener('click', this.closeHistoryModal);
        historyTabs.forEach(tab => {
            tab.removeEventListener('click', this.handleHistoryTabClick);
        });

        // Add listeners
        closeBtn.addEventListener('click', () => this.closeHistoryModal());
        historyTabs.forEach(tab => {
            tab.addEventListener('click', (e) => this.handleHistoryTabClick(e));
        });
    }

    closeHistoryModal() {
        const modal = document.getElementById('history-modal');
        modal.style.display = 'none';
    }

    handleHistoryTabClick(e) {
        const tab = e.target;
        const tabType = tab.dataset.tab;

        // Update active tab
        document.querySelectorAll('.history-tab-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        tab.classList.add('active');

        // Show corresponding content
        document.querySelectorAll('.history-tab-content').forEach(content => {
            content.classList.remove('active');
        });
        document.getElementById(`${tabType}-history-tab`).classList.add('active');
    }

    loadConversations() {
        const llmConversations = this.getStoredConversations('llm');
        const memoryConversations = this.getStoredConversations('memory');

        this.displayConversations(llmConversations, 'llm');
        this.displayConversations(memoryConversations, 'memory');
    }

    getStoredConversations(type) {
        const conversations = [];
        const prefix = type === 'llm' ? 'llm_conversation_' : 'memory_conversation_';

        for (let i = 0; i < localStorage.length; i++) {
            const key = localStorage.key(i);
            if (key && key.startsWith(prefix)) {
                try {
                    const conversation = JSON.parse(localStorage.getItem(key));
                    conversations.push(conversation);
                } catch (error) {
                    console.warn('Failed to parse conversation:', key, error);
                }
            }
        }

        // Sort by timestamp (newest first)
        return conversations.sort((a, b) => b.timestamp - a.timestamp);
    }

    displayConversations(conversations, type) {
        const container = document.getElementById(`${type}-conversations-list`);
        container.innerHTML = '';

        if (conversations.length === 0) {
            container.innerHTML = `
                <div class="no-conversations">
                    <p>No ${type} conversations found</p>
                </div>
            `;
            return;
        }

        conversations.forEach(conversation => {
            const item = document.createElement('div');
            item.className = `conversation-item ${type}-chat`;
            item.dataset.conversationId = conversation.id;

            const lastMessage = conversation.messages[conversation.messages.length - 1];
            const preview = lastMessage ? (lastMessage.assistant || lastMessage.user).substring(0, 100) + '...' : 'No messages';

            const date = new Date(conversation.timestamp).toLocaleDateString();

            item.innerHTML = `
                <div class="conversation-info">
                    <div class="conversation-title">${conversation.title}</div>
                    <div class="conversation-preview">${preview}</div>
                    <div class="conversation-meta">
                        <span>${conversation.messageCount} messages</span>
                        <span>${date}</span>
                    </div>
                </div>
                <div class="conversation-actions">
                    ${type === 'llm' ? '<button class="delete-conversation-btn" data-conversation-id="' + conversation.id + '">🗑️</button>' : ''}
                </div>
            `;

            // Add click handlers
            item.addEventListener('click', (e) => {
                if (!e.target.classList.contains('delete-conversation-btn')) {
                    this.loadConversation(conversation.id, type);
                }
            });

            if (type === 'llm') {
                const deleteBtn = item.querySelector('.delete-conversation-btn');
                deleteBtn.addEventListener('click', (e) => {
                    e.stopPropagation();
                    this.deleteConversation(conversation.id, type);
                });
            }

            container.appendChild(item);
        });
    }

    loadConversation(conversationId, type) {
        try {
            const key = `${type}_conversation_${conversationId}`;
            const conversationData = JSON.parse(localStorage.getItem(key));

            if (!conversationData) {
                ui.showNotification('Conversation not found', 'error');
                return;
            }

            // Switch to the appropriate chat mode
            if (type === 'memory') {
                this.switchToMemoryMode();
                this.memoryChatHistoryData = conversationData.messages;
                this.memoryConversationId = conversationData.id;
                this.renderMemoryConversation();
            } else {
                this.switchToLLMMode();
                this.llmChatHistoryData = conversationData.messages;
                this.llmConversationId = conversationData.id;
                this.renderLLMConversation();
            }

            // Close modal and show notification
            this.closeHistoryModal();
            ui.showNotification(`Loaded conversation: ${conversationData.title}`, 'success');

        } catch (error) {
            console.error('Failed to load conversation:', error);
            ui.showNotification('Failed to load conversation', 'error');
        }
    }

    deleteConversation(conversationId, type) {
        if (!confirm('Are you sure you want to delete this conversation? This cannot be undone.')) {
            return;
        }

        try {
            const key = `${type}_conversation_${conversationId}`;
            localStorage.removeItem(key);

            // Refresh the conversation list
            this.loadConversations();

            ui.showNotification('Conversation deleted', 'success');
        } catch (error) {
            console.error('Failed to delete conversation:', error);
            ui.showNotification('Failed to delete conversation', 'error');
        }
    }

    renderMemoryConversation() {
        this.memoryChatHistory.innerHTML = '';

        if (this.memoryChatHistoryData.length === 0) {
            this.memoryChatHistory.innerHTML = `
                <div class="chat-placeholder">
                    <div class="placeholder-icon">🧠</div>
                    <p>Memory chat messages will appear here</p>
                </div>
            `;
            return;
        }

        this.memoryChatHistoryData.forEach(message => {
            this.addMemoryMessage('user', message.user);
            if (message.assistant) {
                this.addMemoryMessage('assistant', message.assistant, message.metadata || {});
            }
        });
    }

    renderLLMConversation() {
        this.llmChatHistory.innerHTML = '';

        if (this.llmChatHistoryData.length === 0) {
            this.llmChatHistory.innerHTML = `
                <div class="chat-placeholder">
                    <div class="placeholder-icon">💬</div>
                    <p>Chat messages will appear here</p>
                </div>
            `;
            return;
        }

        this.llmChatHistoryData.forEach(message => {
            this.addLLMMessage('user', message.user);
            if (message.assistant) {
                this.addLLMMessage('assistant', message.assistant, message.metadata || {});
            }
        });
    }
}

// Create global dual chat instance
const dualChat = new DualChatInterface();
