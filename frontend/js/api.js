/**
 * Frankenstino AI - API Communication Module
 * Handles all communication with the backend
 */

class FrankenstinoAPI {
    constructor(baseURL = 'http://localhost:8000') {
        this.baseURL = baseURL;
        this.isOnline = false;
    }

    /**
     * Test connection to the backend
     */
    async checkConnection() {
        try {
            const response = await fetch(`${this.baseURL}/health`);
            if (response.ok) {
                this.isOnline = true;
                return true;
            }
        } catch (error) {
            console.warn('Backend connection failed:', error);
        }
        this.isOnline = false;
        return false;
    }

    /**
     * Send a query to the AI
     */
    async sendQuery(query, options = {}) {
        if (!this.isOnline) {
            throw new Error('Backend is not available');
        }

        const payload = {
            query: query.trim(),
            category: options.category || '',
            temperature: options.temperature || 0.7,
            stream: options.stream || false,
            conversation_id: options.conversation_id || null,
            conversation_messages: options.conversation_messages || []  // Add conversation context
        };

        const response = await fetch(`${this.baseURL}/api/query`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            const error = await response.json().catch(() => ({ error: 'Unknown error' }));
            throw new Error(error.error || `HTTP ${response.status}`);
        }

        return await response.json();
    }

    /**
     * Send a query to the memory transformer chat
     */
    async sendMemoryChat(message, conversationHistory = []) {
        if (!this.isOnline) {
            throw new Error('Backend is not available');
        }

        const payload = {
            message: message.trim(),
            conversation_history: conversationHistory
        };

        const response = await fetch(`${this.baseURL}/api/memory_chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            const error = await response.json().catch(() => ({ error: 'Unknown error' }));
            throw new Error(error.error || `HTTP ${response.status}`);
        }

        return await response.json();
    }

    /**
     * Ingest a document
     */
    async ingestDocument(filePath, category = '') {
        if (!this.isOnline) {
            throw new Error('Backend is not available');
        }

        const payload = {
            file_path: filePath,
            category: category
        };

        const response = await fetch(`${this.baseURL}/api/ingest`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            const error = await response.json().catch(() => ({ error: 'Unknown error' }));
            throw new Error(error.error || `HTTP ${response.status}`);
        }

        return await response.json();
    }

    /**
     * Get system status and statistics
     */
    async getStatus() {
        if (!this.isOnline) {
            throw new Error('Backend is not available');
        }

        const response = await fetch(`${this.baseURL}/api/status`);
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        return await response.json();
    }

    /**
     * Get available categories
     */
    async getCategories() {
        if (!this.isOnline) {
            return [];
        }

        try {
            const response = await fetch(`${this.baseURL}/api/categories`);
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }

            const result = await response.json();
            if (result.success) {
                return result.data || [];
            } else {
                console.warn('Failed to get categories:', result);
                return [];
            }
        } catch (error) {
            console.warn('Failed to get categories:', error);
            return [];
        }
    }

    /**
     * Upload and ingest a file
     */
    async uploadFile(file, category = '') {
        if (!this.isOnline) {
            throw new Error('Backend is not available');
        }

        const formData = new FormData();
        formData.append('file', file);
        if (category) {
            formData.append('category', category);
        }

        const response = await fetch(`${this.baseURL}/api/upload`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await response.json().catch(() => ({ error: 'Unknown error' }));
            throw new Error(error.error || `HTTP ${response.status}`);
        }

        return await response.json();
    }

    /**
     * Upload and ingest text content directly
     */
    async uploadText(text, title = '', category = '') {
        if (!this.isOnline) {
            throw new Error('Backend is not available');
        }

        const payload = {
            text: text.trim(),
            title: title.trim(),
            category: category.trim()
        };

        const response = await fetch(`${this.baseURL}/api/upload-text`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            const error = await response.json().catch(() => ({ error: 'Unknown error' }));
            throw new Error(error.error || `HTTP ${response.status}`);
        }

        return await response.json();
    }

    /**
     * Store a conversation in memory
     */
    async storeConversation(conversationId, messages, category = '') {
        if (!this.isOnline) {
            throw new Error('Backend is not available');
        }

        const payload = {
            conversation_id: conversationId,
            messages: messages,
            category: category
        };

        const response = await fetch(`${this.baseURL}/api/conversations`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            const error = await response.json().catch(() => ({ error: 'Unknown error' }));
            throw new Error(error.error || `HTTP ${response.status}`);
        }

        return await response.json();
    }

    /**
     * Get conversation history
     */
    async getConversations(category = '', limit = 10) {
        if (!this.isOnline) {
            throw new Error('Backend is not available');
        }

        const params = new URLSearchParams();
        if (category) params.append('category', category);
        params.append('limit', limit.toString());

        const response = await fetch(`${this.baseURL}/api/conversations?${params}`);
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        return await response.json();
    }

    /**
     * Delete a conversation
     */
    async deleteConversation(conversationId) {
        if (!this.isOnline) {
            throw new Error('Backend is not available');
        }

        const response = await fetch(`${this.baseURL}/api/conversations/${conversationId}`, {
            method: 'DELETE'
        });

        if (!response.ok) {
            const error = await response.json().catch(() => ({ error: 'Unknown error' }));
            throw new Error(error.error || `HTTP ${response.status}`);
        }

        return await response.json();
    }

    /**
     * Browse memory contents
     */
    async browseMemory(options = {}) {
        if (!this.isOnline) {
            throw new Error('Backend is not available');
        }

        const params = new URLSearchParams();
        if (options.category) params.append('category', options.category);
        if (options.contentType) params.append('content_type', options.contentType);
        if (options.limit) params.append('limit', options.limit || 50);
        if (options.offset) params.append('offset', options.offset || 0);

        const response = await fetch(`${this.baseURL}/api/memory/browse?${params}`);
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        return await response.json();
    }

    /**
     * Search through memory
     */
    async searchMemory(query, options = {}) {
        if (!this.isOnline) {
            throw new Error('Backend is not available');
        }

        const params = new URLSearchParams();
        params.append('query', query);
        if (options.category) params.append('category', options.category);
        if (options.limit) params.append('limit', options.limit || 20);

        const response = await fetch(`${this.baseURL}/api/memory/search?${params}`);
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        return await response.json();
    }

    /**
     * Get detailed memory statistics
     */
    async getDetailedMemoryStats() {
        if (!this.isOnline) {
            throw new Error('Backend is not available');
        }

        const response = await fetch(`${this.baseURL}/api/memory/stats`);
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        return await response.json();
    }

    /**
     * Get neural mesh data for visualization
     */
    async getNeuralMeshData() {
        if (!this.isOnline) {
            throw new Error('Backend is not available');
        }

        const response = await fetch(`${this.baseURL}/api/neural-mesh`);
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        return await response.json();
    }

    /**
     * Format API errors for display
     */
    formatError(error) {
        if (error.message.includes('Backend is not available')) {
            return 'Cannot connect to Frankenstino AI server. Please make sure the backend is running.';
        }

        if (error.message.includes('Model not loaded')) {
            return 'The AI model is still loading. Please wait a moment and try again.';
        }

        if (error.message.includes('No relevant context')) {
            return 'I don\'t have enough information in my knowledge base to answer this question accurately. Try ingesting some documents first.';
        }

        return error.message || 'An unexpected error occurred.';
    }
}

// Create global API instance
const api = new FrankenstinoAPI();
