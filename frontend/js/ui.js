/**
 * Frankenstino AI - UI Utilities Module
 * Handles modals, status updates, file ingestion, and other UI interactions
 */

class UIUtils {
    constructor() {
        // Modal elements
        this.settingsModal = document.getElementById('settings-modal');
        this.memoryModal = document.getElementById('memory-modal');
        this.loadingOverlay = document.getElementById('loading-overlay');
        this.notificationContainer = document.getElementById('notification-container');

        // Status elements
        this.statusIndicator = document.getElementById('status-indicator');
        this.memoryStats = document.getElementById('memory-stats');

        // Input elements - Legacy (keeping for compatibility)
        this.messageInput = document.getElementById('message-input');
        this.categorySelect = document.getElementById('category-select');
        this.fileInput = document.getElementById('file-input');

        // New split-screen input elements
        this.memoryQueryInput = document.getElementById('memory-query-input');
        this.llmQueryInput = document.getElementById('llm-query-input');
        this.memoryContextInput = document.getElementById('memory-context-input');

        // Settings elements
        this.globalTemperature = document.getElementById('global-temperature');
        this.globalTempValue = document.getElementById('global-temp-value');
        this.maxTokens = document.getElementById('max-tokens');

        // Memory viewer elements
        this.memoryCategoryFilter = document.getElementById('memory-category-filter');
        this.memorySearchInput = document.getElementById('memory-search-input');

        // Text upload panel elements
        this.textUploadPanel = document.getElementById('text-upload-panel');
        this.textUploadTitle = document.getElementById('text-upload-title');
        this.textUploadContent = document.getElementById('text-upload-content');
        this.textUploadCategory = document.getElementById('text-upload-category');
        this.textUploadSubmit = document.getElementById('text-upload-submit');
        this.textUploadCancel = document.getElementById('text-upload-cancel');
        this.textUploadClose = document.getElementById('text-upload-close');
        this.textUploadCharCount = document.getElementById('text-upload-char-count');

        this.setupEventListeners();
        this.startStatusUpdates();
        this.setupTextareaAutoResize();
        this.loadSettings();
    }

    setupEventListeners() {
        // Modal close buttons
        document.querySelectorAll('.modal-close').forEach(btn => {
            btn.addEventListener('click', () => this.closeAllModals());
        });

        // Close modal when clicking outside
        this.settingsModal.addEventListener('click', (e) => {
            if (e.target === this.settingsModal) {
                this.closeAllModals();
            }
        });

        // Header buttons
        document.getElementById('upload-btn').addEventListener('click', () => this.showFileDialog());
        document.getElementById('text-upload-btn').addEventListener('click', () => this.showTextUploadPanel());
        document.getElementById('new-conversation-btn').addEventListener('click', () => this.startNewConversation());
        document.getElementById('memory-btn').addEventListener('click', () => this.showMemoryViewer());
        document.getElementById('neural-mesh-btn').addEventListener('click', () => this.showNeuralMesh());
        document.getElementById('settings-btn').addEventListener('click', () => this.showSettings());

        // Memory viewer controls
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', (e) => this.switchMemoryTab(e.target));
        });

        this.memoryCategoryFilter.addEventListener('change', () => this.refreshMemoryContent());
        this.memorySearchInput.addEventListener('input', (e) => this.handleMemorySearch(e.target.value));

        // File input
        this.fileInput.addEventListener('change', (e) => this.handleFileSelection(e));

        // Upload modal
        this.uploadModal = document.getElementById('upload-modal');
        this.uploadTabs = document.querySelectorAll('.upload-tab-btn');
        this.uploadTabContents = document.querySelectorAll('.upload-tab-content');
        this.uploadCategorySelect = document.getElementById('upload-category-select');
        this.uploadSubmitBtn = document.getElementById('upload-submit-btn');
        this.uploadCancelBtn = document.getElementById('upload-cancel-btn');

        // Text input elements
        this.textTitleInput = document.getElementById('text-title-input');
        this.textContentInput = document.getElementById('text-content-input');
        this.textCharCount = document.getElementById('text-char-count');

        // File upload elements
        this.fileUploadArea = document.getElementById('file-upload-area');
        this.fileSelectBtn = document.getElementById('file-select-btn');
        this.selectedFiles = document.getElementById('selected-files');
        this.fileList = document.getElementById('file-list');

        // Setup upload modal event listeners
        this.setupUploadModal();

        // Setup text upload panel event listeners
        this.setupTextUploadPanel();

        // Input handling - Legacy (keep for compatibility)
        this.messageInput.addEventListener('keydown', (e) => this.handleInputKeydown(e));

        // Panel controls
        document.getElementById('memory-search-btn').addEventListener('click', () => this.showMemoryViewer());
        document.getElementById('memory-upload-btn').addEventListener('click', () => this.showFileDialog());
        document.getElementById('memory-stats-btn').addEventListener('click', () => this.showMemoryStats());
        document.getElementById('new-conversation-btn').addEventListener('click', () => this.startNewConversation());
        document.getElementById('clear-chat-btn').addEventListener('click', () => this.clearChat());
        document.getElementById('chat-settings-btn').addEventListener('click', () => this.showSettings());

        // Settings
        this.globalTemperature.addEventListener('input', () => {
            this.globalTempValue.textContent = this.globalTemperature.value;
        });

        document.getElementById('save-settings-btn').addEventListener('click', () => this.saveSettings());
        document.getElementById('reset-settings-btn').addEventListener('click', () => this.resetSettings());

        // Category selection
        this.categorySelect.addEventListener('change', () => this.updateCategoryFilter());
    }

    startStatusUpdates() {
        // Check connection every 5 seconds
        setInterval(() => this.updateConnectionStatus(), 5000);

        // Initial status check
        this.updateConnectionStatus();
        this.updateCategories();
    }

    async updateConnectionStatus() {
        const wasOnline = api.isOnline;

        try {
            const isOnline = await api.checkConnection();

            if (isOnline !== wasOnline) {
                this.updateStatusIndicator(isOnline);
                // Update both chat systems
                if (chat && chat.updateConnectionStatus) chat.updateConnectionStatus(isOnline);
                if (dualChat && dualChat.updateConnectionStatus) dualChat.updateConnectionStatus(isOnline);

                if (isOnline && !wasOnline) {
                    this.updateCategories();
                    this.showNotification('Connected to Frankenstino AI server!', 'success');
                } else if (!isOnline && wasOnline) {
                    this.showNotification('Lost connection to server', 'error');
                }
            }

            if (isOnline) {
                this.updateMemoryStats();
            }
        } catch (error) {
            console.warn('Status check failed:', error);
            this.updateStatusIndicator(false);
        }
    }

    updateStatusIndicator(isOnline) {
        if (isOnline) {
            this.statusIndicator.className = 'status-online';
            this.statusIndicator.textContent = '● Online';
        } else {
            this.statusIndicator.className = 'status-offline';
            this.statusIndicator.textContent = '● Offline';
        }
    }

    async updateMemoryStats() {
        try {
            const status = await api.getStatus();
            if (status.success && status.data?.memory_stats) {
                const stats = status.data.memory_stats;
                const docCount = stats.hierarchy?.total_documents || 0;
                const vectorCount = stats.vectors?.total_vectors || 0;
                const meshCount = stats.neural_mesh?.total_nodes || 0;

                this.memoryStats.textContent = `${docCount} docs, ${vectorCount} vectors, ${meshCount} nodes`;
            }
        } catch (error) {
            console.warn('Failed to update memory stats:', error);
            this.memoryStats.textContent = 'Loading...';
        }
    }

    async updateCategories() {
        try {
            console.log('🔄 Updating categories...');
            const categories = await api.getCategories();
            console.log('📂 Categories received:', categories);
            this.populateCategorySelect(categories);
        } catch (error) {
            console.warn('Failed to update categories:', error);
        }
    }

    populateCategorySelect(categories) {
        // Clear existing options except "All Categories"
        while (this.categorySelect.children.length > 1) {
            this.categorySelect.removeChild(this.categorySelect.lastChild);
        }

        // Add categories
        console.log('📝 Populating category select with:', categories);
        categories.forEach(category => {
            const option = document.createElement('option');
            option.value = category.name; // Use name instead of id for category filtering
            option.textContent = `${category.name} (${category.document_count || 0} docs)`;
            this.categorySelect.appendChild(option);
        });
    }

    showSettings() {
        this.showModal(this.settingsModal);
    }

    async showNeuralMesh() {
        const modal = document.getElementById('neural-mesh-modal');
        this.showModal(modal);
        await this.initializeNeuralMesh();
    }

    async initializeNeuralMesh() {
        const container = document.getElementById('neural-mesh-network');
        const nodeCountEl = document.getElementById('mesh-node-count');
        const edgeCountEl = document.getElementById('mesh-edge-count');

        try {
            // Clear any existing network
            if (this.network) {
                this.network.destroy();
            }

            // Show loading
            container.innerHTML = '<div style="display: flex; justify-content: center; align-items: center; height: 100%; color: #cccccc;">Loading neural mesh...</div>';

            // Fetch mesh data
            const response = await api.getNeuralMeshData();
            if (!response.success) {
                throw new Error('Failed to load neural mesh data');
            }

            const { nodes, edges, stats } = response.data;

            // Update stats
            nodeCountEl.textContent = `Nodes: ${stats.total_nodes}`;
            edgeCountEl.textContent = `Edges: ${stats.total_edges}`;

            // Create network
            const data = { nodes, edges };
            const options = {
                nodes: {
                    shape: 'dot',
                    font: {
                        size: 12,
                        color: '#e0e0e0'
                    },
                    borderWidth: 2,
                    shadow: true
                },
                edges: {
                    width: 2,
                    shadow: true,
                    smooth: {
                        type: 'continuous'
                    }
                },
                physics: {
                    enabled: true,
                    barnesHut: {
                        gravitationalConstant: -2000,
                        centralGravity: 0.3,
                        springLength: 95,
                        springConstant: 0.04,
                        damping: 0.09,
                        avoidOverlap: 0.1
                    },
                    stabilization: {
                        enabled: true,
                        iterations: 1000,
                        updateInterval: 25
                    }
                },
                interaction: {
                    dragNodes: true,
                    dragView: true,
                    zoomView: true,
                    hover: true,
                    selectConnectedEdges: false,
                    tooltipDelay: 300
                },
                layout: {
                    improvedLayout: true,
                    hierarchical: false
                }
            };

            // Clear loading message
            container.innerHTML = '';

            // Create network
            this.network = new vis.Network(container, data, options);

            // Add event listeners
            this.setupNeuralMeshControls();

            console.log('🕸️ Neural mesh visualization initialized with', nodes.length, 'nodes and', edges.length, 'edges');

        } catch (error) {
            console.error('Failed to initialize neural mesh:', error);
            container.innerHTML = '<div style="display: flex; justify-content: center; align-items: center; height: 100%; color: #dc3545;">Failed to load neural mesh</div>';
            nodeCountEl.textContent = 'Nodes: 0';
            edgeCountEl.textContent = 'Edges: 0';
        }
    }

    setupNeuralMeshControls() {
        const resetBtn = document.getElementById('reset-layout-btn');
        const physicsBtn = document.getElementById('physics-toggle-btn');
        const fullscreenBtn = document.getElementById('fullscreen-btn');

        if (resetBtn) {
            resetBtn.addEventListener('click', () => {
                if (this.network) {
                    this.network.fit();
                    this.network.stabilize();
                }
            });
        }

        if (physicsBtn) {
            let physicsEnabled = true;
            physicsBtn.addEventListener('click', () => {
                if (this.network) {
                    physicsEnabled = !physicsEnabled;
                    this.network.setOptions({
                        physics: { enabled: physicsEnabled }
                    });
                    physicsBtn.textContent = physicsEnabled ? '⏸️ Pause Physics' : '▶️ Start Physics';
                }
            });
        }

        if (fullscreenBtn) {
            fullscreenBtn.addEventListener('click', () => this.toggleFullscreen());
        }

        // Add delete large connections button
        const deleteBtn = document.getElementById('delete-large-connections-btn');
        if (deleteBtn) {
            deleteBtn.addEventListener('click', () => this.deleteLargeConnections());
        }

        // Add zoom controls
        this.addZoomControls();

        // Add filter controls
        this.addFilterControls();

        // Initialize filter state
        this.activeFilters = new Set(['active', 'short_term', 'long_term', 'archived']);
    }

    addZoomControls() {
        if (!this.network) return;

        // Create zoom control container
        const zoomControls = document.createElement('div');
        zoomControls.className = 'zoom-controls';
        zoomControls.innerHTML = `
            <button id="zoom-in-btn" class="zoom-btn" title="Zoom In">🔍+</button>
            <button id="zoom-out-btn" class="zoom-btn" title="Zoom Out">🔍-</button>
            <button id="zoom-fit-btn" class="zoom-btn" title="Fit to Screen">📐</button>
        `;

        // Add to controls
        const controlsContainer = document.querySelector('.neural-mesh-controls');
        if (controlsContainer) {
            controlsContainer.insertBefore(zoomControls, controlsContainer.firstChild);
        }

        // Add event listeners
        document.getElementById('zoom-in-btn').addEventListener('click', () => {
            if (this.network) {
                const scale = this.network.getScale();
                this.network.moveTo({ scale: scale * 1.2 });
            }
        });

        document.getElementById('zoom-out-btn').addEventListener('click', () => {
            if (this.network) {
                const scale = this.network.getScale();
                this.network.moveTo({ scale: scale * 0.8 });
            }
        });

        document.getElementById('zoom-fit-btn').addEventListener('click', () => {
            if (this.network) {
                this.network.fit();
            }
        });
    }

    toggleFullscreen() {
        const modal = document.getElementById('neural-mesh-modal');
        const isFullscreen = modal.classList.contains('fullscreen');

        if (isFullscreen) {
            modal.classList.remove('fullscreen');
            document.getElementById('fullscreen-btn').textContent = '⛶ Fullscreen';
        } else {
            modal.classList.add('fullscreen');
            document.getElementById('fullscreen-btn').textContent = '⛶ Exit Fullscreen';
        }

        // Re-fit the network after fullscreen change
        setTimeout(() => {
            if (this.network) {
                this.network.fit();
                // Also stabilize the physics
                this.network.stabilize();
            }
        }, 300);
    }

    async showMemoryViewer() {
        this.showModal(this.memoryModal);
        await this.loadMemoryContent();
    }

    async loadMemoryContent() {
        try {
            // Load categories for filter
            console.log('🔍 Loading categories for memory modal...');
            const categories = await api.getCategories();
            console.log('📂 Available categories:', categories);
            this.populateMemoryCategoryFilter(categories);

            // Load initial content (documents tab)
            await this.loadDocumentsTab();
        } catch (error) {
            console.error('Failed to load memory content:', error);
            this.showNotification('Failed to load memory content', 'error');
        }
    }

    populateMemoryCategoryFilter(categories) {
        // Clear existing options
        this.memoryCategoryFilter.innerHTML = '<option value="">All Categories</option>';

        // Add categories
        console.log('📝 Populating memory category filter with:', categories);
        categories.forEach(category => {
            const option = document.createElement('option');
            option.value = category.name; // Use name instead of id for category filtering
            // For general category, show actual conversation count
            let displayCount = category.document_count || 0;
            if (category.name === 'general') {
                // Count only conversation documents in general category
                displayCount = category.document_ids?.filter(id => id.startsWith('conv_')).length || 0;
            }
            option.textContent = `${category.name} (${displayCount})`;
            this.memoryCategoryFilter.appendChild(option);
        });
    }

    async switchMemoryTab(tabButton) {
        // Update tab buttons
        document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
        tabButton.classList.add('active');

        // Update tab content
        document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));

        const tabName = tabButton.dataset.tab;
        const tabContent = document.getElementById(`${tabName}-tab`);
        if (tabContent) {
            tabContent.classList.add('active');
        }

        // Load content for the selected tab
        switch (tabName) {
            case 'documents':
                await this.loadDocumentsTab();
                break;
            case 'conversations':
                await this.loadConversationsTab();
                break;
            case 'search':
                this.showSearchTab();
                break;
            case 'stats':
                await this.loadStatsTab();
                break;
        }
    }

    async loadDocumentsTab() {
        const category = this.memoryCategoryFilter.value;
        const container = document.getElementById('documents-list');

        try {
            container.innerHTML = '<div style="text-align: center; padding: 20px; color: #cccccc;">Loading documents...</div>';

            const result = await api.browseMemory({
                category: category || undefined,
                contentType: 'document',
                limit: 50
            });

            if (result.success) {
                this.displayMemoryItems(container, result.data.items, 'document');
            } else {
                container.innerHTML = '<div style="text-align: center; padding: 20px; color: #dc3545;">Failed to load documents</div>';
            }
        } catch (error) {
            console.error('Failed to load documents:', error);
            container.innerHTML = '<div style="text-align: center; padding: 20px; color: #dc3545;">Error loading documents</div>';
        }
    }

    async loadConversationsTab() {
        const category = this.memoryCategoryFilter.value;
        const container = document.getElementById('conversations-list');

        try {
            console.log('🔍 Loading conversations tab, category filter:', category);
            container.innerHTML = '<div style="text-align: center; padding: 20px; color: #cccccc;">Loading conversations...</div>';

            const result = await api.getConversations(category || undefined, 50);
            console.log('📋 Conversations API result:', result);

            if (result.success) {
                console.log(`✅ Found ${result.data.length} conversations:`, result.data);
                this.displayMemoryItems(container, result.data, 'conversation');
            } else {
                console.error('❌ Conversations API failed:', result);
                container.innerHTML = '<div style="text-align: center; padding: 20px; color: #dc3545;">Failed to load conversations</div>';
            }
        } catch (error) {
            console.error('❌ Failed to load conversations:', error);
            container.innerHTML = '<div style="text-align: center; padding: 20px; color: #dc3545;">Error loading conversations</div>';
        }
    }

    showSearchTab() {
        const searchInput = this.memorySearchInput;
        searchInput.style.display = 'block';
        searchInput.focus();
    }

    async handleMemorySearch(query) {
        const container = document.getElementById('search-results');

        if (!query.trim()) {
            container.innerHTML = '<div style="text-align: center; padding: 20px; color: #cccccc;">Enter a search query to find content in memory</div>';
            return;
        }

        try {
            container.innerHTML = '<div style="text-align: center; padding: 20px; color: #cccccc;">Searching...</div>';

            const result = await api.searchMemory(query, {
                category: this.memoryCategoryFilter.value || undefined,
                limit: 20
            });

            if (result.success) {
                this.displaySearchResults(container, result.data.results);
            } else {
                container.innerHTML = '<div style="text-align: center; padding: 20px; color: #dc3545;">Search failed</div>';
            }
        } catch (error) {
            console.error('Search failed:', error);
            container.innerHTML = '<div style="text-align: center; padding: 20px; color: #dc3545;">Search error</div>';
        }
    }

    async loadStatsTab() {
        const container = document.getElementById('memory-stats-content');

        try {
            container.innerHTML = '<div style="text-align: center; padding: 20px; color: #cccccc;">Loading statistics...</div>';

            const result = await api.getDetailedMemoryStats();

            if (result.success) {
                this.displayMemoryStats(container, result.data);
            } else {
                container.innerHTML = '<div style="text-align: center; padding: 20px; color: #dc3545;">Failed to load statistics</div>';
            }
        } catch (error) {
            console.error('Failed to load stats:', error);
            container.innerHTML = '<div style="text-align: center; padding: 20px; color: #dc3545;">Error loading statistics</div>';
        }
    }

    displayMemoryItems(container, items, type) {
        if (!items || items.length === 0) {
            container.innerHTML = `<div style="text-align: center; padding: 20px; color: #cccccc;">No ${type}s found</div>`;
            return;
        }

        container.innerHTML = '';

        items.forEach(item => {
            const itemDiv = document.createElement('div');
            itemDiv.className = 'memory-item';

            const lastAccessed = new Date(item.last_accessed * 1000).toLocaleDateString();
            const fileSize = item.file_size ? this.formatBytes(item.file_size) : '';

            itemDiv.innerHTML = `
                <div class="memory-item-header">
                    <div class="memory-item-title">${item.file_name || item.conversation_id || 'Unknown'}</div>
                    <div class="memory-item-meta">${lastAccessed}</div>
                </div>
                <div class="memory-item-preview">${item.preview || 'No preview available'}</div>
                <div class="memory-item-stats">
                    <span>Category: ${item.category || 'General'}</span>
                    ${fileSize ? `<span>Size: ${fileSize}</span>` : ''}
                    ${item.chunks ? `<span>Chunks: ${item.chunks}</span>` : ''}
                    ${item.message_count ? `<span>Messages: ${item.message_count}</span>` : ''}
                </div>
            `;

            container.appendChild(itemDiv);
        });
    }

    displaySearchResults(container, results) {
        if (!results || results.length === 0) {
            container.innerHTML = '<div style="text-align: center; padding: 20px; color: #cccccc;">No search results found</div>';
            return;
        }

        container.innerHTML = '';

        results.forEach(result => {
            const itemDiv = document.createElement('div');
            itemDiv.className = 'memory-item';

            const score = Math.round((result.score || 0) * 100);
            const metadata = result.metadata || {};

            itemDiv.innerHTML = `
                <div class="memory-item-header">
                    <div class="memory-item-title">${result.text.substring(0, 100)}...</div>
                    <div class="memory-item-meta">Relevance: ${score}%</div>
                </div>
                <div class="memory-item-preview">${result.text}</div>
                <div class="memory-item-stats">
                    <span>Type: ${metadata.content_type || 'Unknown'}</span>
                    <span>Category: ${metadata.category || 'General'}</span>
                    ${metadata.file_name ? `<span>File: ${metadata.file_name}</span>` : ''}
                </div>
            `;

            container.appendChild(itemDiv);
        });
    }

    displayMemoryStats(container, stats) {
        const hierarchy = stats.hierarchy || {};
        const vectors = stats.vectors || {};
        const neuralMesh = stats.neural_mesh || {};
        const conversations = stats.conversations || [];

        container.innerHTML = `
            <div class="stat-card">
                <h4>📚 Documents</h4>
                <div class="stat-value">${hierarchy.total_documents || 0}</div>
                <small>Total ingested documents</small>
            </div>
            <div class="stat-card">
                <h4>🔍 Vector Database</h4>
                <div class="stat-value">${vectors.total_vectors || 0}</div>
                <small>Embedded text chunks</small>
            </div>
            <div class="stat-card">
                <h4>🧠 Neural Mesh</h4>
                <div class="stat-value">${neuralMesh.total_nodes || 0}</div>
                <small>Connected memory nodes</small>
            </div>
            <div class="stat-card">
                <h4>💬 Conversations</h4>
                <div class="stat-value">${conversations.length}</div>
                <small>Stored chat sessions</small>
            </div>
            <div class="stat-card">
                <h4>📊 Memory Efficiency</h4>
                <div class="stat-value success">${stats.system_health?.memory_efficiency || 'Good'}</div>
                <small>Overall system health</small>
            </div>
            <div class="stat-card">
                <h4>⚡ Search Performance</h4>
                <div class="stat-value success">${stats.system_health?.search_performance || 'Optimal'}</div>
                <small>Query response speed</small>
            </div>
        `;
    }

    async refreshMemoryContent() {
        const activeTab = document.querySelector('.tab-btn.active');
        if (activeTab) {
            await this.switchMemoryTab(activeTab);
        }
    }

    startNewConversation() {
        if (chat && typeof chat.newConversation === 'function') {
            chat.newConversation();
            this.showNotification('Started new conversation', 'info');
        }
    }

    showFileDialog() {
        this.fileInput.click();
    }

    async handleFileSelection(event) {
        const files = Array.from(event.target.files);
        if (files.length === 0) return;

        this.showLoading();

        try {
            let successCount = 0;
            let errorCount = 0;

            for (const file of files) {
                try {
                    // Get selected category
                    const category = this.categorySelect.value === '' ? null : this.categorySelect.value;

                    // Upload file
                    const result = await api.uploadFile(file, category);

                    if (result.success) {
                        successCount++;
                        console.log(`Successfully ingested: ${file.name}`, result.data);
                    } else {
                        throw new Error(result.error || 'Upload failed');
                    }

                } catch (error) {
                    console.error(`Failed to upload ${file.name}:`, error);
                    this.showNotification(`Failed to upload ${file.name}: ${error.message}`, 'error');
                    errorCount++;
                }
            }

            if (successCount > 0) {
                this.showNotification(`Successfully ingested ${successCount} file${successCount > 1 ? 's' : ''}`, 'success');
                this.updateMemoryStats();
                this.updateCategories();
            }

            if (errorCount > 0) {
                this.showNotification(`Failed to upload ${errorCount} file${errorCount > 1 ? 's' : ''}`, 'warning');
            }

        } catch (error) {
            console.error('File upload error:', error);
            this.showNotification('File upload failed', 'error');
        } finally {
            this.hideLoading();
            // Clear file input
            this.fileInput.value = '';
        }
    }

    clearChat() {
        // Keep only the welcome message if it exists
        const messages = document.querySelectorAll('.message');
        messages.forEach(message => {
            if (!message.querySelector('.welcome-message')) {
                message.remove();
            }
        });
        // Clear chat history
        if (chat && typeof chat.clearChatHistory === 'function') {
            chat.clearChatHistory();
        }
    }

    updateCategoryFilter() {
        const selectedCategory = this.categorySelect.value;
        // Update chat to use new category
        if (chat && chat.updateCategory) {
            chat.updateCategory(selectedCategory);
        }
    }

    setupTextareaAutoResize() {
        this.messageInput.addEventListener('input', () => {
            this.messageInput.style.height = 'auto';
            this.messageInput.style.height = Math.min(this.messageInput.scrollHeight, 200) + 'px';
        });
    }

    handleInputKeydown(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            if (!chat.isProcessing && api.isOnline && this.messageInput.value.trim()) {
                chat.sendMessage();
            }
        }
    }

    saveSettings() {
        const settings = {
            temperature: parseFloat(this.globalTemperature.value),
            maxTokens: this.maxTokens.value
        };

        localStorage.setItem('frankenstino_settings', JSON.stringify(settings));
        this.showNotification('Settings saved successfully', 'success');
        this.closeAllModals();
    }

    resetSettings() {
        localStorage.removeItem('frankenstino_settings');
        this.loadSettings();
        this.showNotification('Settings reset to defaults', 'info');
    }

    loadSettings() {
        const saved = localStorage.getItem('frankenstino_settings');
        if (saved) {
            try {
                const settings = JSON.parse(saved);
                this.globalTemperature.value = settings.temperature || 0.7;
                this.globalTempValue.textContent = this.globalTemperature.value;
                this.maxTokens.value = settings.maxTokens || '512';
            } catch (error) {
                console.warn('Failed to load settings:', error);
            }
        }
    }

    showModal(modal) {
        modal.style.display = 'flex';
        document.body.style.overflow = 'hidden';
    }

    closeAllModals() {
        this.settingsModal.style.display = 'none';
        this.memoryModal.style.display = 'none';
        this.closeNeuralMeshModal();
        this.closeTextUploadPanel();
        document.body.style.overflow = 'auto';
    }

    closeNeuralMeshModal() {
        const modal = document.getElementById('neural-mesh-modal');
        modal.style.display = 'none';
        modal.classList.remove('fullscreen');
        document.body.style.overflow = 'auto';

        // Destroy network instance if it exists
        if (this.network) {
            this.network.destroy();
            this.network = null;
        }
    }

    showLoading() {
        this.loadingOverlay.style.display = 'flex';
    }

    hideLoading() {
        this.loadingOverlay.style.display = 'none';
    }

    showNotification(message, type = 'info', duration = 5000) {
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.innerHTML = `
            <span>${message}</span>
            <button class="notification-close">&times;</button>
        `;

        this.notificationContainer.appendChild(notification);

        const closeBtn = notification.querySelector('.notification-close');
        closeBtn.addEventListener('click', () => this.removeNotification(notification));

        // Auto-remove after duration
        setTimeout(() => this.removeNotification(notification), duration);

        // Animate in
        setTimeout(() => notification.classList.add('show'), 10);
    }

    removeNotification(notification) {
        notification.classList.remove('show');
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 300);
    }

    formatBytes(bytes) {
        if (bytes === 0) return '0 B';
        const k = 1024;
        const sizes = ['B', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    addFilterControls() {
        // Add event listeners for filter buttons
        const filterButtons = [
            'filter-active-btn',
            'filter-short-btn',
            'filter-long-btn',
            'filter-archived-btn'
        ];

        filterButtons.forEach(btnId => {
            const button = document.getElementById(btnId);
            if (button) {
                button.addEventListener('click', (e) => this.toggleMemoryTierFilter(e.target));
            }
        });
    }

    toggleMemoryTierFilter(button) {
        const tier = button.dataset.tier;
        const isActive = button.classList.contains('active');

        if (isActive) {
            // Deactivate filter
            button.classList.remove('active');
            button.classList.add('inactive');
            this.activeFilters.delete(tier);
        } else {
            // Activate filter
            button.classList.remove('inactive');
            button.classList.add('active');
            this.activeFilters.add(tier);
        }

        // Apply filters to the network
        this.applyNetworkFilters();
    }

    applyNetworkFilters() {
        if (!this.network) return;

        // Get all nodes
        const allNodes = this.network.body.data.nodes.get();

        // Filter nodes based on active filters
        const visibleNodes = allNodes.filter(node => {
            const nodeTier = node.tier || 'active'; // Default to active if no tier
            return this.activeFilters.has(nodeTier);
        });

        const visibleNodeIds = new Set(visibleNodes.map(node => node.id));

        // Get edges that connect visible nodes
        const allEdges = this.network.body.data.edges.get();
        const visibleEdges = allEdges.filter(edge => {
            return visibleNodeIds.has(edge.from) && visibleNodeIds.has(edge.to);
        });

        // Update network visibility
        this.network.body.data.nodes.update(allNodes.map(node => ({
            ...node,
            hidden: !visibleNodeIds.has(node.id)
        })));

        this.network.body.data.edges.update(allEdges.map(edge => ({
            ...edge,
            hidden: !visibleEdges.some(visEdge => visEdge.id === edge.id)
        })));

        // Update stats
        const nodeCountEl = document.getElementById('mesh-node-count');
        const edgeCountEl = document.getElementById('mesh-edge-count');

        nodeCountEl.textContent = `Nodes: ${visibleNodes.length}`;
        edgeCountEl.textContent = `Edges: ${visibleEdges.length}`;

        // Fit the view to show visible nodes
        setTimeout(() => {
            if (this.network && visibleNodes.length > 0) {
                this.network.fit({
                    nodes: visibleNodes.map(n => n.id),
                    animation: { duration: 500 }
                });
            }
        }, 100);
    }

    deleteLargeConnections() {
        // Prompt user for size threshold
        const threshold = prompt('Enter minimum connection size to delete (0.0-1.0):', '0.8');
        if (threshold === null || isNaN(parseFloat(threshold))) {
            return;
        }

        const minSize = parseFloat(threshold);
        if (minSize < 0 || minSize > 1) {
            alert('Size must be between 0.0 and 1.0');
            return;
        }

        // Confirm deletion
        if (!confirm(`Delete all connections with size >= ${minSize}? This cannot be undone.`)) {
            return;
        }

        // Send delete request to backend
        fetch('/api/autonomous/delete_connections', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ min_size: minSize })
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                alert(`Deleted ${data.deleted_count} large connections`);
                // Reload the visualization
                this.initializeNeuralMesh();
            } else {
                alert('Failed to delete connections: ' + (data.error || 'Unknown error'));
            }
        })
        .catch(error => {
            console.error('Failed to delete connections:', error);
            alert('Failed to delete connections: ' + error.message);
        });
    }

    setupUploadModal() {
        // Tab switching
        this.uploadTabs.forEach(tab => {
            tab.addEventListener('click', () => this.switchUploadTab(tab));
        });

        // Text input validation
        this.textContentInput.addEventListener('input', () => this.updateTextValidation());
        this.textTitleInput.addEventListener('input', () => this.updateUploadButtonState());

        // File upload area
        this.fileUploadArea.addEventListener('click', () => this.fileSelectBtn.click());
        this.fileUploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            this.fileUploadArea.classList.add('dragover');
        });
        this.fileUploadArea.addEventListener('dragleave', () => {
            this.fileUploadArea.classList.remove('dragover');
        });
        this.fileUploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            this.fileUploadArea.classList.remove('dragover');
            this.handleFileDrop(e.dataTransfer.files);
        });

        this.fileSelectBtn.addEventListener('click', () => this.fileInput.click());

        // Upload buttons
        this.uploadSubmitBtn.addEventListener('click', () => this.handleUpload());
        this.uploadCancelBtn.addEventListener('click', () => this.closeUploadModal());

        // Modal close
        this.uploadModal.querySelector('.modal-close').addEventListener('click', () => this.closeUploadModal());
        this.uploadModal.addEventListener('click', (e) => {
            if (e.target === this.uploadModal) {
                this.closeUploadModal();
            }
        });

        // Load categories for upload modal
        this.updateUploadCategories();
    }

    switchUploadTab(tab) {
        // Update tab buttons
        this.uploadTabs.forEach(t => t.classList.remove('active'));
        tab.classList.add('active');

        // Update tab content
        this.uploadTabContents.forEach(content => content.classList.remove('active'));
        const tabName = tab.dataset.tab;
        const content = document.getElementById(`${tabName}-tab`);
        if (content) {
            content.classList.add('active');
        }

        // Update button state
        this.updateUploadButtonState();
    }

    updateTextValidation() {
        const text = this.textContentInput.value;
        const charCount = text.length;
        const maxChars = 100000;

        // Update character counter
        this.textCharCount.textContent = `${charCount} / ${maxChars}`;

        // Update counter color
        this.textCharCount.classList.remove('warning', 'error');
        if (charCount > maxChars * 0.9) {
            this.textCharCount.classList.add('warning');
        }
        if (charCount > maxChars) {
            this.textCharCount.classList.add('error');
        }

        // Update button state
        this.updateUploadButtonState();
    }

    updateUploadButtonState() {
        const activeTab = document.querySelector('.upload-tab-btn.active');
        let isValid = false;

        if (activeTab.dataset.tab === 'files') {
            // File tab: check if files are selected
            isValid = this.selectedFiles && this.selectedFiles.children.length > 0;
        } else if (activeTab.dataset.tab === 'text') {
            // Text tab: check if text content is provided and not over limit
            const text = this.textContentInput.value.trim();
            isValid = text.length > 0 && text.length <= 100000;
        }

        this.uploadSubmitBtn.disabled = !isValid;
        this.uploadSubmitBtn.classList.toggle('disabled', !isValid);
    }

    handleFileDrop(files) {
        this.handleFileSelection({ target: { files } });
    }

    handleFileSelection(event) {
        const files = Array.from(event.target.files);
        if (files.length === 0) return;

        this.selectedFiles.innerHTML = '';
        this.fileList.style.display = 'block';

        files.forEach((file, index) => {
            const fileItem = document.createElement('div');
            fileItem.className = 'file-item';
            fileItem.innerHTML = `
                <div class="file-info">
                    <div class="file-icon">${this.getFileIcon(file.type)}</div>
                    <div class="file-details">
                        <h5>${file.name}</h5>
                        <small>${this.formatBytes(file.size)} • ${file.type || 'Unknown type'}</small>
                    </div>
                </div>
                <button class="file-remove" data-index="${index}" title="Remove file">&times;</button>
            `;

            // Add remove handler
            fileItem.querySelector('.file-remove').addEventListener('click', () => {
                fileItem.remove();
                this.updateUploadButtonState();
                if (this.selectedFiles.children.length === 0) {
                    this.fileList.style.display = 'none';
                }
            });

            this.selectedFiles.appendChild(fileItem);
        });

        this.updateUploadButtonState();
    }

    getFileIcon(mimeType) {
        if (mimeType.startsWith('text/')) return '📄';
        if (mimeType.includes('pdf')) return '📕';
        if (mimeType.includes('word') || mimeType.includes('document')) return '📝';
        if (mimeType.includes('spreadsheet') || mimeType.includes('excel')) return '📊';
        if (mimeType.includes('presentation') || mimeType.includes('powerpoint')) return '📽️';
        if (mimeType.includes('image/')) return '🖼️';
        if (mimeType.includes('audio/')) return '🎵';
        if (mimeType.includes('video/')) return '🎥';
        return '📁';
    }

    async handleUpload() {
        const activeTab = document.querySelector('.upload-tab-btn.active');
        const category = this.uploadCategorySelect.value;

        this.showLoading();

        try {
            if (activeTab.dataset.tab === 'files') {
                await this.uploadFiles(category);
            } else if (activeTab.dataset.tab === 'text') {
                await this.uploadText(category);
            }
        } catch (error) {
            console.error('Upload failed:', error);
            this.showNotification('Upload failed: ' + error.message, 'error');
        } finally {
            this.hideLoading();
        }
    }

    async uploadFiles(category) {
        const fileItems = Array.from(this.selectedFiles.children);
        let successCount = 0;
        let errorCount = 0;

        for (const fileItem of fileItems) {
            try {
                const fileName = fileItem.querySelector('h5').textContent;
                // Find the original file object (this is a limitation - we'd need to store file objects)
                // For now, we'll show a message that file upload from modal needs file objects
                throw new Error('File upload from modal requires file objects. Please use the main upload button.');
            } catch (error) {
                console.error(`Failed to upload ${fileName}:`, error);
                errorCount++;
            }
        }

        if (successCount > 0) {
            this.showNotification(`Successfully uploaded ${successCount} file(s)`, 'success');
            this.closeUploadModal();
            this.updateMemoryStats();
            this.updateCategories();
        }

        if (errorCount > 0) {
            this.showNotification(`Failed to upload ${errorCount} file(s)`, 'error');
        }
    }

    async uploadText(category) {
        const title = this.textTitleInput.value.trim();
        const text = this.textContentInput.value.trim();

        try {
            const result = await api.uploadText(text, title, category);

            if (result.success) {
                this.showNotification('Text document uploaded successfully!', 'success');
                this.closeUploadModal();
                this.resetTextForm();
                this.updateMemoryStats();
                this.updateCategories();
            } else {
                throw new Error(result.error || 'Upload failed');
            }
        } catch (error) {
            console.error('Text upload failed:', error);
            throw error;
        }
    }

    resetTextForm() {
        this.textTitleInput.value = '';
        this.textContentInput.value = '';
        this.updateTextValidation();
    }

    closeUploadModal() {
        this.uploadModal.style.display = 'none';
        document.body.style.overflow = 'auto';
        this.resetUploadModal();
    }

    resetUploadModal() {
        // Reset to files tab
        this.switchUploadTab(this.uploadTabs[0]);

        // Clear file selection
        this.selectedFiles.innerHTML = '';
        this.fileList.style.display = 'none';

        // Clear text form
        this.resetTextForm();

        // Reset category
        this.uploadCategorySelect.value = '';

        // Reset button state
        this.updateUploadButtonState();
    }

    async updateUploadCategories() {
        try {
            const categories = await api.getCategories();
            this.uploadCategorySelect.innerHTML = '<option value="">General</option>';

            categories.forEach(category => {
                const option = document.createElement('option');
                option.value = category.name;
                option.textContent = category.name;
                this.uploadCategorySelect.appendChild(option);
            });
        } catch (error) {
            console.warn('Failed to load upload categories:', error);
        }
    }

    showTextUploadPanel() {
        this.textUploadPanel.classList.remove('hidden');
        document.body.style.overflow = 'hidden';
        this.updateTextUploadCategories();
        this.textUploadContent.focus();
    }

    setupTextUploadPanel() {
        // Text input validation
        this.textUploadContent.addEventListener('input', () => this.updateTextUploadValidation());

        // Button handlers
        this.textUploadSubmit.addEventListener('click', () => this.handleTextUpload());
        this.textUploadCancel.addEventListener('click', () => this.closeTextUploadPanel());
        this.textUploadClose.addEventListener('click', () => this.closeTextUploadPanel());

        // Close on background click
        this.textUploadPanel.addEventListener('click', (e) => {
            if (e.target === this.textUploadPanel) {
                this.closeTextUploadPanel();
            }
        });

        // ESC key to close
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && !this.textUploadPanel.classList.contains('hidden')) {
                this.closeTextUploadPanel();
            }
        });
    }

    updateTextUploadValidation() {
        const text = this.textUploadContent.value;
        const charCount = text.length;
        const maxChars = 100000;

        // Update character counter
        this.textUploadCharCount.textContent = `${charCount} / ${maxChars}`;

        // Update counter color
        this.textUploadCharCount.classList.remove('warning', 'error');
        if (charCount > maxChars * 0.9) {
            this.textUploadCharCount.classList.add('warning');
        }
        if (charCount > maxChars) {
            this.textUploadCharCount.classList.add('error');
        }

        // Update button state
        const isValid = text.trim().length > 0 && text.length <= maxChars;
        this.textUploadSubmit.disabled = !isValid;
        this.textUploadSubmit.classList.toggle('disabled', !isValid);
    }

    async handleTextUpload() {
        const title = this.textUploadTitle.value.trim();
        const text = this.textUploadContent.value.trim();
        const category = this.textUploadCategory.value;

        if (!text) {
            this.showNotification('Please enter some text content', 'warning');
            return;
        }

        if (text.length > 100000) {
            this.showNotification('Text content is too long (max 100,000 characters)', 'error');
            return;
        }

        this.showLoading();

        try {
            const result = await api.uploadText(text, title, category);

            if (result.success) {
                this.showNotification('Text document uploaded successfully!', 'success');
                this.closeTextUploadPanel();
                this.resetTextUploadForm();
                this.updateMemoryStats();
                this.updateCategories();
            } else {
                throw new Error(result.error || 'Upload failed');
            }
        } catch (error) {
            console.error('Text upload failed:', error);
            this.showNotification('Text upload failed: ' + error.message, 'error');
        } finally {
            this.hideLoading();
        }
    }

    closeTextUploadPanel() {
        this.textUploadPanel.classList.add('hidden');
        document.body.style.overflow = 'auto';
        this.resetTextUploadForm();
    }

    resetTextUploadForm() {
        this.textUploadTitle.value = '';
        this.textUploadContent.value = '';
        this.textUploadCategory.value = '';
        this.updateTextUploadValidation();
    }

    async updateTextUploadCategories() {
        try {
            const categories = await api.getCategories();
            this.textUploadCategory.innerHTML = '<option value="">General</option>';

            categories.forEach(category => {
                const option = document.createElement('option');
                option.value = category.name;
                option.textContent = category.name;
                this.textUploadCategory.appendChild(option);
            });
        } catch (error) {
            console.warn('Failed to load text upload categories:', error);
        }
    }

    formatNumber(num) {
        return new Intl.NumberFormat().format(num);
    }
}

// Create global UI instance
const ui = new UIUtils();
