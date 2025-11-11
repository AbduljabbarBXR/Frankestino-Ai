/**
 * Autonomous Neural Mesh Visualizer
 * Real-time visualization of word association network learning
 */
class AutonomousMeshVisualizer {
    constructor(containerId) {
        this.containerId = containerId;
        this.network = null;
        this.nodes = new vis.DataSet();
        this.edges = new vis.DataSet();
        this.wordNodes = new Map();
        this.connectionAnimations = new Map();
        this.learningStats = {};

        // Configuration
        this.config = {
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
                hover: true,
                tooltipDelay: 300,
                zoomView: true,
                dragView: true
            },
            layout: {
                randomSeed: 42
            }
        };

        // Animation settings
        this.animationDuration = 1000;
        this.pulseInterval = 2000;

        // WebSocket connection - DISABLED for static visualization
        // this.ws = null;
        // this.reconnectAttempts = 0;
        // this.maxReconnectAttempts = 5;

        // UI elements
        this.controlsContainer = null;
        this.statsContainer = null;

        this.initialize();
    }

    async initialize() {
        try {
            // Set up the network container
            const container = document.getElementById(this.containerId);
            if (!container) {
                throw new Error(`Container element '${this.containerId}' not found`);
            }

            // Create network
            this.network = new vis.Network(container, {
                nodes: this.nodes,
                edges: this.edges
            }, this.config);

            // Set up event handlers
            this.setupEventHandlers();

            // Create UI controls
            this.createControls();

            // Connect to backend - DISABLED for static visualization
            // this.connectWebSocket();

            // Load initial data
            await this.loadInitialData();

            // Start animations - DISABLED for static visualization
            // this.startPulseAnimation();

            console.log('Autonomous Mesh Visualizer initialized');

        } catch (error) {
            console.error('Failed to initialize Autonomous Mesh Visualizer:', error);
            this.showError('Failed to initialize visualizer: ' + error.message);
        }
    }

    setupEventHandlers() {
        // Node hover events
        this.network.on('hoverNode', (params) => {
            this.handleNodeHover(params.node, true);
        });

        this.network.on('blurNode', (params) => {
            this.handleNodeHover(params.node, false);
        });

        // Edge hover events
        this.network.on('hoverEdge', (params) => {
            this.handleEdgeHover(params.edge, true);
        });

        this.network.on('blurEdge', (params) => {
            this.handleEdgeHover(params.edge, false);
        });

        // Click events
        this.network.on('click', (params) => {
            if (params.nodes.length > 0) {
                this.handleNodeClick(params.nodes[0]);
            } else if (params.edges.length > 0) {
                this.handleEdgeClick(params.edges[0]);
            }
        });

        // Stabilization events
        this.network.on('stabilized', () => {
            console.log('Network stabilized');
        });
    }

    createControls() {
        // Create controls container
        this.controlsContainer = document.createElement('div');
        this.controlsContainer.className = 'neural-mesh-controls';
        this.controlsContainer.innerHTML = `
            <div class="zoom-controls">
                <button id="zoom-in-btn" class="zoom-btn" title="Zoom In">🔍+</button>
                <button id="zoom-out-btn" class="zoom-btn" title="Zoom Out">🔍-</button>
                <button id="zoom-fit-btn" class="zoom-btn" title="Fit to Screen">📐</button>
            </div>
            <div class="filter-controls">
                <button id="filter-active-btn" class="filter-btn active" data-filter="active">🟢 Active</button>
                <button id="filter-recent-btn" class="filter-btn active" data-filter="recent">🔵 Recent</button>
                <button id="filter-semantic-btn" class="filter-btn active" data-filter="semantic">🟡 Semantic</button>
                <button id="filter-associative-btn" class="filter-btn active" data-filter="associative">⚪ Associative</button>
            </div>
            <div class="control-buttons">
                <button id="learning-toggle-btn" class="btn">⏸️ Pause Learning</button>
                <button id="reset-layout-btn" class="btn">🔄 Reset Layout</button>
                <button id="physics-toggle-btn" class="btn">⏸️ Pause Physics</button>
                <button id="delete-large-connections-btn" class="btn">🗑️ Delete Large Connections</button>
            </div>
            <div class="mesh-stats">
                <span id="node-count">Nodes: 0</span>
                <span id="edge-count">Edges: 0</span>
                <span id="learning-status">Learning: Active</span>
            </div>
        `;

        // Insert controls into the network container
        const container = document.getElementById(this.containerId);
        container.style.position = 'relative';
        container.appendChild(this.controlsContainer);

        // Set up control event handlers
        this.setupControlHandlers();
    }

    setupControlHandlers() {
        // Zoom controls
        document.getElementById('zoom-in-btn').onclick = () => {
            const scale = this.network.getScale();
            this.network.moveTo({ scale: scale * 1.2 });
        };

        document.getElementById('zoom-out-btn').onclick = () => {
            const scale = this.network.getScale();
            this.network.moveTo({ scale: scale / 1.2 });
        };

        document.getElementById('zoom-fit-btn').onclick = () => {
            this.network.fit();
        };

        // Filter controls
        document.querySelectorAll('.filter-btn').forEach(btn => {
            btn.onclick = () => {
                btn.classList.toggle('active');
                this.updateFilters();
            };
        });

        // Control buttons
        document.getElementById('learning-toggle-btn').onclick = (e) => {
            this.toggleLearning(e.target);
        };

        document.getElementById('reset-layout-btn').onclick = () => {
            this.resetLayout();
        };

        document.getElementById('physics-toggle-btn').onclick = (e) => {
            this.togglePhysics(e.target);
        };

        document.getElementById('delete-large-connections-btn').onclick = () => {
            this.deleteLargeConnections();
        };
    }

    connectWebSocket() {
        try {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/api/autonomous/updates`;

            this.ws = new WebSocket(wsUrl);

            this.ws.onopen = () => {
                console.log('Connected to autonomous mesh updates');
                this.reconnectAttempts = 0;
                this.updateConnectionStatus(true);
            };

            this.ws.onmessage = (event) => {
                try {
                    const update = JSON.parse(event.data);
                    this.handleNetworkUpdate(update);
                } catch (error) {
                    console.error('Failed to parse WebSocket message:', error);
                }
            };

            this.ws.onclose = () => {
                console.log('Disconnected from autonomous mesh updates');
                this.updateConnectionStatus(false);
                this.attemptReconnect();
            };

            this.ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.updateConnectionStatus(false);
            };

        } catch (error) {
            console.error('Failed to connect WebSocket:', error);
            this.updateConnectionStatus(false);
        }
    }

    attemptReconnect() {
        if (this.reconnectAttempts >= this.maxReconnectAttempts) {
            console.error('Max reconnection attempts reached');
            return;
        }

        this.reconnectAttempts++;
        console.log(`Attempting to reconnect (${this.reconnectAttempts}/${this.maxReconnectAttempts})...`);

        setTimeout(() => {
            this.connectWebSocket();
        }, 2000 * this.reconnectAttempts); // Exponential backoff
    }

    async loadInitialData() {
        try {
            const response = await fetch('/api/autonomous/visualization');
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();
            this.updateNetworkData(data);

        } catch (error) {
            console.error('Failed to load initial network data:', error);
            this.showError('Failed to load network data: ' + error.message);
        }
    }

    updateNetworkData(data) {
        if (!data.nodes || !data.edges) {
            console.warn('Invalid network data received');
            return;
        }

        // Update nodes
        const nodeUpdates = data.nodes.map(node => ({
            id: node.id,
            label: node.label || node.id,
            title: this.createNodeTooltip(node),
            color: this.getNodeColor(node),
            size: this.getNodeSize(node),
            font: { size: 12, color: '#ffffff' },
            shape: 'dot',
            group: node.type || 'unknown'
        }));

        this.nodes.update(nodeUpdates);

        // Update edges
        const edgeUpdates = data.edges.map(edge => ({
            id: `${edge.from}-${edge.to}`,
            from: edge.from,
            to: edge.to,
            title: this.createEdgeTooltip(edge),
            color: this.getEdgeColor(edge),
            width: this.getEdgeWidth(edge),
            arrows: edge.type === 'directed' ? 'to' : undefined,
            smooth: { type: 'continuous' }
        }));

        this.edges.update(edgeUpdates);

        // Update statistics
        this.updateStats(data.stats);
    }

    handleNetworkUpdate(update) {
        switch (update.type) {
            case 'node_added':
                this.animateNodeAddition(update.node);
                break;
            case 'edge_added':
                this.animateEdgeAddition(update.edge);
                break;
            case 'node_updated':
                this.animateNodeUpdate(update.node);
                break;
            case 'edge_updated':
                this.animateEdgeUpdate(update.edge);
                break;
            case 'stats_updated':
                this.updateStats(update.stats);
                break;
            case 'learning_event':
                this.handleLearningEvent(update.event);
                break;
            default:
                console.warn('Unknown update type:', update.type);
        }
    }

    animateNodeAddition(node) {
        const nodeData = {
            id: node.id,
            label: node.word || node.id,
            title: this.createNodeTooltip(node),
            color: { background: '#00ff00', border: '#00aa00' }, // Green for new
            size: 15,
            font: { size: 12, color: '#ffffff' },
            shape: 'dot',
            group: 'word'
        };

        this.nodes.add(nodeData);

        // Animate to normal color
        setTimeout(() => {
            this.nodes.update({
                id: node.id,
                color: this.getNodeColor(node)
            });
        }, this.animationDuration);
    }

    animateEdgeAddition(edge) {
        const edgeData = {
            id: `${edge.from}-${edge.to}`,
            from: edge.from,
            to: edge.to,
            title: this.createEdgeTooltip(edge),
            color: { color: '#00ff00', highlight: '#00aa00' }, // Green for new
            width: 2,
            smooth: { type: 'continuous' }
        };

        this.edges.add(edgeData);

        // Animate to normal color and width
        setTimeout(() => {
            this.edges.update({
                id: edgeData.id,
                color: this.getEdgeColor(edge),
                width: this.getEdgeWidth(edge)
            });
        }, this.animationDuration);
    }

    animateNodeUpdate(node) {
        this.nodes.update({
            id: node.id,
            color: this.getNodeColor(node),
            size: this.getNodeSize(node)
        });
    }

    animateEdgeUpdate(edge) {
        this.edges.update({
            id: `${edge.from}-${edge.to}`,
            color: this.getEdgeColor(edge),
            width: this.getEdgeWidth(edge)
        });
    }

    handleLearningEvent(event) {
        // Highlight learning activity
        if (event.event_type === 'connection_formed') {
            this.highlightLearningActivity(event.neuron_a, event.neuron_b);
        }
    }

    highlightLearningActivity(nodeId1, nodeId2) {
        // Temporarily highlight the learning nodes
        const originalColors = {};

        [nodeId1, nodeId2].forEach(nodeId => {
            try {
                const node = this.nodes.get(nodeId);
                if (node) {
                    originalColors[nodeId] = node.color;
                    this.nodes.update({
                        id: nodeId,
                        color: { background: '#ffff00', border: '#aaaa00' } // Yellow highlight
                    });
                }
            } catch (error) {
                // Node might not exist yet
            }
        });

        // Restore original colors after animation
        setTimeout(() => {
            Object.entries(originalColors).forEach(([nodeId, color]) => {
                this.nodes.update({ id: nodeId, color });
            });
        }, this.animationDuration);
    }

    startPulseAnimation() {
        setInterval(() => {
            // Pulse active learning nodes
            if (this.learningStats.scanning_active) {
                this.pulseActiveNodes();
            }
        }, this.pulseInterval);
    }

    pulseActiveNodes() {
        // Find nodes that have been recently active
        const now = Date.now() / 1000;
        const recentThreshold = 60; // 1 minute

        this.nodes.forEach(node => {
            if (node.last_accessed && (now - node.last_accessed) < recentThreshold) {
                // Temporarily increase size
                const originalSize = node.size || 10;
                this.nodes.update({
                    id: node.id,
                    size: originalSize * 1.2
                });

                // Restore size
                setTimeout(() => {
                    this.nodes.update({
                        id: node.id,
                        size: originalSize
                    });
                }, this.pulseInterval / 2);
            }
        });
    }

    getNodeColor(node) {
        if (node.type === 'word') {
            // Color based on activation level
            const activation = node.activation || 0;
            if (activation > 0.8) {
                return { background: '#ff4444', border: '#aa0000' }; // Red for highly active
            } else if (activation > 0.5) {
                return { background: '#ffaa44', border: '#aa5500' }; // Orange for active
            } else {
                return { background: '#4444ff', border: '#0000aa' }; // Blue for inactive
            }
        } else {
            return { background: '#666666', border: '#333333' }; // Gray for memory nodes
        }
    }

    getNodeSize(node) {
        const baseSize = 10;
        const activation = node.activation || 0;
        const frequency = node.frequency || 0;

        // Size based on activation and frequency
        return baseSize + (activation * 10) + Math.min(frequency / 10, 10);
    }

    getEdgeColor(edge) {
        const weight = edge.weight || 0;

        if (edge.type === 'semantic') {
            return { color: '#00ff00', highlight: '#00aa00' }; // Green for semantic
        } else if (edge.type === 'syntactic') {
            return { color: '#ffff00', highlight: '#aaaa00' }; // Yellow for syntactic
        } else if (edge.type === 'associative') {
            return { color: '#ff8800', highlight: '#aa4400' }; // Orange for associative
        } else {
            return { color: '#ffffff', highlight: '#aaaaaa' }; // White for other
        }
    }

    getEdgeWidth(edge) {
        const weight = edge.weight || 0;
        return Math.max(1, Math.min(weight * 5, 10));
    }

    createNodeTooltip(node) {
        let tooltip = `<b>${node.label || node.id}</b><br>`;
        tooltip += `Type: ${node.type || 'unknown'}<br>`;

        if (node.activation !== undefined) {
            tooltip += `Activation: ${(node.activation * 100).toFixed(1)}%<br>`;
        }

        if (node.frequency !== undefined) {
            tooltip += `Frequency: ${node.frequency}<br>`;
        }

        if (node.last_accessed) {
            const timeAgo = Math.floor((Date.now() / 1000 - node.last_accessed) / 60);
            tooltip += `Last active: ${timeAgo}min ago<br>`;
        }

        return tooltip;
    }

    createEdgeTooltip(edge) {
        let tooltip = `<b>Connection</b><br>`;
        tooltip += `Type: ${edge.type || 'unknown'}<br>`;
        tooltip += `Weight: ${(edge.weight * 100).toFixed(1)}%<br>`;

        if (edge.label) {
            tooltip += `Label: ${edge.label}<br>`;
        }

        return tooltip;
    }

    handleNodeHover(nodeId, isHover) {
        try {
            const node = this.nodes.get(nodeId);
            if (node) {
                const originalSize = node.size || 10;
                this.nodes.update({
                    id: nodeId,
                    size: isHover ? originalSize * 1.3 : originalSize
                });
            }
        } catch (error) {
            // Node might not exist
        }
    }

    handleEdgeHover(edgeId, isHover) {
        try {
            const edge = this.edges.get(edgeId);
            if (edge) {
                const originalWidth = edge.width || 1;
                this.edges.update({
                    id: edgeId,
                    width: isHover ? originalWidth * 1.5 : originalWidth
                });
            }
        } catch (error) {
            // Edge might not exist
        }
    }

    handleNodeClick(nodeId) {
        // Show detailed information about the node
        this.showNodeDetails(nodeId);
    }

    handleEdgeClick(edgeId) {
        // Show detailed information about the connection
        this.showEdgeDetails(edgeId);
    }

    showNodeDetails(nodeId) {
        // Implementation for showing node details modal
        console.log('Showing details for node:', nodeId);
        // TODO: Implement node details modal
    }

    showEdgeDetails(edgeId) {
        // Implementation for showing edge details modal
        console.log('Showing details for edge:', edgeId);
        // TODO: Implement edge details modal
    }

    updateFilters() {
        // Get active filters
        const activeFilters = Array.from(document.querySelectorAll('.filter-btn.active'))
            .map(btn => btn.dataset.filter);

        // Update node visibility
        this.nodes.forEach(node => {
            let visible = true;

            if (!activeFilters.includes('active') && node.activation > 0.5) {
                visible = false;
            }

            if (!activeFilters.includes('recent') && node.last_accessed &&
                (Date.now() / 1000 - node.last_accessed) < 300) { // 5 minutes
                visible = false;
            }

            this.nodes.update({ id: node.id, hidden: !visible });
        });

        // Update edge visibility
        this.edges.forEach(edge => {
            let visible = true;

            if (!activeFilters.includes(edge.type || 'unknown')) {
                visible = false;
            }

            this.edges.update({ id: edge.id, hidden: !visible });
        });
    }

    toggleLearning(button) {
        const isPaused = button.textContent.includes('Resume');
        const newState = isPaused ? 'active' : 'paused';

        // Send command to backend
        fetch('/api/autonomous/learning', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ action: isPaused ? 'start' : 'pause' })
        })
        .then(response => response.json())
        .then(data => {
            button.textContent = isPaused ? '⏸️ Pause Learning' : '▶️ Resume Learning';
            this.updateLearningStatus(newState);
        })
        .catch(error => {
            console.error('Failed to toggle learning:', error);
        });
    }

    togglePhysics(button) {
        const isPaused = button.textContent.includes('Resume');
        const options = {
            physics: {
                enabled: isPaused
            }
        };

        this.network.setOptions(options);
        button.textContent = isPaused ? '⏸️ Pause Physics' : '▶️ Resume Physics';
    }

    resetLayout() {
        // Reset network layout
        this.network.setOptions({
            layout: {
                randomSeed: Date.now()
            }
        });

        // Trigger stabilization
        this.network.stabilize();
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
                this.loadInitialData();
            } else {
                alert('Failed to delete connections: ' + (data.error || 'Unknown error'));
            }
        })
        .catch(error => {
            console.error('Failed to delete connections:', error);
            alert('Failed to delete connections: ' + error.message);
        });
    }

    updateStats(stats) {
        this.learningStats = stats || {};

        const nodeCount = document.getElementById('node-count');
        const edgeCount = document.getElementById('edge-count');
        const learningStatus = document.getElementById('learning-status');

        if (nodeCount) nodeCount.textContent = `Nodes: ${stats.total_neurons || 0}`;
        if (edgeCount) edgeCount.textContent = `Edges: ${stats.total_connections || 0}`;
        if (learningStatus) {
            learningStatus.textContent = `Learning: ${stats.scanning_active ? 'Active' : 'Paused'}`;
        }
    }

    updateConnectionStatus(connected) {
        const statusElement = document.getElementById('learning-status');
        if (statusElement) {
            statusElement.textContent = `Connection: ${connected ? 'Connected' : 'Disconnected'}`;
            statusElement.style.color = connected ? '#00aa00' : '#aa0000';
        }
    }

    updateLearningStatus(status) {
        const statusElement = document.getElementById('learning-status');
        if (statusElement) {
            statusElement.textContent = `Learning: ${status === 'active' ? 'Active' : 'Paused'}`;
        }
    }

    showError(message) {
        console.error(message);

        // Create error notification
        const notification = document.createElement('div');
        notification.className = 'notification error';
        notification.innerHTML = `
            <div class="notification-content">
                <strong>Error:</strong> ${message}
            </div>
            <button onclick="this.parentElement.remove()">×</button>
        `;

        document.body.appendChild(notification);

        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (notification.parentElement) {
                notification.remove();
            }
        }, 5000);
    }

    destroy() {
        // Clean up resources
        if (this.ws) {
            this.ws.close();
        }

        if (this.network) {
            this.network.destroy();
        }

        if (this.controlsContainer) {
            this.controlsContainer.remove();
        }

        console.log('Autonomous Mesh Visualizer destroyed');
    }
}

// Export for use in other modules
window.AutonomousMeshVisualizer = AutonomousMeshVisualizer;
