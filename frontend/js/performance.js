/**
 * Performance Dashboard - Real-time monitoring and visualization
 * Displays system performance, neural network progress, and analytics
 */

class PerformanceDashboard {
    constructor() {
        this.currentTimeRange = '1h';
        this.updateInterval = null;
        this.charts = {};
        this.metrics = {};
        this.isVisible = false;

        this.init();
    }

    init() {
        this.bindEvents();
        this.initializeCharts();
        this.loadInitialData();
        this.startRealTimeUpdates();
    }

    bindEvents() {
        // Performance button
        const performanceBtn = document.getElementById('performance-btn');
        if (performanceBtn) {
            performanceBtn.addEventListener('click', () => this.showDashboard());
        }

        // Modal close
        const modal = document.getElementById('performance-modal');
        const modalClose = modal?.querySelector('.modal-close');
        if (modalClose) {
            modalClose.addEventListener('click', () => this.hideDashboard());
        }

        // Click outside to close
        if (modal) {
            modal.addEventListener('click', (e) => {
                if (e.target === modal) {
                    this.hideDashboard();
                }
            });
        }

        // Performance tab buttons
        const tabBtns = document.querySelectorAll('.performance-tab-btn');
        tabBtns.forEach(btn => {
            btn.addEventListener('click', (e) => {
                const tabName = e.target.getAttribute('data-tab');
                this.switchTab(tabName);
            });
        });

        // Time range buttons
        const timeBtns = document.querySelectorAll('.time-btn');
        timeBtns.forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.setTimeRange(e.target.getAttribute('data-range'));
            });
        });

        // Refresh button
        const refreshBtn = document.getElementById('refresh-performance-btn');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', () => this.refreshData());
        }

        // Export button
        const exportBtn = document.getElementById('export-performance-btn');
        if (exportBtn) {
            exportBtn.addEventListener('click', () => this.exportReport());
        }
    }

    showDashboard() {
        const modal = document.getElementById('performance-modal');
        if (modal) {
            modal.style.display = 'flex';
            this.isVisible = true;
            this.refreshData();
        }
    }

    hideDashboard() {
        const modal = document.getElementById('performance-modal');
        if (modal) {
            modal.style.display = 'none';
            this.isVisible = false;
        }
    }

    switchTab(tabName) {
        // Update tab button states
        const tabBtns = document.querySelectorAll('.performance-tab-btn');
        tabBtns.forEach(btn => {
            btn.classList.toggle('active', btn.getAttribute('data-tab') === tabName);
        });

        // Update tab content visibility
        const tabContents = document.querySelectorAll('.performance-tab-content');
        tabContents.forEach(content => {
            content.classList.toggle('active', content.id === `${tabName}-tab`);
        });

        // Store current tab
        this.currentTab = tabName;

        // Refresh data when switching to certain tabs
        if (tabName === 'trends') {
            this.refreshData();
        }
    }

    setTimeRange(range) {
        this.currentTimeRange = range;

        // Update button states
        const timeBtns = document.querySelectorAll('.time-btn');
        timeBtns.forEach(btn => {
            btn.classList.toggle('active', btn.getAttribute('data-range') === range);
        });

        // Refresh data with new time range
        this.refreshData();
    }

    initializeCharts() {
        // Initialize Chart.js if available
        if (typeof Chart !== 'undefined') {
            this.initializeLatencyChart();
            this.initializeMemoryChart();
            this.initializeCacheChart();
            this.initializeQueryChart();
        } else {
            console.warn('Chart.js not loaded, using fallback visualizations');
        }
    }

    initializeLatencyChart() {
        const ctx = document.getElementById('latency-chart');
        if (!ctx) return;

        this.charts.latency = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Response Time (ms)',
                    data: [],
                    borderColor: '#007bff',
                    backgroundColor: 'rgba(0, 123, 255, 0.1)',
                    tension: 0.4,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        grid: { color: '#444444' },
                        ticks: { color: '#cccccc' }
                    },
                    x: {
                        grid: { color: '#444444' },
                        ticks: { color: '#cccccc' }
                    }
                },
                plugins: {
                    legend: {
                        labels: { color: '#e0e0e0' }
                    }
                }
            }
        });
    }

    initializeMemoryChart() {
        const ctx = document.getElementById('memory-chart');
        if (!ctx) return;

        this.charts.memory = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Memory Usage (GB)',
                    data: [],
                    borderColor: '#28a745',
                    backgroundColor: 'rgba(40, 167, 69, 0.1)',
                    tension: 0.4,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        grid: { color: '#444444' },
                        ticks: { color: '#cccccc' }
                    },
                    x: {
                        grid: { color: '#444444' },
                        ticks: { color: '#cccccc' }
                    }
                },
                plugins: {
                    legend: {
                        labels: { color: '#e0e0e0' }
                    }
                }
            }
        });
    }

    initializeCacheChart() {
        const ctx = document.getElementById('cache-chart');
        if (!ctx) return;

        this.charts.cache = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Hits', 'Misses'],
                datasets: [{
                    data: [0, 0],
                    backgroundColor: ['#28a745', '#dc3545'],
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: { color: '#e0e0e0' }
                    }
                }
            }
        });
    }

    initializeQueryChart() {
        const ctx = document.getElementById('query-chart');
        if (!ctx) return;

        this.charts.query = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: [],
                datasets: [{
                    label: 'Queries per Hour',
                    data: [],
                    backgroundColor: '#007bff',
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        grid: { color: '#444444' },
                        ticks: { color: '#cccccc' }
                    },
                    x: {
                        grid: { color: '#444444' },
                        ticks: { color: '#cccccc' }
                    }
                },
                plugins: {
                    legend: {
                        labels: { color: '#e0e0e0' }
                    }
                }
            }
        });
    }

    async loadInitialData() {
        try {
            await this.refreshData();
        } catch (error) {
            console.error('Failed to load initial performance data:', error);
            this.showFallbackData();
        }
    }

    async refreshData() {
        if (!this.isVisible) return;

        try {
            const [metricsResponse, neuralResponse, healthResponse] = await Promise.all([
                API.call('/api/performance/metrics', { method: 'GET' }),
                API.call('/api/neural/progress', { method: 'GET' }),
                API.call('/api/system/health', { method: 'GET' })
            ]);

            if (metricsResponse.success) {
                this.updateSystemMetrics(metricsResponse.data);
            }

            if (neuralResponse.success) {
                this.updateNeuralProgress(neuralResponse.data);
            }

            if (healthResponse.success) {
                this.updateSystemHealth(healthResponse.data);
            }

        } catch (error) {
            console.error('Failed to refresh performance data:', error);
        }
    }

    updateSystemMetrics(data) {
        this.metrics = { ...this.metrics, ...data };

        // Update latency
        this.updateLatencyDisplay(data.latency_ms || 0);
        this.updateMemoryDisplay(data.memory_usage_mb || 0, data.max_memory_mb || 500);
        this.updateCacheDisplay(data.cache_hit_rate || 0);
        this.updateThroughputDisplay(data.queries_per_second || 0);

        // Update charts
        this.updateCharts(data);
    }

    updateLatencyDisplay(latency) {
        const valueElement = document.getElementById('latency-value');
        const fillElement = document.getElementById('latency-fill');
        const trendElement = document.getElementById('latency-trend');

        if (valueElement) {
            valueElement.textContent = `${latency.toFixed(0)}ms`;
        }

        if (fillElement) {
            // Target is 500ms, so calculate percentage
            const percentage = Math.min((latency / 500) * 100, 100);
            fillElement.style.width = `${percentage}%`;
        }

        // Simple trend calculation (would be more sophisticated in real implementation)
        if (trendElement && this.metrics.previous_latency) {
            const change = latency - this.metrics.previous_latency;
            const trend = change > 0 ? '↗️' : change < 0 ? '↘️' : '→';
            const color = change > 0 ? 'color: #dc3545' : change < 0 ? 'color: #28a745' : '';
            trendElement.innerHTML = `<span style="${color}">${trend}</span> <span>${Math.abs(change).toFixed(1)}ms</span>`;
        }

        this.metrics.previous_latency = latency;
    }

    updateMemoryDisplay(usage, maxMemory) {
        const valueElement = document.getElementById('memory-value');
        const fillElement = document.getElementById('memory-fill');
        const trendElement = document.getElementById('memory-trend');

        if (valueElement) {
            valueElement.textContent = `${usage.toFixed(1)}GB`;
        }

        if (fillElement) {
            const percentage = (usage / maxMemory) * 100;
            fillElement.style.width = `${Math.min(percentage, 100)}%`;
        }

        // Trend calculation
        if (trendElement && this.metrics.previous_memory) {
            const change = usage - this.metrics.previous_memory;
            const trend = change > 0 ? '↗️' : change < 0 ? '↘️' : '→';
            const color = change > 0 ? 'color: #dc3545' : change < 0 ? 'color: #28a745' : '';
            trendElement.innerHTML = `<span style="${color}">${trend}</span> <span>${Math.abs(change).toFixed(1)}GB</span>`;
        }

        this.metrics.previous_memory = usage;
    }

    updateCacheDisplay(hitRate) {
        const valueElement = document.getElementById('cache-value');
        const fillElement = document.getElementById('cache-fill');
        const trendElement = document.getElementById('cache-trend');

        if (valueElement) {
            valueElement.textContent = `${(hitRate * 100).toFixed(1)}%`;
        }

        if (fillElement) {
            fillElement.style.width = `${hitRate * 100}%`;
        }

        // Trend calculation
        if (trendElement && this.metrics.previous_cache_hit_rate !== undefined) {
            const change = hitRate - this.metrics.previous_cache_hit_rate;
            const trend = change > 0 ? '↗️' : change < 0 ? '↘️' : '→';
            const color = change > 0 ? 'color: #28a745' : change < 0 ? 'color: #dc3545' : '';
            trendElement.innerHTML = `<span style="${color}">${trend}</span> <span>${(Math.abs(change) * 100).toFixed(1)}%</span>`;
        }

        this.metrics.previous_cache_hit_rate = hitRate;

        // Update cache chart
        if (this.charts.cache) {
            const hits = Math.round(hitRate * 100);
            const misses = 100 - hits;
            this.charts.cache.data.datasets[0].data = [hits, misses];
            this.charts.cache.update();
        }
    }

    updateThroughputDisplay(qps) {
        const valueElement = document.getElementById('throughput-value');
        const chartElement = document.getElementById('throughput-chart');

        if (valueElement) {
            valueElement.textContent = qps.toFixed(1);
        }

        // Simple bar chart visualization
        if (chartElement) {
            const bars = chartElement.querySelectorAll('.bar');
            if (bars.length === 0) {
                // Create bars
                for (let i = 0; i < 10; i++) {
                    const bar = document.createElement('div');
                    bar.className = 'bar';
                    bar.style.height = `${Math.random() * 40 + 10}px`;
                    chartElement.appendChild(bar);
                }
            } else {
                // Update existing bars
                bars.forEach(bar => {
                    bar.style.height = `${Math.random() * 40 + 10}px`;
                });
            }
        }
    }

    updateNeuralProgress(data) {
        // Update neuron count
        const neuronCount = data.neuron_count || 0;
        const targetNeurons = 1000000000; // 1B
        const percentage = (neuronCount / targetNeurons) * 100;

        const countElement = document.getElementById('neuron-count');
        const percentageElement = document.getElementById('neuron-percentage');
        const progressFill = document.getElementById('neuron-progress-fill');
        const remainingElement = document.getElementById('neuron-remaining');

        if (countElement) {
            countElement.textContent = this.formatNumber(neuronCount);
        }

        if (percentageElement) {
            percentageElement.textContent = `${percentage.toFixed(6)}%`;
        }

        if (progressFill) {
            progressFill.style.width = `${Math.min(percentage, 100)}%`;
        }

        if (remainingElement) {
            const remaining = targetNeurons - neuronCount;
            remainingElement.textContent = `${this.formatNumber(remaining)} remaining`;
        }

        // Update network metrics
        this.updateNetworkMetric('network-density', data.network_density || 0, 'connections/node');
        this.updateNetworkMetric('active-nodes', data.active_nodes || 0, '%');
        this.updateNetworkMetric('daily-growth', data.daily_growth || 0, 'neurons/day');
        this.updateNetworkMetric('consolidation-rate', data.consolidation_rate || 0, '%');
    }

    updateNetworkMetric(elementId, value, unit) {
        const element = document.getElementById(elementId);
        if (element) {
            element.textContent = typeof value === 'number' ? value.toFixed(1) : value;
        }
    }

    updateSystemHealth(data) {
        this.updateHealthMetric('cpu', data.cpu_percent || 0);
        this.updateHealthMetric('ram', data.memory_percent || 0);
        this.updateHealthMetric('disk', data.disk_usage_percent || 0);
        this.updateHealthMetric('gpu', data.gpu_percent || 0);
    }

    updateHealthMetric(type, percentage) {
        const fillElement = document.getElementById(`${type}-fill`);
        const valueElement = document.getElementById(`${type}-value`);

        if (fillElement) {
            fillElement.style.width = `${Math.min(percentage, 100)}%`;
        }

        if (valueElement) {
            valueElement.textContent = `${percentage.toFixed(1)}%`;
        }
    }

    updateCharts(data) {
        // Update latency chart
        if (this.charts.latency && data.latency_history) {
            this.updateChart(this.charts.latency, data.latency_history);
        }

        // Update memory chart
        if (this.charts.memory && data.memory_history) {
            this.updateChart(this.charts.memory, data.memory_history);
        }

        // Update query chart
        if (this.charts.query && data.query_history) {
            this.updateBarChart(this.charts.query, data.query_history);
        }
    }

    updateChart(chart, data) {
        if (!data || !Array.isArray(data)) return;

        const labels = data.map((_, i) => `${i * 5}m ago`).reverse();
        const values = data.reverse();

        chart.data.labels = labels;
        chart.data.datasets[0].data = values;
        chart.update();
    }

    updateBarChart(chart, data) {
        if (!data || !Array.isArray(data)) return;

        const labels = data.map((_, i) => `Hour ${i}`).reverse();
        const values = data.reverse();

        chart.data.labels = labels;
        chart.data.datasets[0].data = values;
        chart.update();
    }

    startRealTimeUpdates() {
        // Update every 30 seconds when dashboard is visible
        this.updateInterval = setInterval(() => {
            if (this.isVisible) {
                this.refreshData();
            }
        }, 30000);
    }

    stopRealTimeUpdates() {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
            this.updateInterval = null;
        }
    }

    async exportReport() {
        try {
            const response = await API.call('/api/performance/export', {
                method: 'POST',
                body: {
                    time_range: this.currentTimeRange,
                    format: 'json'
                }
            });

            if (response.success) {
                // Download the report
                const blob = new Blob([JSON.stringify(response.data, null, 2)],
                                    { type: 'application/json' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `performance-report-${new Date().toISOString().split('T')[0]}.json`;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);

                this.showNotification('Performance report exported successfully', 'success');
            } else {
                this.showNotification('Failed to export performance report', 'error');
            }
        } catch (error) {
            console.error('Export failed:', error);
            this.showNotification('Export failed', 'error');
        }
    }

    showFallbackData() {
        // Show sample data when API is not available
        this.updateSystemMetrics({
            latency_ms: 245,
            memory_usage_mb: 1.2,
            max_memory_mb: 2.0,
            cache_hit_rate: 0.87,
            queries_per_second: 12.5
        });

        this.updateNeuralProgress({
            neuron_count: 842156789,
            network_density: 2.4,
            active_nodes: 67.8,
            daily_growth: 12456,
            consolidation_rate: 23.1
        });

        this.updateSystemHealth({
            cpu_percent: 45.2,
            memory_percent: 68.7,
            disk_usage_percent: 34.1,
            gpu_percent: 78.3
        });
    }

    formatNumber(num) {
        if (num >= 1000000000) {
            return (num / 1000000000).toFixed(1) + 'B';
        } else if (num >= 1000000) {
            return (num / 1000000).toFixed(1) + 'M';
        } else if (num >= 1000) {
            return (num / 1000).toFixed(1) + 'K';
        }
        return num.toString();
    }

    showNotification(message, type = 'info') {
        // Use existing notification system if available
        if (window.showNotification) {
            window.showNotification(message, type);
        } else {
            console.log(`[${type.toUpperCase()}] ${message}`);
        }
    }

    destroy() {
        this.stopRealTimeUpdates();
        this.hideDashboard();

        // Clean up charts
        Object.values(this.charts).forEach(chart => {
            if (chart && typeof chart.destroy === 'function') {
                chart.destroy();
            }
        });
        this.charts = {};
    }
}

// Global performance dashboard instance
let performanceDashboard;

document.addEventListener('DOMContentLoaded', () => {
    performanceDashboard = new PerformanceDashboard();
});

// Export for global access
window.PerformanceDashboard = PerformanceDashboard;
window.performanceDashboard = performanceDashboard;
