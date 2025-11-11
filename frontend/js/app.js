/**
 * Frankenstino AI - Main Application Module
 * Initializes the application and coordinates all modules
 */

class FrankenstinoApp {
    constructor() {
        this.initialized = false;
        this.init();
    }

    async init() {
        try {
            console.log('Initializing Frankenstino AI...');

            // Show loading state
            this.showInitializingMessage();

            // Wait a moment for DOM to be ready
            await this.waitForDOM();

            // Initialize modules (they create their own instances)
            console.log('Modules loaded');

            // Check initial connection
            await this.checkInitialConnection();

            // Mark as initialized
            this.initialized = true;
            console.log('Frankenstino AI initialized successfully');

        } catch (error) {
            console.error('Failed to initialize Frankenstino AI:', error);
            this.showErrorMessage('Failed to initialize the application. Please check the console for details.');
        }
    }

    async waitForDOM() {
        return new Promise(resolve => {
            if (document.readyState === 'loading') {
                document.addEventListener('DOMContentLoaded', resolve);
            } else {
                resolve();
            }
        });
    }

    showInitializingMessage() {
        // Add an initializing message to the memory chat (default mode)
        setTimeout(() => {
            if (dualChat && typeof dualChat.addMemoryMessage === 'function') {
                dualChat.addMemoryMessage('assistant', '🔄 Initializing Frankenstino AI...', { system: true });
            }
        }, 100);
    }

    async checkInitialConnection() {
        console.log('Checking connection to backend...');

        try {
            const isOnline = await api.checkConnection();

            if (isOnline) {
                console.log('Backend connection established');

                // Remove initializing message and add welcome
                setTimeout(() => {
                    if (dualChat) {
                        // Clear any loading messages
                        const loadingMessages = document.querySelectorAll('.loading');
                        loadingMessages.forEach(msg => msg.remove());

                        // The dual chat interface shows placeholders by default
                        // No need to explicitly show welcome messages
                    }
                }, 500);

            } else {
                console.warn('Backend not available');
                this.showBackendOfflineMessage();
            }

        } catch (error) {
            console.error('Connection check failed:', error);
            this.showBackendOfflineMessage();
        }
    }

    showBackendOfflineMessage() {
        const message = `
            <div style="text-align: center; padding: 20px;">
                <h3 style="color: #dc3545; margin-bottom: 15px;">🔌 Backend Server Offline</h3>
                <p style="margin-bottom: 15px;">
                    Frankenstino AI requires a running backend server to function.
                </p>
                <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 15px;">
                    <strong>To start the server:</strong><br>
                    <code style="background: #e9ecef; padding: 2px 6px; border-radius: 3px; font-family: monospace;">
                        python -m uvicorn backend.main:app --reload
                    </code>
                </div>
                <p style="font-size: 0.9rem; color: #6c757d;">
                    The server will be available at <strong>http://localhost:8000</strong>
                </p>
            </div>
        `;

        if (dualChat && typeof dualChat.addMemoryMessage === 'function') {
            // Clear any loading messages
            const loadingMessages = document.querySelectorAll('.loading');
            loadingMessages.forEach(msg => msg.remove());

            dualChat.addMemoryMessage('assistant', message, { error: true, system: true });
        }
    }

    showErrorMessage(message) {
        const errorDiv = document.createElement('div');
        errorDiv.style.cssText = `
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: white;
            border: 2px solid #dc3545;
            border-radius: 12px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            z-index: 10000;
            max-width: 500px;
            text-align: center;
        `;

        errorDiv.innerHTML = `
            <h2 style="color: #dc3545; margin-bottom: 15px;">❌ Initialization Error</h2>
            <p style="margin-bottom: 20px; line-height: 1.5;">${message}</p>
            <button onclick="location.reload()" style="
                background: #dc3545;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 6px;
                cursor: pointer;
                font-weight: 600;
            ">Reload Page</button>
        `;

        document.body.appendChild(errorDiv);
    }

    // Utility methods
    static formatTimestamp(timestamp) {
        return new Date(timestamp).toLocaleString();
    }

    static debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }

    static throttle(func, limit) {
        let inThrottle;
        return function() {
            const args = arguments;
            const context = this;
            if (!inThrottle) {
                func.apply(context, args);
                inThrottle = true;
                setTimeout(() => inThrottle = false, limit);
            }
        }
    }
}

// Global error handling
window.addEventListener('error', (event) => {
    console.error('Global error:', event.error);
});

window.addEventListener('unhandledrejection', (event) => {
    console.error('Unhandled promise rejection:', event.reason);
});

// Initialize the application when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    // Create global app instance
    window.frankenstinoApp = new FrankenstinoApp();
});

// Export for potential use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = FrankenstinoApp;
}
