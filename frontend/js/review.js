/**
 * Memory Review Interface - JavaScript functionality
 * Handles human-in-the-loop validation of AI-processed memory chunks
 */

class MemoryReviewUI {
    constructor() {
        this.currentReview = null;
        this.selectedDecision = null;
        this.reviewerName = this.getReviewerName();
        this.init();
    }

    init() {
        this.bindEvents();
        this.loadReviewQueue();
        this.loadStats();
        this.updateReviewerName();
    }

    bindEvents() {
        // Priority filter
        const priorityFilter = document.getElementById('priority-filter');
        if (priorityFilter) {
            priorityFilter.addEventListener('change', () => this.loadReviewQueue());
        }

        // Refresh button
        const refreshBtn = document.getElementById('refresh-btn');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', () => {
                this.loadReviewQueue();
                this.loadStats();
            });
        }

        // Auto-approve button
        const autoApproveBtn = document.getElementById('auto-approve-btn');
        if (autoApproveBtn) {
            autoApproveBtn.addEventListener('click', () => this.autoApproveHighConfidence());
        }

        // Modal events
        const modal = document.getElementById('review-modal');
        const modalClose = document.querySelector('.modal-close');
        const cancelBtn = document.getElementById('cancel-review');

        if (modalClose) {
            modalClose.addEventListener('click', () => this.closeModal());
        }

        if (cancelBtn) {
            cancelBtn.addEventListener('click', () => this.closeModal());
        }

        // Click outside modal to close
        if (modal) {
            modal.addEventListener('click', (e) => {
                if (e.target === modal) {
                    this.closeModal();
                }
            });
        }

        // Decision buttons
        const decisionBtns = document.querySelectorAll('.decision-btn');
        decisionBtns.forEach(btn => {
            btn.addEventListener('click', (e) => this.selectDecision(e.target));
        });

        // Submit review
        const submitBtn = document.getElementById('submit-review');
        if (submitBtn) {
            submitBtn.addEventListener('click', () => this.submitReview());
        }
    }

    getReviewerName() {
        // Try to get from localStorage, otherwise use default
        return localStorage.getItem('reviewerName') || 'Anonymous Reviewer';
    }

    updateReviewerName() {
        const reviewerName = this.getReviewerName();
        const nameElement = document.getElementById('reviewer-name');
        if (nameElement) {
            nameElement.textContent = reviewerName;
        }
    }

    setReviewerName(name) {
        this.reviewerName = name;
        localStorage.setItem('reviewerName', name);
        this.updateReviewerName();
    }

    async loadReviewQueue() {
        const queueElement = document.getElementById('review-queue');
        if (!queueElement) return;

        // Show loading
        queueElement.innerHTML = `
            <div class="loading-spinner">
                <div class="spinner"></div>
                <p>Loading review queue...</p>
            </div>
        `;

        try {
            const priority = document.getElementById('priority-filter')?.value || '';
            const response = await API.call('/api/review/queue', {
                method: 'GET',
                params: priority ? { priority } : {}
            });

            if (response.success) {
                this.renderReviewQueue(response.data.reviews);
                this.updatePendingCount(response.data.total_pending);
            } else {
                this.showError('Failed to load review queue');
            }
        } catch (error) {
            console.error('Error loading review queue:', error);
            this.showError('Error loading review queue');
        }
    }

    renderReviewQueue(reviews) {
        const queueElement = document.getElementById('review-queue');

        if (!reviews || reviews.length === 0) {
            queueElement.innerHTML = `
                <div style="text-align: center; padding: 60px; color: #6c757d;">
                    <h3>🎉 All caught up!</h3>
                    <p>No pending reviews at this time.</p>
                </div>
            `;
            return;
        }

        const reviewCards = reviews.map(review => this.createReviewCard(review)).join('');
        queueElement.innerHTML = reviewCards;
    }

    createReviewCard(review) {
        const priority = review.priority || 'normal';
        const submittedTime = new Date(review.submitted_at * 1000);
        const timeAgo = this.getTimeAgo(submittedTime);

        // Truncate text for preview
        const originalText = review.chunk?.text || '';
        const previewText = originalText.length > 150 ?
            originalText.substring(0, 150) + '...' : originalText;

        const summary = review.chunk?.summary || 'No summary available';

        return `
            <div class="review-card" data-review-id="${review.review_id}">
                <div class="review-card-header">
                    <h3 class="review-card-title">Review ${review.review_id.split('_')[1]}</h3>
                    <div class="review-card-meta">
                        <span class="priority-badge priority-${priority}">${priority}</span>
                        <span>${timeAgo}</span>
                    </div>
                </div>

                <div class="review-card-content">
                    <div class="chunk-preview">${this.escapeHtml(previewText)}</div>
                    <div class="chunk-summary">${this.escapeHtml(summary)}</div>
                </div>

                <div class="review-card-actions">
                    <button class="btn-small btn-quick-reject" onclick="reviewUI.quickReject('${review.review_id}')">
                        ❌ Quick Reject
                    </button>
                    <button class="btn-small btn-quick-accept" onclick="reviewUI.quickAccept('${review.review_id}')">
                        ✅ Quick Accept
                    </button>
                    <button class="btn-small btn-review" onclick="reviewUI.openReviewModal('${review.review_id}')">
                        👁️ Review
                    </button>
                </div>
            </div>
        `;
    }

    async loadStats() {
        try {
            const response = await API.call('/api/review/stats', { method: 'GET' });

            if (response.success) {
                this.updateStats(response.data);
            }
        } catch (error) {
            console.error('Error loading stats:', error);
        }
    }

    updateStats(stats) {
        // Update pending count
        this.updatePendingCount(stats.pending_reviews || 0);

        // Update detailed stats
        const statElements = {
            'stat-pending': stats.pending_reviews || 0,
            'stat-accepted': stats.accepted || 0,
            'stat-rejected': stats.rejected || 0,
            'stat-modified': stats.modified || 0
        };

        Object.entries(statElements).forEach(([elementId, value]) => {
            const element = document.getElementById(elementId);
            if (element) {
                element.textContent = value;
            }
        });
    }

    updatePendingCount(count) {
        const element = document.getElementById('pending-count');
        if (element) {
            element.textContent = `${count} pending reviews`;
        }
    }

    async quickAccept(reviewId) {
        await this.processQuickDecision(reviewId, 'accept');
    }

    async quickReject(reviewId) {
        await this.processQuickDecision(reviewId, 'reject');
    }

    async processQuickDecision(reviewId, decision) {
        try {
            const response = await API.call(`/api/review/${reviewId}`, {
                method: 'POST',
                body: {
                    decision: decision,
                    reviewer: this.reviewerName,
                    feedback: decision === 'reject' ? 'Quick rejection' : ''
                }
            });

            if (response.success) {
                this.showSuccess(`Review ${decision}ed successfully`);
                this.loadReviewQueue();
                this.loadStats();
            } else {
                this.showError(`Failed to ${decision} review`);
            }
        } catch (error) {
            console.error(`Error processing quick ${decision}:`, error);
            this.showError(`Error processing ${decision}`);
        }
    }

    openReviewModal(reviewId) {
        // Find the review data (in a real app, you'd fetch this from the server)
        const reviewCard = document.querySelector(`[data-review-id="${reviewId}"]`);
        if (!reviewCard) return;

        // For now, we'll need to refetch the review data
        this.loadReviewDetails(reviewId);
    }

    async loadReviewDetails(reviewId) {
        try {
            // Get the review from the queue
            const queueResponse = await API.call('/api/review/queue', { method: 'GET' });
            if (!queueResponse.success) return;

            const review = queueResponse.data.reviews.find(r => r.review_id === reviewId);
            if (!review) return;

            this.currentReview = review;
            this.populateModal(review);
            this.showModal();

        } catch (error) {
            console.error('Error loading review details:', error);
            this.showError('Failed to load review details');
        }
    }

    populateModal(review) {
        // Update modal elements
        const chunkIdElement = document.querySelector('.chunk-id');
        const chunkPriorityElement = document.querySelector('.chunk-priority');
        const chunkReasonElement = document.querySelector('.chunk-reason');

        if (chunkIdElement) chunkIdElement.textContent = `ID: ${review.review_id}`;
        if (chunkPriorityElement) chunkPriorityElement.textContent = `Priority: ${review.priority || 'normal'}`;
        if (chunkReasonElement) chunkReasonElement.textContent = `Reason: ${review.reason || 'quality_check'}`;

        // Update content
        const originalTextElement = document.getElementById('original-text');
        const aiSummaryElement = document.getElementById('ai-summary');
        const metadataElement = document.getElementById('chunk-metadata');

        if (originalTextElement) {
            originalTextElement.textContent = review.chunk?.text || 'No text available';
        }

        if (aiSummaryElement) {
            aiSummaryElement.textContent = review.chunk?.summary || 'No summary available';
        }

        if (metadataElement) {
            const metadata = review.chunk || {};
            metadataElement.textContent = JSON.stringify(metadata, null, 2);
        }

        // Reset form state
        this.selectedDecision = null;
        this.resetDecisionButtons();
        this.hideFeedbackSection();
    }

    selectDecision(buttonElement) {
        const decision = buttonElement.getAttribute('data-decision');

        // Reset all buttons
        this.resetDecisionButtons();

        // Select the clicked button
        buttonElement.classList.add('selected');
        this.selectedDecision = decision;

        // Show feedback section for modify decision
        if (decision === 'modify') {
            this.showFeedbackSection();
        } else {
            this.hideFeedbackSection();
        }
    }

    resetDecisionButtons() {
        const buttons = document.querySelectorAll('.decision-btn');
        buttons.forEach(btn => btn.classList.remove('selected'));
    }

    showFeedbackSection() {
        const feedbackSection = document.getElementById('feedback-section');
        if (feedbackSection) {
            feedbackSection.style.display = 'block';
        }
    }

    hideFeedbackSection() {
        const feedbackSection = document.getElementById('feedback-section');
        if (feedbackSection) {
            feedbackSection.style.display = 'none';
        }
    }

    async submitReview() {
        if (!this.currentReview || !this.selectedDecision) {
            this.showError('Please select a decision');
            return;
        }

        const feedback = document.getElementById('review-feedback')?.value || '';

        try {
            const response = await API.call(`/api/review/${this.currentReview.review_id}`, {
                method: 'POST',
                body: {
                    decision: this.selectedDecision,
                    reviewer: this.reviewerName,
                    feedback: feedback
                }
            });

            if (response.success) {
                this.showSuccess(`Review ${this.selectedDecision}ed successfully`);
                this.closeModal();
                this.loadReviewQueue();
                this.loadStats();
            } else {
                this.showError('Failed to submit review');
            }
        } catch (error) {
            console.error('Error submitting review:', error);
            this.showError('Error submitting review');
        }
    }

    showModal() {
        const modal = document.getElementById('review-modal');
        if (modal) {
            modal.style.display = 'block';
        }
    }

    closeModal() {
        const modal = document.getElementById('review-modal');
        if (modal) {
            modal.style.display = 'none';
        }
        this.currentReview = null;
        this.selectedDecision = null;
    }

    async autoApproveHighConfidence() {
        // This would implement auto-approval logic
        // For now, just show a message
        this.showInfo('Auto-approval feature coming soon!');
    }

    // Utility methods
    getTimeAgo(date) {
        const now = new Date();
        const diffMs = now - date;
        const diffMins = Math.floor(diffMs / (1000 * 60));
        const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
        const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));

        if (diffMins < 1) return 'Just now';
        if (diffMins < 60) return `${diffMins}m ago`;
        if (diffHours < 24) return `${diffHours}h ago`;
        return `${diffDays}d ago`;
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    showSuccess(message) {
        this.showNotification(message, 'success');
    }

    showError(message) {
        this.showNotification(message, 'error');
    }

    showInfo(message) {
        this.showNotification(message, 'info');
    }

    showNotification(message, type = 'info') {
        // Simple notification - in a real app, you'd use a proper notification system
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.textContent = message;
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 15px 20px;
            border-radius: 6px;
            color: white;
            font-weight: 600;
            z-index: 10000;
            animation: slideIn 0.3s ease-out;
        `;

        // Set background color based on type
        const colors = {
            success: '#28a745',
            error: '#dc3545',
            info: '#007bff'
        };
        notification.style.backgroundColor = colors[type] || colors.info;

        document.body.appendChild(notification);

        // Remove after 3 seconds
        setTimeout(() => {
            notification.style.animation = 'slideOut 0.3s ease-in';
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.parentNode.removeChild(notification);
                }
            }, 300);
        }, 3000);
    }
}

// Add notification animations to CSS (inject into head)
const notificationStyles = `
    @keyframes slideIn {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }

    @keyframes slideOut {
        from { transform: translateX(0); opacity: 1; }
        to { transform: translateX(100%); opacity: 0; }
    }
`;

const styleSheet = document.createElement('style');
styleSheet.textContent = notificationStyles;
document.head.appendChild(styleSheet);

// Initialize the review UI when DOM is loaded
let reviewUI;
document.addEventListener('DOMContentLoaded', () => {
    reviewUI = new MemoryReviewUI();
});

// Export for global access (for onclick handlers)
window.reviewUI = reviewUI;
