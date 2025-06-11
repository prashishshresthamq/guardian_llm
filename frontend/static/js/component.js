// Guardian LLM UI Components
// Reusable components for the dashboard

// Component namespace
const Components = {
    // Initialize all components
    init: function() {
        this.initCharts();
        this.initAlerts();
        this.initModals();
        this.bindEvents();
    },

    // Chart configurations
    charts: {
        riskGauge: null,
        sentimentChart: null,
        trendChart: null,
        categoryChart: null
    },

    // Create Risk Gauge Chart
    createRiskGauge: function(elementId, value, title = 'Risk Level') {
        const ctx = document.getElementById(elementId);
        if (!ctx) return;

        const data = {
            datasets: [{
                data: [value, 100 - value],
                backgroundColor: [
                    value > 70 ? '#dc3545' : value > 40 ? '#ffc107' : '#28a745',
                    '#e9ecef'
                ],
                borderWidth: 0
            }]
        };

        const options = {
            responsive: true,
            maintainAspectRatio: false,
            circumference: 180,
            rotation: 270,
            cutout: '80%',
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    enabled: false
                },
                title: {
                    display: true,
                    text: title,
                    font: {
                        size: 16,
                        weight: 'bold'
                    }
                }
            }
        };

        // Add center text
        const centerText = {
            id: 'centerText',
            afterDraw: function(chart) {
                const ctx = chart.ctx;
                ctx.save();
                const centerX = chart.width / 2;
                const centerY = chart.height - 30;
                
                ctx.font = 'bold 24px Arial';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillStyle = value > 70 ? '#dc3545' : value > 40 ? '#ffc107' : '#28a745';
                ctx.fillText(value + '%', centerX, centerY);
                
                ctx.font = '14px Arial';
                ctx.fillStyle = '#666';
                ctx.fillText(value > 70 ? 'High' : value > 40 ? 'Medium' : 'Low', centerX, centerY + 25);
                ctx.restore();
            }
        };

        return new Chart(ctx, {
            type: 'doughnut',
            data: data,
            options: options,
            plugins: [centerText]
        });
    },

    // Create Sentiment Chart
    createSentimentChart: function(elementId, sentimentData) {
        const ctx = document.getElementById(elementId);
        if (!ctx) return;

        const data = {
            labels: ['Positive', 'Neutral', 'Negative'],
            datasets: [{
                label: 'Sentiment Distribution',
                data: [
                    sentimentData.positive || 0,
                    sentimentData.neutral || 0,
                    sentimentData.negative || 0
                ],
                backgroundColor: [
                    'rgba(40, 167, 69, 0.8)',
                    'rgba(108, 117, 125, 0.8)',
                    'rgba(220, 53, 69, 0.8)'
                ],
                borderColor: [
                    'rgb(40, 167, 69)',
                    'rgb(108, 117, 125)',
                    'rgb(220, 53, 69)'
                ],
                borderWidth: 1
            }]
        };

        const options = {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                },
                title: {
                    display: true,
                    text: 'Sentiment Analysis',
                    font: {
                        size: 16,
                        weight: 'bold'
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    grid: {
                        display: true,
                        color: 'rgba(0, 0, 0, 0.05)'
                    }
                },
                x: {
                    grid: {
                        display: false
                    }
                }
            }
        };

        return new Chart(ctx, {
            type: 'bar',
            data: data,
            options: options
        });
    },

    // Create Trend Chart
    createTrendChart: function(elementId, trendData) {
        const ctx = document.getElementById(elementId);
        if (!ctx) return;

        const data = {
            labels: trendData.labels || ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
            datasets: [{
                label: 'Critical Risk',
                data: trendData.criticalRisk || [65, 59, 80, 81, 56, 55, 40],
                borderColor: 'rgb(220, 53, 69)',
                backgroundColor: 'rgba(220, 53, 69, 0.1)',
                tension: 0.4
            }, {
                label: 'Overall Risk',
                data: trendData.overallRisk || [45, 39, 60, 61, 36, 35, 20],
                borderColor: 'rgb(255, 193, 7)',
                backgroundColor: 'rgba(255, 193, 7, 0.1)',
                tension: 0.4
            }]
        };

        const options = {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                },
                title: {
                    display: true,
                    text: 'Risk Trends Over Time',
                    font: {
                        size: 16,
                        weight: 'bold'
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    ticks: {
                        callback: function(value) {
                            return value + '%';
                        }
                    },
                    grid: {
                        display: true,
                        color: 'rgba(0, 0, 0, 0.05)'
                    }
                },
                x: {
                    grid: {
                        display: false
                    }
                }
            }
        };

        return new Chart(ctx, {
            type: 'line',
            data: data,
            options: options
        });
    },

    // Create Category Distribution Chart
    createCategoryChart: function(elementId, categoryData) {
        const ctx = document.getElementById(elementId);
        if (!ctx) return;

        const data = {
            labels: categoryData.labels || ['Violence', 'Self-Harm', 'Hate Speech', 'Harassment', 'Adult Content'],
            datasets: [{
                data: categoryData.values || [30, 25, 20, 15, 10],
                backgroundColor: [
                    'rgba(220, 53, 69, 0.8)',
                    'rgba(255, 99, 132, 0.8)',
                    'rgba(255, 159, 64, 0.8)',
                    'rgba(255, 193, 7, 0.8)',
                    'rgba(108, 117, 125, 0.8)'
                ],
                borderColor: [
                    'rgb(220, 53, 69)',
                    'rgb(255, 99, 132)',
                    'rgb(255, 159, 64)',
                    'rgb(255, 193, 7)',
                    'rgb(108, 117, 125)'
                ],
                borderWidth: 1
            }]
        };

        const options = {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'right',
                },
                title: {
                    display: true,
                    text: 'Risk Categories',
                    font: {
                        size: 16,
                        weight: 'bold'
                    }
                }
            }
        };

        return new Chart(ctx, {
            type: 'pie',
            data: data,
            options: options
        });
    },

    // Initialize all charts
    initCharts: function() {
        // Initialize with default values if charts exist
        if (document.getElementById('riskGaugeChart')) {
            this.charts.riskGauge = this.createRiskGauge('riskGaugeChart', 0, 'Critical Risk');
        }
        if (document.getElementById('sentimentChart')) {
            this.charts.sentimentChart = this.createSentimentChart('sentimentChart', {
                positive: 0,
                neutral: 0,
                negative: 0
            });
        }
        if (document.getElementById('trendChart')) {
            this.charts.trendChart = this.createTrendChart('trendChart', {});
        }
        if (document.getElementById('categoryChart')) {
            this.charts.categoryChart = this.createCategoryChart('categoryChart', {});
        }
    },

    // Update Risk Gauge
    updateRiskGauge: function(value) {
        if (this.charts.riskGauge) {
            this.charts.riskGauge.data.datasets[0].data = [value, 100 - value];
            this.charts.riskGauge.data.datasets[0].backgroundColor[0] = 
                value > 70 ? '#dc3545' : value > 40 ? '#ffc107' : '#28a745';
            this.charts.riskGauge.update();
        }
    },

    // Alert Component
    showAlert: function(message, type = 'info', duration = 5000) {
        const alertHtml = `
            <div class="alert alert-${type} alert-dismissible fade show animated fadeInDown" role="alert">
                <i class="fas fa-${this.getAlertIcon(type)} me-2"></i>
                ${message}
                <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
            </div>
        `;

        const alertContainer = document.getElementById('alertContainer') || this.createAlertContainer();
        const alertElement = document.createElement('div');
        alertElement.innerHTML = alertHtml;
        alertContainer.appendChild(alertElement.firstElementChild);

        if (duration > 0) {
            setTimeout(() => {
                const alert = alertContainer.lastElementChild;
                if (alert) {
                    alert.classList.remove('fadeInDown');
                    alert.classList.add('fadeOutUp');
                    setTimeout(() => alert.remove(), 500);
                }
            }, duration);
        }
    },

    // Create alert container if it doesn't exist
    createAlertContainer: function() {
        const container = document.createElement('div');
        container.id = 'alertContainer';
        container.style.cssText = 'position: fixed; top: 20px; right: 20px; z-index: 9999; max-width: 400px;';
        document.body.appendChild(container);
        return container;
    },

    // Get alert icon based on type
    getAlertIcon: function(type) {
        const icons = {
            'success': 'check-circle',
            'danger': 'exclamation-triangle',
            'warning': 'exclamation-circle',
            'info': 'info-circle',
            'primary': 'star'
        };
        return icons[type] || 'info-circle';
    },

    // Loading Spinner
    showLoading: function(elementId) {
        const element = document.getElementById(elementId);
        if (element) {
            element.innerHTML = `
                <div class="text-center p-4">
                    <div class="spinner-border text-primary" role="status">
                        <span class="visually-hidden">Loading...</span>
                    </div>
                    <p class="mt-2 text-muted">Analyzing content...</p>
                </div>
            `;
        }
    },

    hideLoading: function(elementId) {
        const element = document.getElementById(elementId);
        if (element) {
            element.innerHTML = '';
        }
    },

    // Progress Bar
    updateProgress: function(elementId, value, label = '') {
        const element = document.getElementById(elementId);
        if (element) {
            const color = value > 70 ? 'danger' : value > 40 ? 'warning' : 'success';
            element.innerHTML = `
                <div class="progress" style="height: 25px;">
                    <div class="progress-bar bg-${color} progress-bar-striped progress-bar-animated" 
                         role="progressbar" 
                         style="width: ${value}%;" 
                         aria-valuenow="${value}" 
                         aria-valuemin="0" 
                         aria-valuemax="100">
                        ${label || value + '%'}
                    </div>
                </div>
            `;
        }
    },

    // Modal Management
    initModals: function() {
        // Initialize Bootstrap modals if they exist
        const modals = document.querySelectorAll('.modal');
        modals.forEach(modal => {
            new bootstrap.Modal(modal);
        });
    },

    showModal: function(modalId) {
        const modalElement = document.getElementById(modalId);
        if (modalElement) {
            const modal = bootstrap.Modal.getInstance(modalElement) || new bootstrap.Modal(modalElement);
            modal.show();
        }
    },

    hideModal: function(modalId) {
        const modalElement = document.getElementById(modalId);
        if (modalElement) {
            const modal = bootstrap.Modal.getInstance(modalElement);
            if (modal) modal.hide();
        }
    },

    // Results Display Component
    displayAnalysisResults: function(containerId, results) {
        const container = document.getElementById(containerId);
        if (!container) return;

        const riskClass = results.analysis.criticalRisk.level === 'high' ? 'danger' : 
                         results.analysis.criticalRisk.level === 'medium' ? 'warning' : 'success';

        const resultsHtml = `
            <div class="card border-${riskClass}">
                <div class="card-header bg-${riskClass} text-white">
                    <h5 class="mb-0">
                        <i class="fas fa-shield-alt me-2"></i>
                        Analysis Results
                    </h5>
                </div>
                <div class="card-body">
                    <div class="row">
                        <div class="col-md-6">
                            <h6>Critical Risk Score</h6>
                            <div class="d-flex align-items-center mb-3">
                                <div class="flex-grow-1 me-3">
                                    ${this.createProgressBar(results.analysis.criticalRisk.percentage, riskClass)}
                                </div>
                                <span class="badge bg-${riskClass}">${results.analysis.criticalRisk.percentage}%</span>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <h6>Overall Risk Score</h6>
                            <div class="d-flex align-items-center mb-3">
                                <div class="flex-grow-1 me-3">
                                    ${this.createProgressBar(results.analysis.overallRisk.percentage, 
                                        results.analysis.overallRisk.level === 'high' ? 'danger' : 
                                        results.analysis.overallRisk.level === 'medium' ? 'warning' : 'success')}
                                </div>
                                <span class="badge bg-secondary">${results.analysis.overallRisk.percentage}%</span>
                            </div>
                        </div>
                    </div>
                    
                    <div class="row mt-3">
                        <div class="col-md-12">
                            <h6>Recommendations</h6>
                            <div class="list-group">
                                ${results.recommendations.map(rec => `
                                    <div class="list-group-item list-group-item-${this.getRecommendationType(rec.type)}">
                                        <div class="d-flex w-100 justify-content-between">
                                            <h6 class="mb-1">${rec.message}</h6>
                                            <small>${rec.type}</small>
                                        </div>
                                        <p class="mb-1">${rec.action}</p>
                                    </div>
                                `).join('')}
                            </div>
                        </div>
                    </div>
                    
                    <div class="row mt-3">
                        <div class="col-md-6">
                            <h6>Sentiment Analysis</h6>
                            <ul class="list-unstyled">
                                <li><strong>Score:</strong> ${results.analysis.sentiment.score}</li>
                                <li><strong>Positive Words:</strong> ${results.analysis.sentiment.positive.length || 0}</li>
                                <li><strong>Negative Words:</strong> ${results.analysis.sentiment.negative.length || 0}</li>
                            </ul>
                        </div>
                        <div class="col-md-6">
                            <h6>Text Statistics</h6>
                            <ul class="list-unstyled">
                                <li><strong>Word Count:</strong> ${results.analysis.statistics.wordCount}</li>
                                <li><strong>High Risk Keywords:</strong> ${results.analysis.statistics.highRiskKeywords}</li>
                                <li><strong>Medium Risk Keywords:</strong> ${results.analysis.statistics.mediumRiskKeywords}</li>
                            </ul>
                        </div>
                    </div>
                </div>
                <div class="card-footer text-muted">
                    <small>Analyzed at: ${new Date(results.timestamp).toLocaleString()}</small>
                </div>
            </div>
        `;

        container.innerHTML = resultsHtml;
    },

    // Helper function to create progress bar HTML
    createProgressBar: function(value, colorClass) {
        return `
            <div class="progress" style="height: 20px;">
                <div class="progress-bar bg-${colorClass}" 
                     role="progressbar" 
                     style="width: ${value}%;" 
                     aria-valuenow="${value}" 
                     aria-valuemin="0" 
                     aria-valuemax="100">
                </div>
            </div>
        `;
    },

    // Get recommendation type for styling
    getRecommendationType: function(type) {
        const types = {
            'critical': 'danger',
            'warning': 'warning',
            'sentiment': 'info',
            'keywords': 'warning',
            'safe': 'success'
        };
        return types[type] || 'secondary';
    },

    // Bind global events
    bindEvents: function() {
        // Add any global event listeners here
        document.addEventListener('DOMContentLoaded', () => {
            console.log('Guardian LLM Components initialized');
        });
    },

    // Utility function to format numbers
    formatNumber: function(num) {
        return new Intl.NumberFormat().format(num);
    },

    // Utility function to format percentages
    formatPercentage: function(value, decimals = 1) {
        return value.toFixed(decimals) + '%';
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = Components;
}