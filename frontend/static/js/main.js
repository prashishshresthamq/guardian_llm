// Guardian LLM - Main Application JavaScript
// Handles core functionality and page interactions

// Global application state
const App = {
    // Configuration
    config: {
        apiBaseUrl: '/api',
        maxTextLength: 5000,
        autoSaveInterval: 30000, // 30 seconds
        animationDuration: 300
    },

    // Application state
    state: {
        currentAnalysis: null,
        analysisHistory: [],
        isAnalyzing: false,
        autoSaveEnabled: true,
        darkMode: localStorage.getItem('darkMode') === 'true'
    },

    // Initialize application
    init: function() {
        console.log('Initializing Guardian LLM...');
        
        // Initialize components
        if (typeof Components !== 'undefined') {
            Components.init();
        }
        
        // Setup event listeners
        this.setupEventListeners();
        
        // Load saved data
        this.loadSavedData();
        
        // Initialize theme
        this.applyTheme();
        
        // Check API health
        this.checkAPIHealth();
        
        // Setup auto-save
        if (this.state.autoSaveEnabled) {
            this.setupAutoSave();
        }
        
        console.log('Guardian LLM initialized successfully');
    },

    // Setup all event listeners
    setupEventListeners: function() {
        // Text analysis form
        const analyzeForm = document.getElementById('analyzeForm');
        if (analyzeForm) {
            analyzeForm.addEventListener('submit', (e) => {
                e.preventDefault();
                this.analyzeText();
            });
        }

        // Text input character counter
        const textInput = document.getElementById('textInput');
        if (textInput) {
            textInput.addEventListener('input', (e) => {
                this.updateCharacterCount(e.target);
            });
        }

        // File upload
        const fileUpload = document.getElementById('fileUpload');
        if (fileUpload) {
            fileUpload.addEventListener('change', (e) => {
                this.handleFileUpload(e);
            });
        }

        // Clear button
        const clearBtn = document.getElementById('clearBtn');
        if (clearBtn) {
            clearBtn.addEventListener('click', () => {
                this.clearAnalysis();
            });
        }

        // Theme toggle
        const themeToggle = document.getElementById('themeToggle');
        if (themeToggle) {
            themeToggle.addEventListener('click', () => {
                this.toggleTheme();
            });
        }

        // Export results button
        const exportBtn = document.getElementById('exportBtn');
        if (exportBtn) {
            exportBtn.addEventListener('click', () => {
                this.exportResults();
            });
        }

        // Batch analysis button
        const batchBtn = document.getElementById('batchAnalyzeBtn');
        if (batchBtn) {
            batchBtn.addEventListener('click', () => {
                this.showBatchAnalysisModal();
            });
        }

        // Real-time analysis toggle
        const realtimeToggle = document.getElementById('realtimeAnalysis');
        if (realtimeToggle) {
            realtimeToggle.addEventListener('change', (e) => {
                this.toggleRealtimeAnalysis(e.target.checked);
            });
        }

        // History items
        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('history-item')) {
                const index = e.target.dataset.index;
                this.loadHistoryItem(index);
            }
        });
    },

    // Analyze text function
    analyzeText: async function() {
        const textInput = document.getElementById('textInput');
        const text = textInput ? textInput.value.trim() : '';

        if (!text) {
            Components.showAlert('Please enter some text to analyze', 'warning');
            return;
        }

        if (text.length > this.config.maxTextLength) {
            Components.showAlert(`Text exceeds maximum length of ${this.config.maxTextLength} characters`, 'warning');
            return;
        }

        // Set analyzing state
        this.state.isAnalyzing = true;
        this.updateUIState();

        try {
            // Show loading
            Components.showLoading('resultsArea');
            
            // Make API call
            const response = await fetch(`${this.config.apiBaseUrl}/analyze`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ text: text })
            });

            if (!response.ok) {
                throw new Error('Analysis failed');
            }

            const results = await response.json();
            
            // Store results
            this.state.currentAnalysis = results;
            this.addToHistory(results);
            
            // Display results
            this.displayResults(results);
            
            // Show success message
            Components.showAlert('Analysis completed successfully', 'success');
            
        } catch (error) {
            console.error('Analysis error:', error);
            Components.showAlert('An error occurred during analysis. Please try again.', 'danger');
        } finally {
            this.state.isAnalyzing = false;
            this.updateUIState();
            Components.hideLoading('resultsArea');
        }
    },

    // Display analysis results
    displayResults: function(results) {
        // Update risk gauges
        if (Components.charts.riskGauge) {
            Components.updateRiskGauge(results.analysis.criticalRisk.percentage);
        }

        // Update overall risk score
        const overallRiskEl = document.getElementById('overallRiskScore');
        if (overallRiskEl) {
            overallRiskEl.textContent = results.analysis.overallRisk.percentage + '%';
            overallRiskEl.className = 'h2 mb-0 text-' + this.getRiskColorClass(results.analysis.overallRisk.level);
        }

        // Update sentiment chart
        if (Components.charts.sentimentChart) {
            const sentimentData = {
                positive: results.analysis.sentiment.positive.length || 0,
                neutral: Math.max(0, results.analysis.statistics.wordCount - 
                    (results.analysis.sentiment.positive.length || 0) - 
                    (results.analysis.sentiment.negative.length || 0)),
                negative: results.analysis.sentiment.negative.length || 0
            };
            
            Components.charts.sentimentChart.data.datasets[0].data = [
                sentimentData.positive,
                sentimentData.neutral,
                sentimentData.negative
            ];
            Components.charts.sentimentChart.update();
        }

        // Display detailed results
        Components.displayAnalysisResults('resultsArea', results);

        // Update statistics
        this.updateStatistics(results);

        // Show recommendations
        this.displayRecommendations(results.recommendations);
    },

    // Update statistics display
    updateStatistics: function(results) {
        const stats = results.analysis.statistics;
        
        // Update stat cards
        const wordCountEl = document.getElementById('wordCount');
        if (wordCountEl) {
            wordCountEl.textContent = Components.formatNumber(stats.wordCount);
        }

        const highRiskEl = document.getElementById('highRiskCount');
        if (highRiskEl) {
            highRiskEl.textContent = stats.highRiskKeywords;
        }

        const sentimentScoreEl = document.getElementById('sentimentScore');
        if (sentimentScoreEl) {
            sentimentScoreEl.textContent = results.analysis.sentiment.score.toFixed(2);
        }
    },

    // Display recommendations
    displayRecommendations: function(recommendations) {
        const container = document.getElementById('recommendationsArea');
        if (!container) return;

        const html = recommendations.map(rec => {
            const icon = this.getRecommendationIcon(rec.type);
            const color = this.getRecommendationColor(rec.type);
            
            return `
                <div class="alert alert-${color} d-flex align-items-center" role="alert">
                    <i class="fas fa-${icon} fa-2x me-3"></i>
                    <div>
                        <h6 class="alert-heading mb-1">${rec.message}</h6>
                        <p class="mb-0">${rec.action}</p>
                    </div>
                </div>
            `;
        }).join('');

        container.innerHTML = html;
    },

    // Handle file upload
    handleFileUpload: function(event) {
        const file = event.target.files[0];
        if (!file) return;

        const reader = new FileReader();
        reader.onload = (e) => {
            const content = e.target.result;
            const textInput = document.getElementById('textInput');
            if (textInput) {
                textInput.value = content;
                this.updateCharacterCount(textInput);
                Components.showAlert('File loaded successfully', 'success');
            }
        };

        reader.onerror = () => {
            Components.showAlert('Error reading file', 'danger');
        };

        if (file.type.startsWith('text/')) {
            reader.readAsText(file);
        } else {
            Components.showAlert('Please upload a text file', 'warning');
        }
    },

    // Update character count
    updateCharacterCount: function(textarea) {
        const count = textarea.value.length;
        const maxLength = this.config.maxTextLength;
        const percentage = (count / maxLength) * 100;
        
        const countEl = document.getElementById('charCount');
        if (countEl) {
            countEl.textContent = `${count} / ${maxLength}`;
            countEl.className = percentage > 90 ? 'text-danger' : 
                               percentage > 70 ? 'text-warning' : 'text-muted';
        }

        // Update progress bar if exists
        const progressEl = document.getElementById('charProgress');
        if (progressEl) {
            progressEl.style.width = percentage + '%';
            progressEl.className = `progress-bar ${percentage > 90 ? 'bg-danger' : 
                                   percentage > 70 ? 'bg-warning' : 'bg-success'}`;
        }
    },

    // Clear analysis
    clearAnalysis: function() {
        const textInput = document.getElementById('textInput');
        if (textInput) {
            textInput.value = '';
            this.updateCharacterCount(textInput);
        }

        // Clear results
        const resultsArea = document.getElementById('resultsArea');
        if (resultsArea) {
            resultsArea.innerHTML = '<p class="text-muted text-center">No analysis results yet. Enter text above to begin.</p>';
        }

        // Reset charts
        if (Components.charts.riskGauge) {
            Components.updateRiskGauge(0);
        }

        // Clear recommendations
        const recommendationsArea = document.getElementById('recommendationsArea');
        if (recommendationsArea) {
            recommendationsArea.innerHTML = '';
        }

        this.state.currentAnalysis = null;
        Components.showAlert('Analysis cleared', 'info');
    },

    // Add to history
    addToHistory: function(analysis) {
        this.state.analysisHistory.unshift({
            timestamp: new Date().toISOString(),
            text: analysis.text.substring(0, 100) + '...',
            risk: analysis.analysis.criticalRisk.percentage,
            results: analysis
        });

        // Keep only last 10 items
        if (this.state.analysisHistory.length > 10) {
            this.state.analysisHistory = this.state.analysisHistory.slice(0, 10);
        }

        this.updateHistoryDisplay();
        this.saveData();
    },

    // Update history display
    updateHistoryDisplay: function() {
        const historyList = document.getElementById('historyList');
        if (!historyList) return;

        const html = this.state.analysisHistory.map((item, index) => {
            const date = new Date(item.timestamp);
            const riskClass = item.risk > 70 ? 'danger' : item.risk > 40 ? 'warning' : 'success';
            
            return `
                <div class="history-item list-group-item list-group-item-action" data-index="${index}">
                    <div class="d-flex w-100 justify-content-between">
                        <h6 class="mb-1">${item.text}</h6>
                        <small>${date.toLocaleTimeString()}</small>
                    </div>
                    <p class="mb-1">
                        <span class="badge bg-${riskClass}">Risk: ${item.risk}%</span>
                    </p>
                </div>
            `;
        }).join('');

        historyList.innerHTML = html || '<p class="text-muted text-center">No history yet</p>';
    },

    // Load history item
    loadHistoryItem: function(index) {
        const item = this.state.analysisHistory[index];
        if (!item) return;

        const textInput = document.getElementById('textInput');
        if (textInput) {
            textInput.value = item.results.text;
            this.updateCharacterCount(textInput);
        }

        this.displayResults(item.results);
        Components.showAlert('History item loaded', 'info');
    },

    // Export results
    exportResults: function() {
        if (!this.state.currentAnalysis) {
            Components.showAlert('No analysis results to export', 'warning');
            return;
        }

        const data = {
            exportDate: new Date().toISOString(),
            analysis: this.state.currentAnalysis,
            metadata: {
                version: '1.0',
                tool: 'Guardian LLM'
            }
        };

        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `guardian-llm-analysis-${Date.now()}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

        Components.showAlert('Results exported successfully', 'success');
    },

    // Batch analysis modal
    showBatchAnalysisModal: function() {
        const modalHtml = `
            <div class="modal fade" id="batchModal" tabindex="-1">
                <div class="modal-dialog modal-lg">
                    <div class="modal-content">
                        <div class="modal-header">
                            <h5 class="modal-title">Batch Text Analysis</h5>
                            <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                        </div>
                        <div class="modal-body">
                            <div class="mb-3">
                                <label for="batchTextarea" class="form-label">Enter multiple texts (one per line)</label>
                                <textarea class="form-control" id="batchTextarea" rows="10" 
                                          placeholder="Enter text 1&#10;Enter text 2&#10;Enter text 3"></textarea>
                            </div>
                            <div class="alert alert-info">
                                <i class="fas fa-info-circle me-2"></i>
                                Enter each text on a new line. Empty lines will be ignored.
                            </div>
                        </div>
                        <div class="modal-footer">
                            <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Cancel</button>
                            <button type="button" class="btn btn-primary" onclick="App.runBatchAnalysis()">
                                <i class="fas fa-play me-2"></i>Run Analysis
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        `;

        // Add modal to page if not exists
        if (!document.getElementById('batchModal')) {
            document.body.insertAdjacentHTML('beforeend', modalHtml);
        }

        Components.showModal('batchModal');
    },

    // Run batch analysis
    runBatchAnalysis: async function() {
        const textarea = document.getElementById('batchTextarea');
        if (!textarea) return;

        const texts = textarea.value.split('\n').filter(text => text.trim().length > 0);
        
        if (texts.length === 0) {
            Components.showAlert('Please enter at least one text', 'warning');
            return;
        }

        Components.hideModal('batchModal');
        Components.showLoading('resultsArea');

        try {
            const response = await fetch(`${this.config.apiBaseUrl}/analyze/batch`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ texts: texts })
            });

            if (!response.ok) {
                throw new Error('Batch analysis failed');
            }

            const results = await response.json();
            this.displayBatchResults(results);
            
            Components.showAlert(`Batch analysis completed for ${results.count} texts`, 'success');
            
        } catch (error) {
            console.error('Batch analysis error:', error);
            Components.showAlert('An error occurred during batch analysis', 'danger');
        } finally {
            Components.hideLoading('resultsArea');
        }
    },

    // Display batch results
    displayBatchResults: function(results) {
        const html = `
            <div class="card">
                <div class="card-header">
                    <h5 class="mb-0">Batch Analysis Results</h5>
                </div>
                <div class="card-body">
                    <div class="row mb-3">
                        <div class="col-md-4">
                            <div class="text-center">
                                <h6>Total Analyzed</h6>
                                <p class="h3">${results.count}</p>
                            </div>
                        </div>
                        <div class="col-md-4">
                            <div class="text-center">
                                <h6>Average Risk</h6>
                                <p class="h3 text-${this.getRiskColorClass(
                                    results.summary.averageCriticalRisk > 0.7 ? 'high' : 
                                    results.summary.averageCriticalRisk > 0.4 ? 'medium' : 'low'
                                )}">
                                    ${Math.round(results.summary.averageCriticalRisk * 100)}%
                                </p>
                            </div>
                        </div>
                        <div class="col-md-4">
                            <div class="text-center">
                                <h6>High Risk Count</h6>
                                <p class="h3 text-danger">${results.summary.highRiskCount}</p>
                            </div>
                        </div>
                    </div>
                    
                    <h6>Individual Results</h6>
                    <div class="table-responsive">
                        <table class="table table-hover">
                            <thead>
                                <tr>
                                    <th>#</th>
                                    <th>Text Preview</th>
                                    <th>Critical Risk</th>
                                    <th>Overall Risk</th>
                                    <th>Sentiment</th>
                                </tr>
                            </thead>
                            <tbody>
                                ${results.results.map((item, index) => `
                                    <tr>
                                        <td>${index + 1}</td>
                                        <td>${item.text.substring(0, 50)}...</td>
                                        <td>
                                            <span class="badge bg-${this.getRiskColorClass(
                                                item.criticalRisk > 0.7 ? 'high' : 
                                                item.criticalRisk > 0.4 ? 'medium' : 'low'
                                            )}">
                                                ${Math.round(item.criticalRisk * 100)}%
                                            </span>
                                        </td>
                                        <td>${Math.round(item.overallRisk * 100)}%</td>
                                        <td>${item.sentiment.toFixed(2)}</td>
                                    </tr>
                                `).join('')}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        `;

        const resultsArea = document.getElementById('resultsArea');
        if (resultsArea) {
            resultsArea.innerHTML = html;
        }
    },

    // Toggle real-time analysis
    toggleRealtimeAnalysis: function(enabled) {
        if (enabled) {
            const textInput = document.getElementById('textInput');
            if (textInput) {
                let timeout;
                textInput.addEventListener('input', (e) => {
                    clearTimeout(timeout);
                    timeout = setTimeout(() => {
                        if (e.target.value.trim().length > 50) {
                            this.analyzeText();
                        }
                    }, 1000);
                });
            }
        }
        
        localStorage.setItem('realtimeAnalysis', enabled);
        Components.showAlert(`Real-time analysis ${enabled ? 'enabled' : 'disabled'}`, 'info');
    },

    // Theme management
    toggleTheme: function() {
        this.state.darkMode = !this.state.darkMode;
        this.applyTheme();
        localStorage.setItem('darkMode', this.state.darkMode);
    },

    applyTheme: function() {
        if (this.state.darkMode) {
            document.body.classList.add('dark-mode');
            document.documentElement.setAttribute('data-bs-theme', 'dark');
        } else {
            document.body.classList.remove('dark-mode');
            document.documentElement.setAttribute('data-bs-theme', 'light');
        }

        // Update theme icon
        const themeIcon = document.querySelector('#themeToggle i');
        if (themeIcon) {
            themeIcon.className = this.state.darkMode ? 'fas fa-sun' : 'fas fa-moon';
        }
    },

    // Update UI state
    updateUIState: function() {
        const analyzeBtn = document.getElementById('analyzeBtn');
        if (analyzeBtn) {
            analyzeBtn.disabled = this.state.isAnalyzing;
            analyzeBtn.innerHTML = this.state.isAnalyzing ? 
                '<span class="spinner-border spinner-border-sm me-2"></span>Analyzing...' : 
                '<i class="fas fa-search me-2"></i>Analyze';
        }
    },

    // Check API health
    checkAPIHealth: async function() {
        try {
            const response = await fetch(`${this.config.apiBaseUrl}/health`);
            if (response.ok) {
                const health = await response.json();
                console.log('API Health:', health);
                
                // Update status indicator if exists
                const statusEl = document.getElementById('apiStatus');
                if (statusEl) {
                    statusEl.className = 'badge bg-success';
                    statusEl.textContent = 'API Online';
                }
            }
        } catch (error) {
            console.error('API health check failed:', error);
            const statusEl = document.getElementById('apiStatus');
            if (statusEl) {
                statusEl.className = 'badge bg-danger';
                statusEl.textContent = 'API Offline';
            }
        }
    },

    // Save and load data
    saveData: function() {
        const data = {
            history: this.state.analysisHistory,
            settings: {
                darkMode: this.state.darkMode,
                autoSave: this.state.autoSaveEnabled
            }
        };
        localStorage.setItem('guardianLLMData', JSON.stringify(data));
    },

    loadSavedData: function() {
        const saved = localStorage.getItem('guardianLLMData');
        if (saved) {
            try {
                const data = JSON.parse(saved);
                this.state.analysisHistory = data.history || [];
                this.state.darkMode = data.settings?.darkMode || false;
                this.state.autoSaveEnabled = data.settings?.autoSave !== false;
                
                this.updateHistoryDisplay();
            } catch (error) {
                console.error('Error loading saved data:', error);
            }
        }
    },

    // Setup auto-save
    setupAutoSave: function() {
        setInterval(() => {
            if (this.state.currentAnalysis) {
                this.saveData();
            }
        }, this.config.autoSaveInterval);
    },

    // Helper functions
    getRiskColorClass: function(level) {
        const colors = {
            'high': 'danger',
            'medium': 'warning',
            'low': 'success'
        };
        return colors[level] || 'secondary';
    },

    getRecommendationIcon: function(type) {
        const icons = {
            'critical': 'exclamation-triangle',
            'warning': 'exclamation-circle',
            'sentiment': 'brain',
            'keywords': 'key',
            'safe': 'shield-alt'
        };
        return icons[type] || 'info-circle';
    },

    getRecommendationColor: function(type) {
        const colors = {
            'critical': 'danger',
            'warning': 'warning',
            'sentiment': 'info',
            'keywords': 'warning',
            'safe': 'success'
        };
        return colors[type] || 'secondary';
    }
};

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    App.init();
});

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = App;
}