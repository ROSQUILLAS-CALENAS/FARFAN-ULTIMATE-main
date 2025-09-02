/**
 * AtroZ Dashboard Main Controller
 * Orchestrates particle system and neural network with dynamic data integration
 */

class AtroZDashboard {
    constructor() {
        this.particleSystem = null;
        this.neuralNetwork = null;
        this.config = null;
        this.updateTimer = null;
        
        this.init();
    }
    
    async init() {
        try {
            // Load configuration from embedded JSON
            this.loadConfiguration();
            
            // Initialize visualization components
            this.initializeParticleSystem();
            this.initializeNeuralNetwork();
            
            // Setup event listeners
            this.setupEventListeners();
            
            // Start data loading if API endpoints are configured
            if (this.config && this.config.api_endpoints) {
                this.setupDynamicDataSources();
                await this.initialDataLoad();
            }
            
            console.log('AtroZ Dashboard initialized successfully');
            
        } catch (error) {
            console.error('Failed to initialize AtroZ Dashboard:', error);
            this.showErrorMessage('Dashboard initialization failed. Using static data mode.');
        }
    }
    
    loadConfiguration() {
        const configElement = document.getElementById('dashboard-config');
        if (configElement) {
            try {
                this.config = JSON.parse(configElement.textContent);
                console.log('Loaded dashboard configuration:', this.config);
            } catch (error) {
                console.warn('Failed to parse dashboard configuration:', error);
                this.config = this.getDefaultConfiguration();
            }
        } else {
            console.log('No configuration found, using defaults');
            this.config = this.getDefaultConfiguration();
        }
    }
    
    getDefaultConfiguration() {
        return {
            api_endpoints: {
                pdet_regions: '/api/analysis/pdet-regions',
                evidence_scores: '/api/analysis/evidence-scores',
                neural_weights: '/api/analysis/neural-weights'
            },
            update_interval: 30000,
            particle_config: {
                max_particles: 200,
                evidence_color_map: {
                    high: '#ff4444',
                    medium: '#ffaa44',
                    low: '#44ff44'
                }
            },
            neural_config: {
                node_count: 50,
                connection_threshold: 0.3
            }
        };
    }
    
    initializeParticleSystem() {
        this.particleSystem = new AtroZParticleSystem('particle-canvas', {
            maxParticles: this.config.particle_config.max_particles,
            evidenceColorMap: this.config.particle_config.evidence_color_map,
            particleSize: { min: 3, max: 10 },
            connectionDistance: 100,
            animationSpeed: 0.015
        });
        
        console.log('Particle system initialized');
    }
    
    initializeNeuralNetwork() {
        this.neuralNetwork = new AtroZNeuralNetwork('neural-canvas', {
            nodeCount: this.config.neural_config.node_count,
            connectionThreshold: this.config.neural_config.connection_threshold,
            layerCount: 4,
            nodeSize: { min: 5, max: 15 },
            animationSpeed: 0.02
        });
        
        console.log('Neural network initialized');
    }
    
    setupDynamicDataSources() {
        if (!this.config.api_endpoints) return;
        
        // Configure particle system data sources
        this.particleSystem.configureDataSources({
            pdet_regions: this.config.api_endpoints.pdet_regions,
            evidence_scores: this.config.api_endpoints.evidence_scores
        }, this.config.update_interval);
        
        // Configure neural network data sources  
        this.neuralNetwork.configureDataSources({
            neural_weights: this.config.api_endpoints.neural_weights
        }, this.config.update_interval);
        
        console.log('Dynamic data sources configured');
    }
    
    async initialDataLoad() {
        try {
            document.getElementById('data-source').textContent = 'Loading Initial Data...';
            
            // Load data for both systems in parallel
            await Promise.all([
                this.particleSystem.loadDynamicData(),
                this.neuralNetwork.loadDynamicData()
            ]);
            
            document.getElementById('data-source').textContent = 'Live Data';
            
        } catch (error) {
            console.error('Initial data load failed:', error);
            document.getElementById('data-source').textContent = 'Static Data (API Error)';
        }
    }
    
    setupEventListeners() {
        // Refresh data button
        const refreshButton = document.getElementById('refresh-data');
        if (refreshButton) {
            refreshButton.addEventListener('click', () => this.refreshAllData());
        }
        
        // Evidence threshold slider
        const evidenceSlider = document.getElementById('evidence-threshold');
        const thresholdValue = document.getElementById('threshold-value');
        
        if (evidenceSlider && thresholdValue) {
            evidenceSlider.addEventListener('input', (e) => {
                const threshold = parseFloat(e.target.value);
                thresholdValue.textContent = threshold.toFixed(1);
                this.particleSystem.setEvidenceThreshold(threshold);
            });
        }
        
        // Window resize handling
        window.addEventListener('resize', () => {
            if (this.particleSystem) this.particleSystem.setupCanvas();
            if (this.neuralNetwork) this.neuralNetwork.setupCanvas();
        });
        
        // Periodic data updates
        if (this.config.update_interval && this.config.api_endpoints) {
            this.updateTimer = setInterval(() => {
                this.refreshAllData();
            }, this.config.update_interval);
        }
    }
    
    async refreshAllData() {
        const refreshButton = document.getElementById('refresh-data');
        const dataSource = document.getElementById('data-source');
        
        // Update UI state
        if (refreshButton) {
            refreshButton.disabled = true;
            refreshButton.textContent = 'Refreshing...';
        }
        
        if (dataSource) {
            dataSource.classList.add('loading');
        }
        
        try {
            // Refresh both visualization systems
            await Promise.all([
                this.particleSystem.refreshData(),
                this.neuralNetwork.refreshData()
            ]);
            
            console.log('Data refresh completed successfully');
            
        } catch (error) {
            console.error('Data refresh failed:', error);
            this.showErrorMessage('Data refresh failed. Some visualizations may show stale data.');
            
        } finally {
            // Restore UI state
            if (refreshButton) {
                refreshButton.disabled = false;
                refreshButton.textContent = 'Refresh Data';
            }
            
            if (dataSource) {
                dataSource.classList.remove('loading');
            }
        }
    }
    
    showErrorMessage(message) {
        // Create or update error message display
        let errorDiv = document.getElementById('error-message');
        if (!errorDiv) {
            errorDiv = document.createElement('div');
            errorDiv.id = 'error-message';
            errorDiv.style.cssText = `
                position: fixed;
                top: 20px;
                right: 20px;
                background: rgba(255, 68, 68, 0.9);
                color: white;
                padding: 1rem;
                border-radius: 5px;
                z-index: 1000;
                max-width: 300px;
            `;
            document.body.appendChild(errorDiv);
        }
        
        errorDiv.textContent = message;
        
        // Auto-hide after 5 seconds
        setTimeout(() => {
            if (errorDiv.parentNode) {
                errorDiv.parentNode.removeChild(errorDiv);
            }
        }, 5000);
    }
    
    // Public API methods
    
    // Configure data attributes from backend template
    configureFromDataAttributes(element) {
        const pdetData = element.dataset.pdetRegions;
        const evidenceData = element.dataset.evidenceScores;
        const neuralData = element.dataset.neuralWeights;
        
        if (pdetData) {
            try {
                const regions = JSON.parse(pdetData);
                this.particleSystem.updateParticlesFromData(regions, 
                    evidenceData ? JSON.parse(evidenceData) : null);
            } catch (error) {
                console.warn('Failed to parse PDET region data:', error);
            }
        }
        
        if (neuralData) {
            try {
                const weights = JSON.parse(neuralData);
                this.neuralNetwork.updateNetworkFromData(weights);
            } catch (error) {
                console.warn('Failed to parse neural weight data:', error);
            }
        }
    }
    
    // Update configuration at runtime
    updateConfiguration(newConfig) {
        this.config = { ...this.config, ...newConfig };
        
        if (newConfig.api_endpoints) {
            this.setupDynamicDataSources();
        }
        
        if (newConfig.update_interval) {
            if (this.updateTimer) {
                clearInterval(this.updateTimer);
            }
            this.updateTimer = setInterval(() => {
                this.refreshAllData();
            }, newConfig.update_interval);
        }
    }
    
    // Get current dashboard statistics
    getDashboardStats() {
        return {
            particles: {
                total: this.particleSystem.particles.length,
                active: this.particleSystem.getActiveParticles().length,
                connections: this.particleSystem.connections.length
            },
            neural: this.neuralNetwork.getNetworkStats(),
            dataSource: this.particleSystem.dataSource,
            lastUpdate: this.particleSystem.lastDataUpdate
        };
    }
    
    // Cleanup and destroy
    destroy() {
        if (this.updateTimer) {
            clearInterval(this.updateTimer);
        }
        
        if (this.particleSystem) {
            this.particleSystem.destroy();
        }
        
        if (this.neuralNetwork) {
            this.neuralNetwork.destroy();
        }
        
        console.log('AtroZ Dashboard destroyed');
    }
}

// Global dashboard instance
let atroZDashboard = null;

// Initialize dashboard when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    atroZDashboard = new AtroZDashboard();
    
    // Expose dashboard to global scope for debugging/external control
    window.atroZDashboard = atroZDashboard;
});

// Handle page unload
window.addEventListener('beforeunload', () => {
    if (atroZDashboard) {
        atroZDashboard.destroy();
    }
});