/**
 * AtroZ Data Binding Usage Example
 * Demonstrates how to initialize and use the data binding module
 */

// Example API response structure for reference
const exampleApiResponse = {
    pdetRegions: [
        {
            id: 'pdet-001',
            type: 'analysis',
            content: 'Pattern detected in sector 7',
            confidence: 0.89,
            coordinates: { x: 120, y: 85 },
            animate: true
        },
        {
            id: 'pdet-002', 
            type: 'anomaly',
            content: 'Anomalous reading detected',
            confidence: 0.76,
            coordinates: { x: 240, y: 165 },
            animate: false
        }
    ],
    constellationData: {
        nodes: [
            {
                id: 'node-001',
                label: 'Data Cluster A',
                weight: 2.3,
                category: 'primary',
                position: { x: 300, y: 200 },
                highlight: true
            },
            {
                id: 'node-002',
                label: 'Data Cluster B', 
                weight: 1.7,
                category: 'secondary',
                position: { x: 450, y: 180 },
                highlight: false
            }
        ],
        connections: [
            {
                id: 'conn-001',
                source: 'node-001',
                target: 'node-002',
                strength: 0.8,
                coordinates: { x1: 300, y1: 200, x2: 450, y2: 180 }
            }
        ]
    },
    evidenceItems: [
        {
            id: 'evidence-001',
            title: 'Pattern Analysis Result',
            content: 'Significant correlation found between variables X and Y',
            pageReference: 42,
            timestamp: '2024-01-15T10:30:00Z',
            confidence: 0.92,
            source: 'analysis_engine',
            isNew: true
        },
        {
            id: 'evidence-002',
            title: 'Historical Comparison',
            content: 'Current patterns match 73% with historical data from Q3',
            pageReference: 38,
            timestamp: '2024-01-15T10:25:00Z',
            confidence: 0.73,
            source: 'comparison_engine',
            isNew: false
        }
    ]
};

// Initialize the data binding system
document.addEventListener('DOMContentLoaded', function() {
    
    // Configuration options
    const config = {
        apiEndpoint: '/api/analysis-results',
        updateInterval: 30000, // Update every 30 seconds
        retryAttempts: 3,
        retryDelay: 1000
    };

    // Create the data binding instance
    const atroZBinding = new AtroZDataBinding(config);

    // Set up event listeners for system events
    atroZBinding.on('initialized', (event) => {
        console.log('AtroZ Data Binding initialized successfully');
        showStatus('System initialized', 'success');
    });

    atroZBinding.on('dataFetched', (event) => {
        const data = event.detail;
        console.log('New data fetched:', data);
        updateDataTimestamp();
    });

    atroZBinding.on('updated', (event) => {
        console.log('Dashboard components updated');
        showStatus('Data updated', 'info');
    });

    atroZBinding.on('pdetUpdated', (event) => {
        const regions = event.detail;
        console.log(`Updated ${regions.length} PDET regions`);
        updateRegionCounter(regions.length);
    });

    atroZBinding.on('constellationUpdated', (event) => {
        const constellation = event.detail;
        console.log(`Updated constellation with ${constellation.nodes?.length || 0} nodes`);
        updateNodeCounter(constellation.nodes?.length || 0);
    });

    atroZBinding.on('evidenceUpdated', (event) => {
        const evidence = event.detail;
        console.log(`Updated ${evidence.length} evidence items`);
        updateEvidenceCounter(evidence.length);
    });

    atroZBinding.on('error', (event) => {
        const error = event.detail;
        console.error('AtroZ Data Binding error:', error);
        showStatus(`Error: ${error.type}`, 'error');
    });

    atroZBinding.on('networkStatus', (event) => {
        const status = event.detail;
        console.log('Network status:', status.online ? 'online' : 'offline');
        showNetworkStatus(status.online);
    });

    // Control panel event handlers
    const setupControls = () => {
        // Force update button
        const forceUpdateBtn = document.getElementById('force-update-btn');
        if (forceUpdateBtn) {
            forceUpdateBtn.addEventListener('click', async () => {
                showStatus('Forcing update...', 'info');
                await atroZBinding.forceUpdate();
            });
        }

        // Update interval control
        const intervalSelect = document.getElementById('update-interval');
        if (intervalSelect) {
            intervalSelect.addEventListener('change', (e) => {
                const newInterval = parseInt(e.target.value);
                atroZBinding.setUpdateInterval(newInterval);
                showStatus(`Update interval set to ${newInterval/1000}s`, 'info');
            });
        }

        // Pause/resume controls
        const pauseBtn = document.getElementById('pause-updates-btn');
        const resumeBtn = document.getElementById('resume-updates-btn');
        
        if (pauseBtn) {
            pauseBtn.addEventListener('click', () => {
                atroZBinding.pauseAutoUpdate();
                showStatus('Auto-updates paused', 'warning');
            });
        }

        if (resumeBtn) {
            resumeBtn.addEventListener('click', () => {
                atroZBinding.resumeAutoUpdate();
                showStatus('Auto-updates resumed', 'success');
            });
        }
    };

    // Utility functions for UI updates
    function showStatus(message, type = 'info') {
        const statusElement = document.getElementById('system-status');
        if (statusElement) {
            statusElement.textContent = message;
            statusElement.className = `status status-${type}`;
            
            // Auto-clear info messages
            if (type === 'info') {
                setTimeout(() => {
                    statusElement.textContent = 'Ready';
                    statusElement.className = 'status';
                }, 3000);
            }
        }
    }

    function updateDataTimestamp() {
        const timestampElement = document.getElementById('last-update-time');
        if (timestampElement) {
            timestampElement.textContent = new Date().toLocaleTimeString();
        }
    }

    function updateRegionCounter(count) {
        const counterElement = document.getElementById('pdet-region-count');
        if (counterElement) {
            counterElement.textContent = count;
        }
    }

    function updateNodeCounter(count) {
        const counterElement = document.getElementById('constellation-node-count');
        if (counterElement) {
            counterElement.textContent = count;
        }
    }

    function updateEvidenceCounter(count) {
        const counterElement = document.getElementById('evidence-item-count');
        if (counterElement) {
            counterElement.textContent = count;
        }
    }

    function showNetworkStatus(online) {
        const statusElement = document.getElementById('network-status');
        if (statusElement) {
            statusElement.textContent = online ? 'Online' : 'Offline';
            statusElement.className = `network-status ${online ? 'online' : 'offline'}`;
        }
    }

    // Initialize controls after a short delay to ensure DOM is ready
    setTimeout(setupControls, 100);

    // Make the binding instance globally available for debugging
    window.atroZBinding = atroZBinding;

    // Cleanup on page unload
    window.addEventListener('beforeunload', () => {
        atroZBinding.destroy();
    });
});

// Mock API server for testing (remove in production)
if (window.location.hostname === 'localhost' && !window.mockApiServer) {
    window.mockApiServer = true;
    
    // Intercept fetch requests for testing
    const originalFetch = window.fetch;
    window.fetch = function(url, options) {
        if (url.includes('/api/analysis-results')) {
            return new Promise((resolve) => {
                setTimeout(() => {
                    resolve({
                        ok: true,
                        json: () => Promise.resolve(generateMockData())
                    });
                }, 500); // Simulate network delay
            });
        }
        return originalFetch.apply(this, arguments);
    };

    function generateMockData() {
        const now = new Date().toISOString();
        return {
            pdetRegions: [
                {
                    id: `pdet-${Date.now()}-1`,
                    type: 'analysis',
                    content: `Analysis complete at ${new Date().toLocaleTimeString()}`,
                    confidence: Math.random() * 0.5 + 0.5,
                    coordinates: { 
                        x: Math.random() * 400 + 50, 
                        y: Math.random() * 300 + 50 
                    },
                    animate: Math.random() > 0.5
                },
                {
                    id: `pdet-${Date.now()}-2`,
                    type: 'anomaly',
                    content: 'Anomaly detected in data stream',
                    confidence: Math.random() * 0.5 + 0.5,
                    coordinates: { 
                        x: Math.random() * 400 + 50, 
                        y: Math.random() * 300 + 50 
                    },
                    animate: Math.random() > 0.7
                }
            ],
            constellationData: {
                nodes: Array.from({length: 5}, (_, i) => ({
                    id: `node-${Date.now()}-${i}`,
                    label: `Cluster ${String.fromCharCode(65 + i)}`,
                    weight: Math.random() * 3 + 0.5,
                    category: i % 2 === 0 ? 'primary' : 'secondary',
                    position: { 
                        x: Math.random() * 500 + 100, 
                        y: Math.random() * 400 + 100 
                    },
                    highlight: Math.random() > 0.8
                })),
                connections: []
            },
            evidenceItems: Array.from({length: 3}, (_, i) => ({
                id: `evidence-${Date.now()}-${i}`,
                title: `Evidence Item ${i + 1}`,
                content: `Generated evidence content at ${new Date().toLocaleTimeString()}`,
                pageReference: Math.floor(Math.random() * 100) + 1,
                timestamp: now,
                confidence: Math.random() * 0.5 + 0.5,
                source: 'mock_engine',
                isNew: i === 0
            }))
        };
    }
}