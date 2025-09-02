/**
 * AtroZ Neural Network Visualization
 * Renders dynamic neural network connections with real-time weight data
 */

class AtroZNeuralNetwork {
    constructor(canvasId, config = {}) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.nodes = [];
        this.connections = [];
        
        // Configuration
        this.config = {
            nodeCount: config.nodeCount || 50,
            connectionThreshold: config.connectionThreshold || 0.3,
            layerCount: config.layerCount || 4,
            nodeSize: config.nodeSize || { min: 4, max: 12 },
            weightRange: config.weightRange || { min: -1, max: 1 },
            animationSpeed: config.animationSpeed || 0.02,
            ...config
        };
        
        // Data management
        this.dataSource = 'static';
        this.apiEndpoints = {};
        this.lastDataUpdate = 0;
        this.updateInterval = 30000;
        this.weightData = null;
        
        // Animation properties
        this.animationFrame = null;
        this.pulseTime = 0;
        this.dataFlowAnimations = [];
        
        // Metrics tracking
        this.metrics = {
            activeConnections: 0,
            averageWeight: 0,
            totalNodes: 0
        };
        
        this.setupCanvas();
        this.initializeNetwork();
        this.startAnimation();
    }
    
    setupCanvas() {
        const rect = this.canvas.getBoundingClientRect();
        this.canvas.width = rect.width * window.devicePixelRatio;
        this.canvas.height = rect.height * window.devicePixelRatio;
        this.ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
        
        this.canvas.style.width = rect.width + 'px';
        this.canvas.style.height = rect.height + 'px';
    }
    
    // Configure dynamic data sources
    configureDataSources(endpoints, updateInterval = 30000) {
        this.apiEndpoints = endpoints;
        this.updateInterval = updateInterval;
        this.dataSource = 'api';
    }
    
    // Initialize network with static/random data
    initializeNetwork() {
        this.createNodes();
        this.generateStaticConnections();
        this.updateMetrics();
    }
    
    createNodes() {
        this.nodes = [];
        const canvasWidth = this.canvas.width / window.devicePixelRatio;
        const canvasHeight = this.canvas.height / window.devicePixelRatio;
        
        const layerWidth = canvasWidth / (this.config.layerCount + 1);
        const nodesPerLayer = Math.ceil(this.config.nodeCount / this.config.layerCount);
        
        for (let layer = 0; layer < this.config.layerCount; layer++) {
            const layerNodeCount = Math.min(nodesPerLayer, this.config.nodeCount - layer * nodesPerLayer);
            const layerHeight = canvasHeight / (layerNodeCount + 1);
            
            for (let nodeIndex = 0; nodeIndex < layerNodeCount; nodeIndex++) {
                const node = {
                    id: `layer_${layer}_node_${nodeIndex}`,
                    x: layerWidth * (layer + 1),
                    y: layerHeight * (nodeIndex + 1),
                    layer: layer,
                    size: this.config.nodeSize.min + Math.random() * (this.config.nodeSize.max - this.config.nodeSize.min),
                    activation: Math.random(),
                    bias: (Math.random() - 0.5) * 2,
                    pulsePhase: Math.random() * Math.PI * 2,
                    connections: [],
                    type: layer === 0 ? 'input' : (layer === this.config.layerCount - 1 ? 'output' : 'hidden')
                };
                
                this.nodes.push(node);
                
                if (this.nodes.length >= this.config.nodeCount) break;
            }
            
            if (this.nodes.length >= this.config.nodeCount) break;
        }
    }
    
    generateStaticConnections() {
        this.connections = [];
        
        // Create connections between adjacent layers
        for (let i = 0; i < this.nodes.length; i++) {
            const sourceNode = this.nodes[i];
            
            // Connect to nodes in next layer
            const targetNodes = this.nodes.filter(node => node.layer === sourceNode.layer + 1);
            
            targetNodes.forEach(targetNode => {
                const weight = (Math.random() - 0.5) * 2; // Random weight between -1 and 1
                
                if (Math.abs(weight) >= this.config.connectionThreshold) {
                    const connection = {
                        id: `${sourceNode.id}_to_${targetNode.id}`,
                        source: sourceNode,
                        target: targetNode,
                        weight: weight,
                        strength: Math.abs(weight),
                        active: Math.random() > 0.3,
                        dataFlow: 0,
                        lastActivation: 0
                    };
                    
                    this.connections.push(connection);
                    sourceNode.connections.push(connection);
                }
            });
        }
    }
    
    // Load dynamic weight data from backend
    async loadDynamicData() {
        if (this.dataSource !== 'api' || !this.apiEndpoints.neural_weights) {
            return;
        }
        
        try {
            const response = await fetch(this.apiEndpoints.neural_weights);
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
            
            const weightData = await response.json();
            this.updateNetworkFromData(weightData);
            this.lastDataUpdate = Date.now();
            
        } catch (error) {
            console.warn('Failed to load neural weight data:', error);
            this.dataSource = 'static';
        }
    }
    
    // Update network with real backend data
    updateNetworkFromData(weightData) {
        if (!weightData || !weightData.weights) {
            console.warn('Invalid weight data structure');
            return;
        }
        
        // Update connection weights from API data
        weightData.weights.forEach(weightInfo => {
            const connection = this.connections.find(conn => 
                conn.id === weightInfo.connection_id || 
                (conn.source.id === weightInfo.source && conn.target.id === weightInfo.target)
            );
            
            if (connection) {
                connection.weight = weightInfo.weight;
                connection.strength = Math.abs(weightInfo.weight);
                connection.active = Math.abs(weightInfo.weight) >= this.config.connectionThreshold;
                connection.lastActivation = Date.now();
            }
        });
        
        // Update node activations if provided
        if (weightData.activations) {
            weightData.activations.forEach(activation => {
                const node = this.nodes.find(n => n.id === activation.node_id);
                if (node) {
                    node.activation = activation.value;
                    node.bias = activation.bias || node.bias;
                }
            });
        }
        
        this.updateMetrics();
        console.log(`Updated neural network with ${weightData.weights.length} weights`);
    }
    
    // Update network metrics
    updateMetrics() {
        const activeConnections = this.connections.filter(conn => conn.active);
        
        this.metrics = {
            activeConnections: activeConnections.length,
            averageWeight: this.connections.length > 0 ? 
                this.connections.reduce((sum, conn) => sum + Math.abs(conn.weight), 0) / this.connections.length : 0,
            totalNodes: this.nodes.length
        };
        
        // Update UI metrics
        const activeConnectionsEl = document.getElementById('active-connections');
        const avgWeightEl = document.getElementById('avg-weight');
        
        if (activeConnectionsEl) activeConnectionsEl.textContent = this.metrics.activeConnections;
        if (avgWeightEl) avgWeightEl.textContent = this.metrics.averageWeight.toFixed(3);
    }
    
    // Simulate neural network forward pass
    forwardPass() {
        // Reset activations for hidden and output layers
        this.nodes.forEach(node => {
            if (node.type !== 'input') {
                node.activation = node.bias;
            }
        });
        
        // Process each layer
        for (let layer = 0; layer < this.config.layerCount - 1; layer++) {
            const layerNodes = this.nodes.filter(node => node.layer === layer);
            
            layerNodes.forEach(sourceNode => {
                sourceNode.connections.forEach(connection => {
                    if (connection.active) {
                        const signal = sourceNode.activation * connection.weight;
                        connection.target.activation += signal;
                        
                        // Add data flow animation
                        if (Math.abs(signal) > 0.1) {
                            this.dataFlowAnimations.push({
                                connection: connection,
                                progress: 0,
                                intensity: Math.abs(signal)
                            });
                        }
                    }
                });
            });
            
            // Apply activation function (sigmoid) to next layer
            const nextLayerNodes = this.nodes.filter(node => node.layer === layer + 1);
            nextLayerNodes.forEach(node => {
                node.activation = this.sigmoid(node.activation);
            });
        }
    }
    
    sigmoid(x) {
        return 1 / (1 + Math.exp(-x));
    }
    
    // Update animations and data flows
    updateAnimations() {
        this.pulseTime += this.config.animationSpeed;
        
        // Update node pulse phases
        this.nodes.forEach(node => {
            node.pulsePhase += 0.05 + (node.activation * 0.1);
        });
        
        // Update data flow animations
        this.dataFlowAnimations = this.dataFlowAnimations.filter(flow => {
            flow.progress += 0.05;
            return flow.progress <= 1;
        });
        
        // Periodically run forward pass
        if (Math.floor(this.pulseTime * 10) % 30 === 0) {
            this.forwardPass();
        }
    }
    
    // Render the neural network
    render() {
        // Clear canvas
        this.ctx.fillStyle = 'rgba(26, 26, 46, 0.1)';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Draw connections
        this.connections.forEach(connection => {
            if (!connection.active) return;
            
            const opacity = connection.strength * 0.6;
            const weight = connection.weight;
            
            // Connection color based on weight (positive = blue, negative = red)
            const color = weight >= 0 ? 
                `rgba(0, 212, 255, ${opacity})` : 
                `rgba(255, 68, 68, ${opacity})`;
            
            this.ctx.strokeStyle = color;
            this.ctx.lineWidth = connection.strength * 3;
            this.ctx.beginPath();
            this.ctx.moveTo(connection.source.x, connection.source.y);
            this.ctx.lineTo(connection.target.x, connection.target.y);
            this.ctx.stroke();
        });
        
        // Draw data flow animations
        this.dataFlowAnimations.forEach(flow => {
            const { connection, progress, intensity } = flow;
            const x = connection.source.x + (connection.target.x - connection.source.x) * progress;
            const y = connection.source.y + (connection.target.y - connection.source.y) * progress;
            
            const gradient = this.ctx.createRadialGradient(x, y, 0, x, y, 8);
            gradient.addColorStop(0, `rgba(255, 255, 255, ${intensity})`);
            gradient.addColorStop(1, 'rgba(255, 255, 255, 0)');
            
            this.ctx.fillStyle = gradient;
            this.ctx.beginPath();
            this.ctx.arc(x, y, 6, 0, Math.PI * 2);
            this.ctx.fill();
        });
        
        // Draw nodes
        this.nodes.forEach(node => {
            const pulseSize = node.size + Math.sin(node.pulsePhase) * 2;
            const activation = node.activation;
            
            // Node color based on type and activation
            let baseColor = { r: 100, g: 150, b: 255 }; // Default blue
            if (node.type === 'input') baseColor = { r: 68, g: 255, b: 68 }; // Green
            else if (node.type === 'output') baseColor = { r: 255, g: 68, b: 68 }; // Red
            
            const intensity = 0.5 + activation * 0.5;
            const color = `rgba(${baseColor.r}, ${baseColor.g}, ${baseColor.b}, ${intensity})`;
            
            // Node glow effect
            const gradient = this.ctx.createRadialGradient(
                node.x, node.y, 0,
                node.x, node.y, pulseSize * 2
            );
            gradient.addColorStop(0, color);
            gradient.addColorStop(1, 'rgba(0, 0, 0, 0)');
            
            this.ctx.fillStyle = gradient;
            this.ctx.beginPath();
            this.ctx.arc(node.x, node.y, pulseSize * 1.5, 0, Math.PI * 2);
            this.ctx.fill();
            
            // Main node
            this.ctx.fillStyle = color;
            this.ctx.beginPath();
            this.ctx.arc(node.x, node.y, pulseSize, 0, Math.PI * 2);
            this.ctx.fill();
            
            // Node border
            this.ctx.strokeStyle = 'rgba(255, 255, 255, 0.5)';
            this.ctx.lineWidth = 1;
            this.ctx.stroke();
        });
    }
    
    // Animation loop
    animate() {
        this.updateAnimations();
        this.render();
        
        // Check for data updates
        if (this.dataSource === 'api' && Date.now() - this.lastDataUpdate > this.updateInterval) {
            this.loadDynamicData();
        }
        
        this.animationFrame = requestAnimationFrame(() => this.animate());
    }
    
    startAnimation() {
        if (this.animationFrame) {
            cancelAnimationFrame(this.animationFrame);
        }
        this.animate();
    }
    
    stopAnimation() {
        if (this.animationFrame) {
            cancelAnimationFrame(this.animationFrame);
            this.animationFrame = null;
        }
    }
    
    // Public methods
    refreshData() {
        if (this.dataSource === 'api') {
            this.loadDynamicData();
        } else {
            this.generateStaticConnections();
            this.updateMetrics();
        }
    }
    
    setConnectionThreshold(threshold) {
        this.config.connectionThreshold = threshold;
        this.connections.forEach(connection => {
            connection.active = connection.strength >= threshold;
        });
        this.updateMetrics();
    }
    
    getNetworkStats() {
        return {
            ...this.metrics,
            connectionDensity: this.connections.length / (this.nodes.length * this.nodes.length),
            layers: this.config.layerCount
        };
    }
    
    destroy() {
        this.stopAnimation();
    }
}