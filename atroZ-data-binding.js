/**
 * AtroZ Dashboard Data Binding Module
 * Handles dynamic population of dashboard elements with backend data
 * while preserving existing CSS classes and animations
 */

class AtroZDataBinding {
    constructor(config = {}) {
        this.apiEndpoint = config.apiEndpoint || '/api/analysis-results';
        this.updateInterval = config.updateInterval || 30000; // 30 seconds
        this.retryAttempts = config.retryAttempts || 3;
        this.retryDelay = config.retryDelay || 1000;
        
        this.eventBus = new EventTarget();
        this.isInitialized = false;
        this.updateTimer = null;
        
        this.selectors = {
            pdetContainer: '.pdet-container',
            pdetHexagon: '.pdet-hexagon',
            constellationContainer: '.constellation-container', 
            constellationNode: '.constellation-node',
            evidenceStream: '.evidence-stream',
            evidenceItem: '.evidence-item'
        };
        
        this.init();
    }

    async init() {
        try {
            await this.loadInitialData();
            this.setupEventListeners();
            this.startAutoUpdate();
            this.isInitialized = true;
            this.emit('initialized');
        } catch (error) {
            console.error('AtroZ Data Binding initialization failed:', error);
            this.emit('error', { type: 'initialization', error });
        }
    }

    async loadInitialData() {
        const data = await this.fetchData();
        if (data) {
            await this.updateAllComponents(data);
        }
    }

    async fetchData(attempt = 1) {
        try {
            const response = await fetch(this.apiEndpoint, {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json',
                    'Cache-Control': 'no-cache'
                }
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();
            this.emit('dataFetched', data);
            return data;

        } catch (error) {
            if (attempt < this.retryAttempts) {
                await this.delay(this.retryDelay * attempt);
                return this.fetchData(attempt + 1);
            }
            
            console.error('Failed to fetch data after', this.retryAttempts, 'attempts:', error);
            this.emit('error', { type: 'fetch', error, attempt });
            return null;
        }
    }

    async updateAllComponents(data) {
        const updatePromises = [];

        if (data.pdetRegions) {
            updatePromises.push(this.updatePdetNodes(data.pdetRegions));
        }

        if (data.constellationData) {
            updatePromises.push(this.updateConstellationNetwork(data.constellationData));
        }

        if (data.evidenceItems) {
            updatePromises.push(this.updateEvidenceStream(data.evidenceItems));
        }

        try {
            await Promise.all(updatePromises);
            this.emit('updated', data);
        } catch (error) {
            console.error('Component update failed:', error);
            this.emit('error', { type: 'update', error });
        }
    }

    async updatePdetNodes(pdetRegions) {
        const container = document.querySelector(this.selectors.pdetContainer);
        if (!container) return;

        pdetRegions.forEach((region, index) => {
            const existingNode = container.querySelector(`[data-pdet-id="${region.id}"]`);
            
            if (existingNode) {
                this.updatePdetNode(existingNode, region);
            } else {
                this.createPdetNode(container, region, index);
            }
        });

        this.emit('pdetUpdated', pdetRegions);
    }

    updatePdetNode(node, region) {
        // Preserve existing classes and data attributes
        const preservedClasses = node.className;
        const preservedDataAttrs = this.getDataAttributes(node);

        // Update content without disrupting structure
        const contentElement = node.querySelector('.pdet-content') || node;
        contentElement.innerHTML = this.sanitizeHTML(region.content || '');

        // Update position if provided
        if (region.coordinates) {
            node.style.transform = `translate(${region.coordinates.x}px, ${region.coordinates.y}px)`;
        }

        // Update data attributes for new data
        node.setAttribute('data-pdet-id', region.id);
        node.setAttribute('data-region-type', region.type || 'default');
        node.setAttribute('data-confidence', region.confidence || '0');

        // Restore preserved attributes
        node.className = preservedClasses;
        this.restoreDataAttributes(node, preservedDataAttrs);

        // Trigger animation if specified
        if (region.animate) {
            node.setAttribute('data-animate', 'true');
            node.classList.add('animate-update');
            setTimeout(() => node.classList.remove('animate-update'), 300);
        }
    }

    createPdetNode(container, region, index) {
        const node = document.createElement('div');
        node.className = 'pdet-hexagon';
        node.setAttribute('data-pdet-id', region.id);
        node.setAttribute('data-region-type', region.type || 'default');
        node.setAttribute('data-confidence', region.confidence || '0');
        node.setAttribute('data-index', index);

        const hexContent = document.createElement('div');
        hexContent.className = 'pdet-content';
        hexContent.innerHTML = this.sanitizeHTML(region.content || '');

        node.appendChild(hexContent);

        if (region.coordinates) {
            node.style.transform = `translate(${region.coordinates.x}px, ${region.coordinates.y}px)`;
        }

        container.appendChild(node);

        // Trigger entrance animation
        requestAnimationFrame(() => {
            node.classList.add('animate-enter');
        });
    }

    async updateConstellationNetwork(constellationData) {
        const container = document.querySelector(this.selectors.constellationContainer);
        if (!container) return;

        // Update nodes
        if (constellationData.nodes) {
            this.updateConstellationNodes(container, constellationData.nodes);
        }

        // Update connections
        if (constellationData.connections) {
            this.updateConstellationConnections(container, constellationData.connections);
        }

        this.emit('constellationUpdated', constellationData);
    }

    updateConstellationNodes(container, nodes) {
        nodes.forEach(nodeData => {
            const existingNode = container.querySelector(`[data-node-id="${nodeData.id}"]`);
            
            if (existingNode) {
                this.updateConstellationNode(existingNode, nodeData);
            } else {
                this.createConstellationNode(container, nodeData);
            }
        });
    }

    updateConstellationNode(node, nodeData) {
        const preservedClasses = node.className;
        const preservedDataAttrs = this.getDataAttributes(node);

        // Update node content
        const labelElement = node.querySelector('.node-label') || node;
        labelElement.textContent = nodeData.label || '';

        // Update position
        if (nodeData.position) {
            node.style.left = `${nodeData.position.x}px`;
            node.style.top = `${nodeData.position.y}px`;
        }

        // Update data attributes
        node.setAttribute('data-node-id', nodeData.id);
        node.setAttribute('data-weight', nodeData.weight || '1');
        node.setAttribute('data-category', nodeData.category || 'default');

        // Preserve existing attributes
        node.className = preservedClasses;
        this.restoreDataAttributes(node, preservedDataAttrs);

        if (nodeData.highlight) {
            node.classList.add('highlighted');
            setTimeout(() => node.classList.remove('highlighted'), 2000);
        }
    }

    createConstellationNode(container, nodeData) {
        const node = document.createElement('div');
        node.className = 'constellation-node';
        node.setAttribute('data-node-id', nodeData.id);
        node.setAttribute('data-weight', nodeData.weight || '1');
        node.setAttribute('data-category', nodeData.category || 'default');

        const label = document.createElement('span');
        label.className = 'node-label';
        label.textContent = nodeData.label || '';
        node.appendChild(label);

        if (nodeData.position) {
            node.style.left = `${nodeData.position.x}px`;
            node.style.top = `${nodeData.position.y}px`;
        }

        container.appendChild(node);

        requestAnimationFrame(() => {
            node.classList.add('animate-enter');
        });
    }

    updateConstellationConnections(container, connections) {
        // Remove existing connections
        container.querySelectorAll('.constellation-connection').forEach(conn => {
            if (!connections.find(c => c.id === conn.getAttribute('data-connection-id'))) {
                conn.remove();
            }
        });

        connections.forEach(connection => {
            const existingConn = container.querySelector(`[data-connection-id="${connection.id}"]`);
            
            if (!existingConn) {
                this.createConnection(container, connection);
            } else {
                this.updateConnection(existingConn, connection);
            }
        });
    }

    createConnection(container, connection) {
        const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        line.setAttribute('data-connection-id', connection.id);
        line.setAttribute('data-source', connection.source);
        line.setAttribute('data-target', connection.target);
        line.setAttribute('data-strength', connection.strength || '1');

        if (connection.coordinates) {
            line.setAttribute('x1', connection.coordinates.x1);
            line.setAttribute('y1', connection.coordinates.y1);
            line.setAttribute('x2', connection.coordinates.x2);
            line.setAttribute('y2', connection.coordinates.y2);
        }

        line.classList.add('constellation-connection');
        
        let svg = container.querySelector('svg.connections-layer');
        if (!svg) {
            svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
            svg.classList.add('connections-layer');
            container.appendChild(svg);
        }
        
        svg.appendChild(line);
    }

    async updateEvidenceStream(evidenceItems) {
        const stream = document.querySelector(this.selectors.evidenceStream);
        if (!stream) return;

        // Preserve scroll position
        const scrollTop = stream.scrollTop;

        evidenceItems.forEach((evidence, index) => {
            const existingItem = stream.querySelector(`[data-evidence-id="${evidence.id}"]`);
            
            if (existingItem) {
                this.updateEvidenceItem(existingItem, evidence);
            } else {
                this.createEvidenceItem(stream, evidence, index);
            }
        });

        // Restore scroll position
        stream.scrollTop = scrollTop;

        this.emit('evidenceUpdated', evidenceItems);
    }

    updateEvidenceItem(item, evidence) {
        const preservedClasses = item.className;
        const preservedDataAttrs = this.getDataAttributes(item);

        // Update content elements
        const titleElement = item.querySelector('.evidence-title');
        if (titleElement) {
            titleElement.textContent = evidence.title || '';
        }

        const contentElement = item.querySelector('.evidence-content');
        if (contentElement) {
            contentElement.innerHTML = this.sanitizeHTML(evidence.content || '');
        }

        const pageRefElement = item.querySelector('.page-reference');
        if (pageRefElement && evidence.pageReference) {
            pageRefElement.textContent = `Page ${evidence.pageReference}`;
        }

        const timestampElement = item.querySelector('.timestamp');
        if (timestampElement && evidence.timestamp) {
            timestampElement.textContent = new Date(evidence.timestamp).toLocaleString();
        }

        // Update data attributes
        item.setAttribute('data-evidence-id', evidence.id);
        item.setAttribute('data-confidence', evidence.confidence || '0');
        item.setAttribute('data-source', evidence.source || '');

        // Preserve existing attributes
        item.className = preservedClasses;
        this.restoreDataAttributes(item, preservedDataAttrs);

        if (evidence.isNew) {
            item.classList.add('new-evidence');
            setTimeout(() => item.classList.remove('new-evidence'), 3000);
        }
    }

    createEvidenceItem(stream, evidence, index) {
        const item = document.createElement('div');
        item.className = 'evidence-item';
        item.setAttribute('data-evidence-id', evidence.id);
        item.setAttribute('data-confidence', evidence.confidence || '0');
        item.setAttribute('data-source', evidence.source || '');
        item.setAttribute('data-index', index);

        const title = document.createElement('h4');
        title.className = 'evidence-title';
        title.textContent = evidence.title || '';

        const content = document.createElement('div');
        content.className = 'evidence-content';
        content.innerHTML = this.sanitizeHTML(evidence.content || '');

        const metadata = document.createElement('div');
        metadata.className = 'evidence-metadata';

        if (evidence.pageReference) {
            const pageRef = document.createElement('span');
            pageRef.className = 'page-reference';
            pageRef.textContent = `Page ${evidence.pageReference}`;
            metadata.appendChild(pageRef);
        }

        if (evidence.timestamp) {
            const timestamp = document.createElement('span');
            timestamp.className = 'timestamp';
            timestamp.textContent = new Date(evidence.timestamp).toLocaleString();
            metadata.appendChild(timestamp);
        }

        item.appendChild(title);
        item.appendChild(content);
        item.appendChild(metadata);

        stream.appendChild(item);

        requestAnimationFrame(() => {
            item.classList.add('animate-enter');
        });
    }

    setupEventListeners() {
        // Handle visibility changes to pause/resume updates
        document.addEventListener('visibilitychange', () => {
            if (document.hidden) {
                this.pauseAutoUpdate();
            } else {
                this.resumeAutoUpdate();
            }
        });

        // Handle network status changes
        window.addEventListener('online', () => {
            this.emit('networkStatus', { online: true });
            this.resumeAutoUpdate();
        });

        window.addEventListener('offline', () => {
            this.emit('networkStatus', { online: false });
            this.pauseAutoUpdate();
        });
    }

    startAutoUpdate() {
        this.updateTimer = setInterval(async () => {
            const data = await this.fetchData();
            if (data) {
                await this.updateAllComponents(data);
            }
        }, this.updateInterval);
    }

    pauseAutoUpdate() {
        if (this.updateTimer) {
            clearInterval(this.updateTimer);
            this.updateTimer = null;
        }
    }

    resumeAutoUpdate() {
        if (!this.updateTimer && this.isInitialized) {
            this.startAutoUpdate();
        }
    }

    // Utility methods
    getDataAttributes(element) {
        const attrs = {};
        for (const attr of element.attributes) {
            if (attr.name.startsWith('data-animate') || 
                attr.name.startsWith('data-trigger') ||
                attr.name.startsWith('data-effect')) {
                attrs[attr.name] = attr.value;
            }
        }
        return attrs;
    }

    restoreDataAttributes(element, attrs) {
        for (const [name, value] of Object.entries(attrs)) {
            element.setAttribute(name, value);
        }
    }

    sanitizeHTML(html) {
        const temp = document.createElement('div');
        temp.textContent = html;
        return temp.innerHTML;
    }

    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    emit(eventType, data = null) {
        this.eventBus.dispatchEvent(new CustomEvent(eventType, { detail: data }));
    }

    on(eventType, callback) {
        this.eventBus.addEventListener(eventType, callback);
    }

    off(eventType, callback) {
        this.eventBus.removeEventListener(eventType, callback);
    }

    // Public API methods
    async forceUpdate() {
        const data = await this.fetchData();
        if (data) {
            await this.updateAllComponents(data);
        }
    }

    setUpdateInterval(interval) {
        this.updateInterval = interval;
        if (this.updateTimer) {
            this.pauseAutoUpdate();
            this.startAutoUpdate();
        }
    }

    destroy() {
        this.pauseAutoUpdate();
        this.eventBus = null;
        this.isInitialized = false;
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = AtroZDataBinding;
} else if (typeof define === 'function' && define.amd) {
    define([], () => AtroZDataBinding);
} else {
    window.AtroZDataBinding = AtroZDataBinding;
}