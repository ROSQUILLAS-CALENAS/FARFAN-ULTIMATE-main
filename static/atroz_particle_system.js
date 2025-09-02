/**
 * AtroZ Particle System - Dynamic PDET Region Visualization
 * Supports both static/random data and dynamic backend API data
 */

class AtroZParticleSystem {
    constructor(canvasId, config = {}) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.particles = [];
        this.connections = [];
        
        // Configuration with defaults
        this.config = {
            maxParticles: config.maxParticles || 200,
            evidenceColorMap: config.evidenceColorMap || {
                high: '#ff4444',
                medium: '#ffaa44',
                low: '#44ff44'
            },
            particleSize: config.particleSize || { min: 2, max: 8 },
            connectionDistance: config.connectionDistance || 80,
            animationSpeed: config.animationSpeed || 0.02,
            ...config
        };
        
        // Data sources
        this.dataSource = 'static'; // 'static' or 'api'
        this.apiEndpoints = {};
        this.lastDataUpdate = 0;
        this.updateInterval = 30000; // 30 seconds default
        
        // Particle animation properties
        this.mouseX = 0;
        this.mouseY = 0;
        this.animationFrame = null;
        
        this.setupCanvas();
        this.setupEventListeners();
        this.initializeParticles();
        this.startAnimation();
    }
    
    setupCanvas() {
        const rect = this.canvas.getBoundingClientRect();
        this.canvas.width = rect.width * window.devicePixelRatio;
        this.canvas.height = rect.height * window.devicePixelRatio;
        this.ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
        
        // Set canvas display size
        this.canvas.style.width = rect.width + 'px';
        this.canvas.style.height = rect.height + 'px';
    }
    
    setupEventListeners() {
        // Mouse interaction
        this.canvas.addEventListener('mousemove', (e) => {
            const rect = this.canvas.getBoundingClientRect();
            this.mouseX = e.clientX - rect.left;
            this.mouseY = e.clientY - rect.top;
        });
        
        // Window resize
        window.addEventListener('resize', () => {
            this.setupCanvas();
        });
    }
    
    // Configure data sources from backend
    configureDataSources(endpoints, updateInterval = 30000) {
        this.apiEndpoints = endpoints;
        this.updateInterval = updateInterval;
        this.dataSource = 'api';
    }
    
    // Initialize particles with static/random data
    initializeParticles() {
        this.particles = [];
        const particleCount = Math.min(this.config.maxParticles, 150);
        
        for (let i = 0; i < particleCount; i++) {
            this.particles.push(this.createStaticParticle());
        }
    }
    
    createStaticParticle() {
        return {
            x: Math.random() * this.canvas.width / window.devicePixelRatio,
            y: Math.random() * this.canvas.height / window.devicePixelRatio,
            vx: (Math.random() - 0.5) * 2,
            vy: (Math.random() - 0.5) * 2,
            size: this.config.particleSize.min + Math.random() * (this.config.particleSize.max - this.config.particleSize.min),
            evidenceScore: Math.random(),
            region: `PDET_${Math.floor(Math.random() * 10) + 1}`,
            color: this.getEvidenceColor(Math.random()),
            opacity: 0.7 + Math.random() * 0.3,
            pulsePhase: Math.random() * Math.PI * 2
        };
    }
    
    // Load dynamic data from backend API
    async loadDynamicData() {
        if (this.dataSource !== 'api' || !this.apiEndpoints.pdet_regions) {
            return;
        }
        
        try {
            document.getElementById('data-source').textContent = 'Loading...';
            document.getElementById('data-source').classList.add('loading');
            
            const [regionsResponse, evidenceResponse] = await Promise.all([
                fetch(this.apiEndpoints.pdet_regions),
                fetch(this.apiEndpoints.evidence_scores)
            ]);
            
            if (!regionsResponse.ok || !evidenceResponse.ok) {
                throw new Error('Failed to fetch data');
            }
            
            const regionsData = await regionsResponse.json();
            const evidenceData = await evidenceResponse.json();
            
            this.updateParticlesFromData(regionsData, evidenceData);
            this.lastDataUpdate = Date.now();
            
            document.getElementById('data-source').textContent = 'Live Data';
            document.getElementById('data-source').classList.remove('loading');
            
        } catch (error) {
            console.warn('Failed to load dynamic data, falling back to static:', error);
            this.dataSource = 'static';
            document.getElementById('data-source').textContent = 'Static Data (API Error)';
            document.getElementById('data-source').classList.remove('loading');
        }
    }
    
    // Update particles with real backend data
    updateParticlesFromData(regionsData, evidenceData) {
        this.particles = [];
        
        // Process PDET regions data
        if (regionsData && regionsData.regions) {
            regionsData.regions.forEach((region, index) => {
                // Create particles based on region coordinates
                const evidenceScore = evidenceData?.scores?.[region.id] || Math.random();
                
                this.particles.push({
                    x: this.normalizeCoordinate(region.coordinates.x, regionsData.bounds.x),
                    y: this.normalizeCoordinate(region.coordinates.y, regionsData.bounds.y),
                    vx: (region.velocity?.x || (Math.random() - 0.5)) * 2,
                    vy: (region.velocity?.y || (Math.random() - 0.5)) * 2,
                    size: this.mapEvidenceToSize(evidenceScore),
                    evidenceScore: evidenceScore,
                    region: region.id,
                    color: this.getEvidenceColor(evidenceScore),
                    opacity: 0.6 + evidenceScore * 0.4,
                    pulsePhase: (index * Math.PI / 4) % (Math.PI * 2),
                    metadata: region.metadata || {}
                });
            });
        }
        
        // Ensure minimum particle count for visual appeal
        while (this.particles.length < 50) {
            this.particles.push(this.createStaticParticle());
        }
        
        console.log(`Updated ${this.particles.length} particles from API data`);
    }
    
    // Normalize API coordinates to canvas coordinates
    normalizeCoordinate(value, bounds) {
        const canvasWidth = this.canvas.width / window.devicePixelRatio;
        const canvasHeight = this.canvas.height / window.devicePixelRatio;
        
        if (bounds && bounds.min !== undefined && bounds.max !== undefined) {
            const normalized = (value - bounds.min) / (bounds.max - bounds.min);
            return normalized * (bounds.axis === 'x' ? canvasWidth : canvasHeight);
        }
        
        // Fallback to direct mapping
        return Math.max(0, Math.min(value, bounds.axis === 'x' ? canvasWidth : canvasHeight));
    }
    
    // Map evidence score to particle size
    mapEvidenceToSize(evidenceScore) {
        const minSize = this.config.particleSize.min;
        const maxSize = this.config.particleSize.max;
        return minSize + (evidenceScore * (maxSize - minSize));
    }
    
    // Get color based on evidence score
    getEvidenceColor(evidenceScore) {
        if (evidenceScore >= 0.7) return this.config.evidenceColorMap.high;
        if (evidenceScore >= 0.4) return this.config.evidenceColorMap.medium;
        return this.config.evidenceColorMap.low;
    }
    
    // Update particle positions and behaviors
    updateParticles() {
        const canvasWidth = this.canvas.width / window.devicePixelRatio;
        const canvasHeight = this.canvas.height / window.devicePixelRatio;
        
        this.particles.forEach(particle => {
            // Update position
            particle.x += particle.vx * this.config.animationSpeed;
            particle.y += particle.vy * this.config.animationSpeed;
            
            // Boundary collision
            if (particle.x <= 0 || particle.x >= canvasWidth) {
                particle.vx *= -1;
                particle.x = Math.max(0, Math.min(canvasWidth, particle.x));
            }
            if (particle.y <= 0 || particle.y >= canvasHeight) {
                particle.vy *= -1;
                particle.y = Math.max(0, Math.min(canvasHeight, particle.y));
            }
            
            // Mouse interaction
            const mouseDistance = Math.sqrt(
                Math.pow(this.mouseX - particle.x, 2) + 
                Math.pow(this.mouseY - particle.y, 2)
            );
            
            if (mouseDistance < 100) {
                const force = (100 - mouseDistance) / 100;
                const angle = Math.atan2(particle.y - this.mouseY, particle.x - this.mouseX);
                particle.vx += Math.cos(angle) * force * 0.5;
                particle.vy += Math.sin(angle) * force * 0.5;
            }
            
            // Update pulse animation
            particle.pulsePhase += 0.1;
        });
    }
    
    // Calculate particle connections based on distance and evidence scores
    calculateConnections() {
        this.connections = [];
        
        for (let i = 0; i < this.particles.length; i++) {
            for (let j = i + 1; j < this.particles.length; j++) {
                const particle1 = this.particles[i];
                const particle2 = this.particles[j];
                
                const distance = Math.sqrt(
                    Math.pow(particle1.x - particle2.x, 2) + 
                    Math.pow(particle1.y - particle2.y, 2)
                );
                
                if (distance < this.config.connectionDistance) {
                    const strength = 1 - (distance / this.config.connectionDistance);
                    const evidenceWeight = (particle1.evidenceScore + particle2.evidenceScore) / 2;
                    
                    this.connections.push({
                        particle1,
                        particle2,
                        strength,
                        evidenceWeight,
                        distance
                    });
                }
            }
        }
    }
    
    // Render the particle system
    render() {
        // Clear canvas with fade effect
        this.ctx.fillStyle = 'rgba(26, 26, 46, 0.1)';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Draw connections
        this.connections.forEach(connection => {
            const opacity = connection.strength * connection.evidenceWeight * 0.3;
            
            this.ctx.strokeStyle = `rgba(0, 212, 255, ${opacity})`;
            this.ctx.lineWidth = connection.evidenceWeight * 2;
            this.ctx.beginPath();
            this.ctx.moveTo(connection.particle1.x, connection.particle1.y);
            this.ctx.lineTo(connection.particle2.x, connection.particle2.y);
            this.ctx.stroke();
        });
        
        // Draw particles
        this.particles.forEach(particle => {
            const pulseSize = particle.size + Math.sin(particle.pulsePhase) * 2;
            const pulseOpacity = particle.opacity + Math.sin(particle.pulsePhase) * 0.2;
            
            // Particle glow effect
            const gradient = this.ctx.createRadialGradient(
                particle.x, particle.y, 0,
                particle.x, particle.y, pulseSize * 2
            );
            gradient.addColorStop(0, particle.color);
            gradient.addColorStop(1, 'rgba(0, 0, 0, 0)');
            
            this.ctx.globalAlpha = pulseOpacity * 0.5;
            this.ctx.fillStyle = gradient;
            this.ctx.beginPath();
            this.ctx.arc(particle.x, particle.y, pulseSize * 2, 0, Math.PI * 2);
            this.ctx.fill();
            
            // Main particle
            this.ctx.globalAlpha = pulseOpacity;
            this.ctx.fillStyle = particle.color;
            this.ctx.beginPath();
            this.ctx.arc(particle.x, particle.y, pulseSize, 0, Math.PI * 2);
            this.ctx.fill();
        });
        
        this.ctx.globalAlpha = 1;
    }
    
    // Animation loop
    animate() {
        this.updateParticles();
        this.calculateConnections();
        this.render();
        
        // Check if data update is needed
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
    
    // Public methods for external control
    refreshData() {
        if (this.dataSource === 'api') {
            this.loadDynamicData();
        } else {
            this.initializeParticles();
        }
    }
    
    setEvidenceThreshold(threshold) {
        this.particles.forEach(particle => {
            particle.opacity = particle.evidenceScore >= threshold ? 0.9 : 0.3;
        });
    }
    
    getActiveParticles() {
        return this.particles.filter(p => p.evidenceScore >= 0.5);
    }
    
    destroy() {
        this.stopAnimation();
        this.canvas.removeEventListener('mousemove', this.mouseMoveHandler);
        window.removeEventListener('resize', this.resizeHandler);
    }
}