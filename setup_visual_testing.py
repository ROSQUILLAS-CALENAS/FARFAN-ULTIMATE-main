"""
Setup script for Visual Testing Framework
"""

import subprocess
import sys
from pathlib import Path


def install_dependencies():
    """Install required dependencies for visual testing"""
    print("📦 Installing visual testing dependencies...")
    
    try:
        # Install Python dependencies
        subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "-r", "requirements_visual_testing.txt"
        ], check=True)
        
        print("✅ Python dependencies installed")
        
        # Install Playwright browsers
        print("🎭 Installing Playwright browsers...")
        subprocess.run([
            sys.executable, "-m", "playwright", "install", "chromium"
        ], check=True)
        
        print("✅ Playwright browsers installed")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Installation failed: {e}")
        return False
    
    return True


def create_demo_html():
    """Create demo HTML file for testing"""
    demo_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AtroZ Dashboard Demo</title>
    <style>
        body {
            margin: 0;
            background: #000;
            color: #00ff00;
            font-family: 'Courier New', monospace;
            overflow: hidden;
        }
        
        .dashboard {
            width: 100vw;
            height: 100vh;
            position: relative;
        }
        
        /* Organic Pulse Animation */
        .organic-pulse {
            position: absolute;
            top: 20%;
            left: 20%;
            width: 200px;
            height: 200px;
            background: radial-gradient(circle, #00ff00 0%, #004400 100%);
            border-radius: 50%;
            animation: organicPulse 2s infinite ease-in-out;
        }
        
        @keyframes organicPulse {
            0% { transform: scale(1); opacity: 1; }
            50% { transform: scale(1.1); opacity: 0.8; }
            100% { transform: scale(1); opacity: 1; }
        }
        
        /* Neural Flow Animation */
        .neural-flow {
            position: absolute;
            top: 20%;
            right: 20%;
            width: 300px;
            height: 100px;
            background: linear-gradient(45deg, #ff0080, #0080ff, #80ff00);
            background-size: 400% 400%;
            animation: neuralFlow 3s infinite linear;
            border-radius: 10px;
        }
        
        @keyframes neuralFlow {
            0% { background-position: 0% 0%; filter: hue-rotate(0deg); }
            25% { background-position: 25% 25%; filter: hue-rotate(90deg); }
            50% { background-position: 50% 50%; filter: hue-rotate(180deg); }
            75% { background-position: 75% 75%; filter: hue-rotate(270deg); }
            100% { background-position: 100% 100%; filter: hue-rotate(360deg); }
        }
        
        /* Glitch Effect */
        .glitch-effect {
            position: absolute;
            bottom: 20%;
            left: 50%;
            transform: translateX(-50%);
            color: #ff0000;
            font-size: 48px;
            font-weight: bold;
            animation: glitchEffect 0.5s infinite;
        }
        
        @keyframes glitchEffect {
            0% { transform: translateX(-50%) translateY(0); filter: hue-rotate(0deg); }
            10% { transform: translateX(calc(-50% - 2px)) translateY(0); filter: hue-rotate(90deg); }
            20% { transform: translateX(calc(-50% + 2px)) translateY(0); filter: hue-rotate(180deg); }
            30% { transform: translateX(calc(-50% - 1px)) translateY(0); filter: hue-rotate(270deg); }
            100% { transform: translateX(-50%) translateY(0); filter: hue-rotate(0deg); }
        }
        
        /* Interactive Elements */
        .dashboard-button {
            position: absolute;
            top: 10px;
            left: 10px;
            padding: 10px 20px;
            background: #006600;
            color: #00ff00;
            border: 2px solid #00ff00;
            border-radius: 5px;
            cursor: pointer;
            transition: all 0.2s ease;
        }
        
        .dashboard-button:hover {
            background: #00ff00;
            color: #006600;
            transform: scale(1.05);
        }
        
        .menu-item {
            position: absolute;
            top: 10px;
            left: 150px;
            padding: 8px 15px;
            background: #333;
            color: #fff;
            border: 1px solid #666;
            cursor: pointer;
            transition: background 0.15s ease;
        }
        
        .menu-item:hover {
            background: #555;
        }
        
        .control-panel {
            position: absolute;
            top: 10px;
            right: 10px;
            width: 200px;
            height: 50px;
            background: #222;
            border: 2px solid #444;
            border-radius: 8px;
            cursor: pointer;
            transition: border-color 0.2s ease;
        }
        
        .control-panel:hover {
            border-color: #00ff00;
        }
        
        .data-widget {
            position: absolute;
            bottom: 10px;
            right: 10px;
            width: 150px;
            height: 100px;
            background: linear-gradient(135deg, #2a2a2a, #1a1a1a);
            border: 1px solid #555;
            border-radius: 6px;
            cursor: pointer;
            transition: transform 0.2s ease;
        }
        
        .data-widget:hover {
            transform: translateY(-2px);
        }
        
        /* Particle System Canvas */
        .particle-system {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            z-index: -1;
        }
    </style>
</head>
<body>
    <div class="dashboard">
        <canvas class="particle-system" id="particleCanvas"></canvas>
        
        <div class="organic-pulse"></div>
        <div class="neural-flow"></div>
        <div class="glitch-effect">ATROZ SYSTEM</div>
        
        <button class="dashboard-button">Dashboard</button>
        <div class="menu-item">Menu Item</div>
        <div class="control-panel"></div>
        <div class="data-widget"></div>
    </div>
    
    <script>
        // Simple particle system
        const canvas = document.getElementById('particleCanvas');
        const ctx = canvas.getContext('2d');
        
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
        
        const particles = [];
        const numParticles = 50;
        
        class Particle {
            constructor() {
                this.x = Math.random() * canvas.width;
                this.y = Math.random() * canvas.height;
                this.vx = (Math.random() - 0.5) * 2;
                this.vy = (Math.random() - 0.5) * 2;
                this.size = Math.random() * 3 + 1;
                this.opacity = Math.random() * 0.8 + 0.2;
            }
            
            update() {
                this.x += this.vx;
                this.y += this.vy;
                
                if (this.x < 0 || this.x > canvas.width) this.vx *= -1;
                if (this.y < 0 || this.y > canvas.height) this.vy *= -1;
            }
            
            draw() {
                ctx.beginPath();
                ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
                ctx.fillStyle = `rgba(0, 255, 0, ${this.opacity})`;
                ctx.fill();
            }
        }
        
        // Initialize particles
        for (let i = 0; i < numParticles; i++) {
            particles.push(new Particle());
        }
        
        function animate() {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            particles.forEach(particle => {
                particle.update();
                particle.draw();
            });
            
            requestAnimationFrame(animate);
        }
        
        animate();
        
        // Interactive element handlers
        document.querySelectorAll('.dashboard-button, .menu-item, .control-panel, .data-widget').forEach(el => {
            el.addEventListener('click', function() {
                this.style.transform = 'scale(0.95)';
                setTimeout(() => {
                    this.style.transform = '';
                }, 100);
            });
        });
        
        // Resize canvas on window resize
        window.addEventListener('resize', () => {
            canvas.width = window.innerWidth;
            canvas.height = window.innerHeight;
        });
    </script>
</body>
</html>
    """
    
    demo_path = Path("demo_dashboard.html")
    with open(demo_path, 'w', encoding='utf-8') as f:
        f.write(demo_html)
    
    print(f"✅ Demo dashboard created: {demo_path}")
    return str(demo_path)


def create_test_runner():
    """Create a test runner script"""
    runner_script = """
#!/usr/bin/env python3
\"\"\"
Visual Testing Framework Runner
\"\"\"

import asyncio
import sys
from pathlib import Path
from visual_testing_framework import run_visual_tests


async def main():
    # Default to demo dashboard
    demo_path = Path("demo_dashboard.html").absolute()
    dashboard_url = f"file://{demo_path}"
    
    # Use provided URL if given
    if len(sys.argv) > 1:
        dashboard_url = sys.argv[1]
    
    print(f"🎯 Running visual tests on: {dashboard_url}")
    
    # Run tests
    results = await run_visual_tests(dashboard_url)
    
    if results:
        if results['summary']['failed'] == 0:
            print("\\n🎉 All tests passed!")
            sys.exit(0)
        else:
            print("\\n💥 Some tests failed!")
            sys.exit(1)
    else:
        print("\\n❌ Test execution failed!")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
    """
    
    runner_path = Path("run_visual_tests.py")
    with open(runner_path, 'w') as f:
        f.write(runner_script)
    
    runner_path.chmod(0o755)  # Make executable
    print(f"✅ Test runner created: {runner_path}")


def main():
    """Main setup function"""
    print("🎭 Setting up AtroZ Dashboard Visual Testing Framework")
    print("=" * 60)
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Setup failed during dependency installation")
        sys.exit(1)
    
    # Create demo files
    demo_path = create_demo_html()
    create_test_runner()
    
    # Create directory structure
    directories = [
        "visual_tests/baselines",
        "visual_tests/outputs", 
        "visual_tests/reports"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✅ Directory structure created")
    
    print("\n🎉 Setup complete!")
    print("\nNext steps:")
    print("1. Run tests on demo: python run_visual_tests.py")
    print("2. Run tests on your dashboard: python run_visual_tests.py http://your-dashboard-url")
    print("3. Generate baselines: python visual_testing_framework.py --create-baselines")
    print("4. View results in: visual_tests/reports/")
    
    print(f"\nDemo dashboard available at: file://{Path(demo_path).absolute()}")


if __name__ == "__main__":
    main()