# AtroZ Dashboard Visual Testing Framework

A comprehensive automated testing framework that validates CSS animations, particle systems, and interactive elements using browser automation to ensure visual consistency and performance.

## 🎯 Features

### Animation Testing
- **CSS Keyframe Validation**: Monitors `organicPulse`, `neuralFlow`, and `glitchEffect` animations
- **Timeline Accuracy**: Verifies animation timing and smoothness
- **Property Tracking**: Validates CSS transform, opacity, filter changes

### Visual Regression Testing  
- **Screenshot Comparison**: Baseline vs current implementation
- **Pixel-level Diff**: Quantifies visual changes with percentage scores
- **Automated Baselines**: Captures reference images for future comparisons

### Performance Monitoring
- **Frame Rate Analysis**: Ensures animations maintain 30+ FPS
- **Memory Usage**: Tracks JavaScript heap consumption
- **Response Time**: Measures interactive element responsiveness (<100ms)

### Browser Automation
- **Playwright Support**: Modern browser automation with async capabilities  
- **Selenium Fallback**: Compatible with traditional WebDriver setup
- **Cross-browser Testing**: Chromium, Firefox, Safari support

## 📦 Installation

```bash
# Install dependencies
pip install -r requirements_visual_testing.txt

# Setup framework (installs browsers and creates demo)
python setup_visual_testing.py
```

### Dependencies
```
playwright>=1.40.0
selenium>=4.15.0  
Pillow>=10.0.0
numpy>=1.24.0
```

## 🚀 Quick Start

### 1. Run Tests on Demo Dashboard
```bash
python run_visual_tests.py
```

### 2. Test Your Dashboard
```bash
python run_visual_tests.py http://localhost:3000
```

### 3. Generate Baseline Images
```bash
python visual_testing_framework.py --create-baselines http://your-dashboard
```

## 🧪 Test Cases

### Animation Tests
The framework validates these specific AtroZ dashboard animations:

#### organicPulse
```css
@keyframes organicPulse {
    0% { transform: scale(1); opacity: 1; }
    50% { transform: scale(1.1); opacity: 0.8; }
    100% { transform: scale(1); opacity: 1; }
}
```
- **Duration**: 2.0s
- **Selector**: `.organic-pulse`
- **Validated Properties**: `transform`, `opacity`

#### neuralFlow  
```css
@keyframes neuralFlow {
    0% { background-position: 0% 0%; filter: hue-rotate(0deg); }
    25% { background-position: 25% 25%; filter: hue-rotate(90deg); }
    50% { background-position: 50% 50%; filter: hue-rotate(180deg); }
    75% { background-position: 75% 75%; filter: hue-rotate(270deg); }
    100% { background-position: 100% 100%; filter: hue-rotate(360deg); }
}
```
- **Duration**: 3.0s  
- **Selector**: `.neural-flow`
- **Validated Properties**: `background-position`, `filter`

#### glitchEffect
```css
@keyframes glitchEffect {
    0% { transform: translateX(0); filter: hue-rotate(0deg); }
    10% { transform: translateX(-2px); filter: hue-rotate(90deg); }
    20% { transform: translateX(2px); filter: hue-rotate(180deg); }
    30% { transform: translateX(-1px); filter: hue-rotate(270deg); }
    100% { transform: translateX(0); filter: hue-rotate(0deg); }
}
```
- **Duration**: 0.5s
- **Selector**: `.glitch-effect`  
- **Validated Properties**: `transform`, `filter`

### Interactive Elements
- **Dashboard Button**: `.dashboard-button`
- **Menu Items**: `.menu-item`
- **Control Panel**: `.control-panel` 
- **Data Widgets**: `.data-widget`

### Particle System
- **Canvas Element**: `canvas.particle-system`
- **Update Detection**: Monitors pixel changes frame-to-frame
- **Performance**: Validates smooth 30+ FPS rendering

## 📊 Test Reports

### Report Structure
```json
{
  "summary": {
    "total_tests": 12,
    "passed": 10,
    "failed": 2,
    "success_rate": 83.3,
    "total_execution_time": 45.2,
    "timestamp": "2024-01-15T10:30:00"
  },
  "animation_tests": {
    "organicPulse": {
      "status": "passed",
      "keyframe_validation": true,
      "visual_snapshot": true,
      "execution_time": 2.1
    }
  },
  "visual_regression": {
    "snapshots_captured": 8,
    "visual_differences": [
      {
        "test": "neuralFlow_animation", 
        "diff_score": 2.3,
        "screenshot": "/path/to/screenshot.png"
      }
    ]
  },
  "performance_metrics": {
    "avgFPS": 45.2,
    "minFPS": 32.1,
    "frameRateStability": 85.4,
    "memoryUsage": [{"used": 42.5, "total": 128.0}]
  }
}
```

### Report Locations
- **JSON Reports**: `visual_tests/reports/test_report_YYYYMMDD_HHMMSS.json`
- **Screenshots**: `visual_tests/outputs/`
- **Baselines**: `visual_tests/baselines/`

## 🎛️ Configuration

### Animation Configuration
```python
AnimationTestConfig(
    name="customAnimation",
    selector=".custom-animation",
    animation_duration=1.5,
    keyframes=["0%", "50%", "100%"],
    expected_properties={
        "transform": ["scale(1)", "scale(1.2)", "scale(1)"],
        "opacity": [1.0, 0.5, 1.0]
    },
    performance_thresholds={
        "fps": 30.0,
        "memory_usage": 50.0,
        "cpu_usage": 70.0
    }
)
```

### Framework Options
```python
framework = VisualTestingFramework(
    use_playwright=True,  # Use Playwright vs Selenium
    baseline_dir="custom/baselines",
    output_dir="custom/outputs",
    report_dir="custom/reports"
)
```

## 🔧 Advanced Usage

### Custom Test Suite
```python
import asyncio
from visual_testing_framework import VisualTestingFramework

async def custom_tests():
    framework = VisualTestingFramework()
    await framework.initialize_browser()
    await framework.load_dashboard("http://localhost:3000")
    
    # Test specific animation
    result = await framework.test_animation_keyframes(
        framework.animations["organicPulse"]
    )
    
    # Capture comparison snapshots
    baseline = await framework.capture_visual_snapshot("test", is_baseline=True)
    current = await framework.capture_visual_snapshot("test", is_baseline=False)
    
    # Compare visually
    diff_score = framework.compare_visual_snapshots(baseline, current)
    
    await framework.cleanup_browser()
    return result

asyncio.run(custom_tests())
```

### CI/CD Integration
```yaml
# .github/workflows/visual-tests.yml
name: Visual Regression Tests
on: [push, pull_request]

jobs:
  visual-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup Python
        uses: actions/setup-python@v3
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: |
          pip install -r requirements_visual_testing.txt
          playwright install chromium
      - name: Run visual tests
        run: python run_visual_tests.py http://localhost:3000
      - name: Upload test results
        uses: actions/upload-artifact@v3
        with:
          name: visual-test-results
          path: visual_tests/reports/
```

## 🐛 Troubleshooting

### Common Issues

#### Browser Not Found
```bash
# Install Playwright browsers
python -m playwright install chromium

# Or download ChromeDriver for Selenium
# https://chromedriver.chromium.org/
```

#### Permission Errors
```bash
chmod +x run_visual_tests.py
chmod +x setup_visual_testing.py
```

#### Memory Issues
```python
# Reduce test scope
framework = VisualTestingFramework()
# Test specific animations only
result = await framework.test_animation_keyframes(config)
```

### Debugging
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable verbose output
framework.logger.setLevel(logging.DEBUG)
```

## 📈 Performance Benchmarks

### Expected Metrics
- **Animation FPS**: 30+ (60+ for glitch effects)
- **Response Time**: <100ms for interactive elements
- **Memory Usage**: <50MB for standard animations
- **Visual Diff**: <5% for minor updates, <1% for patches

### Optimization Tips
1. **Use Playwright**: ~40% faster than Selenium
2. **Headless Mode**: Reduces resource usage by ~30%
3. **Selective Testing**: Target specific animations vs full suite
4. **Parallel Execution**: Run tests across multiple browser instances

## 🤝 Contributing

1. Fork the repository
2. Add new animation configurations in `visual_testing_framework.py`
3. Update test cases in `test_visual_framework.py`
4. Submit pull request with test results

## 📄 License

This framework is part of the AtroZ Dashboard testing suite and follows the project's licensing terms.