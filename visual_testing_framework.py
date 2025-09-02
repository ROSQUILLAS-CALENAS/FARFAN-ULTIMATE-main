"""
Visual Animation Testing Framework for AtroZ Dashboard
=====================================================

Comprehensive automated testing framework that validates CSS animations,
particle systems, and interactive elements using browser automation.
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import logging

# Browser automation imports
try:
    from playwright.async_api import async_playwright, Page, Browser
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    
try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.options import Options
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

# Image comparison
try:
    from PIL import Image, ImageChops
    import numpy as np
    IMAGE_COMPARISON_AVAILABLE = True
except ImportError:
    IMAGE_COMPARISON_AVAILABLE = False


@dataclass
class AnimationTestConfig:
    """Configuration for animation testing"""
    name: str
    selector: str
    animation_duration: float
    keyframes: List[str]
    expected_properties: Dict[str, Any]
    performance_thresholds: Dict[str, float]


@dataclass
class TestResult:
    """Test execution result"""
    test_name: str
    passed: bool
    execution_time: float
    screenshot_path: Optional[str] = None
    error_message: Optional[str] = None
    metrics: Dict[str, float] = None
    visual_diff_score: Optional[float] = None


class VisualTestingFramework:
    """
    Comprehensive visual testing framework for AtroZ dashboard animations
    """
    
    def __init__(self, use_playwright: bool = True):
        self.use_playwright = use_playwright and PLAYWRIGHT_AVAILABLE
        self.browser = None
        self.page = None
        self.driver = None
        self.test_results: List[TestResult] = []
        self.baseline_dir = Path("visual_tests/baselines")
        self.output_dir = Path("visual_tests/outputs")
        self.report_dir = Path("visual_tests/reports")
        
        # Create directories
        for directory in [self.baseline_dir, self.output_dir, self.report_dir]:
            directory.mkdir(parents=True, exist_ok=True)
            
        # Animation configurations
        self.animations = {
            "organicPulse": AnimationTestConfig(
                name="organicPulse",
                selector=".organic-pulse",
                animation_duration=2.0,
                keyframes=["0%", "50%", "100%"],
                expected_properties={
                    "transform": ["scale(1)", "scale(1.1)", "scale(1)"],
                    "opacity": [1.0, 0.8, 1.0]
                },
                performance_thresholds={
                    "fps": 30.0,
                    "memory_usage": 50.0,  # MB
                    "cpu_usage": 70.0      # %
                }
            ),
            "neuralFlow": AnimationTestConfig(
                name="neuralFlow",
                selector=".neural-flow",
                animation_duration=3.0,
                keyframes=["0%", "25%", "50%", "75%", "100%"],
                expected_properties={
                    "background-position": ["0% 0%", "25% 25%", "50% 50%", "75% 75%", "100% 100%"],
                    "filter": ["hue-rotate(0deg)", "hue-rotate(90deg)", "hue-rotate(180deg)", "hue-rotate(270deg)", "hue-rotate(360deg)"]
                },
                performance_thresholds={
                    "fps": 30.0,
                    "memory_usage": 75.0,
                    "cpu_usage": 80.0
                }
            ),
            "glitchEffect": AnimationTestConfig(
                name="glitchEffect",
                selector=".glitch-effect",
                animation_duration=0.5,
                keyframes=["0%", "10%", "20%", "30%", "100%"],
                expected_properties={
                    "transform": ["translateX(0)", "translateX(-2px)", "translateX(2px)", "translateX(-1px)", "translateX(0)"],
                    "filter": ["hue-rotate(0deg)", "hue-rotate(90deg)", "hue-rotate(180deg)", "hue-rotate(270deg)", "hue-rotate(0deg)"]
                },
                performance_thresholds={
                    "fps": 60.0,
                    "memory_usage": 30.0,
                    "cpu_usage": 50.0
                }
            )
        }
        
        self.logger = logging.getLogger(__name__)

    async def initialize_browser(self):
        """Initialize browser for testing"""
        if self.use_playwright:
            await self._initialize_playwright()
        else:
            self._initialize_selenium()

    async def _initialize_playwright(self):
        """Initialize Playwright browser"""
        if not PLAYWRIGHT_AVAILABLE:
            raise RuntimeError("Playwright not available")
            
        playwright = await async_playwright().start()
        self.browser = await playwright.chromium.launch(headless=True)
        self.page = await self.browser.new_page()
        await self.page.set_viewport_size({"width": 1920, "height": 1080})

    def _initialize_selenium(self):
        """Initialize Selenium WebDriver"""
        if not SELENIUM_AVAILABLE:
            raise RuntimeError("Selenium not available")
            
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--window-size=1920,1080")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        self.driver = webdriver.Chrome(options=options)

    async def load_dashboard(self, url: str):
        """Load the AtroZ dashboard"""
        if self.use_playwright:
            await self.page.goto(url)
            await self.page.wait_for_load_state("networkidle")
        else:
            self.driver.get(url)
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )

    async def test_animation_keyframes(self, animation_config: AnimationTestConfig) -> TestResult:
        """Test CSS keyframe execution for specific animation"""
        start_time = time.time()
        
        try:
            if self.use_playwright:
                element = await self.page.wait_for_selector(animation_config.selector)
                
                # Monitor animation states at different keyframes
                keyframe_results = []
                duration = animation_config.animation_duration
                
                for i, keyframe in enumerate(animation_config.keyframes):
                    progress = i / (len(animation_config.keyframes) - 1)
                    wait_time = duration * progress
                    
                    await asyncio.sleep(wait_time)
                    
                    # Capture computed styles
                    styles = await self.page.evaluate(f"""
                        const element = document.querySelector('{animation_config.selector}');
                        const computed = window.getComputedStyle(element);
                        return {{
                            transform: computed.transform,
                            opacity: computed.opacity,
                            filter: computed.filter,
                            backgroundPosition: computed.backgroundPosition
                        }};
                    """)
                    
                    keyframe_results.append({
                        "keyframe": keyframe,
                        "styles": styles,
                        "timestamp": wait_time
                    })
                
                # Validate keyframe execution
                validation_passed = self._validate_keyframes(keyframe_results, animation_config)
                
                execution_time = time.time() - start_time
                return TestResult(
                    test_name=f"keyframes_{animation_config.name}",
                    passed=validation_passed,
                    execution_time=execution_time,
                    metrics={"keyframes_checked": len(keyframe_results)}
                )
                
        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                test_name=f"keyframes_{animation_config.name}",
                passed=False,
                execution_time=execution_time,
                error_message=str(e)
            )

    def _validate_keyframes(self, keyframe_results: List[Dict], config: AnimationTestConfig) -> bool:
        """Validate keyframe execution matches expected properties"""
        try:
            for i, result in enumerate(keyframe_results):
                expected_props = config.expected_properties
                actual_styles = result["styles"]
                
                # Check each expected property
                for prop, expected_values in expected_props.items():
                    if i < len(expected_values):
                        expected_value = expected_values[i]
                        
                        # Map property names
                        style_key = {
                            "transform": "transform",
                            "opacity": "opacity",
                            "filter": "filter",
                            "background-position": "backgroundPosition"
                        }.get(prop, prop)
                        
                        actual_value = actual_styles.get(style_key)
                        
                        # Simplified validation - in production would need more sophisticated comparison
                        if not self._compare_style_values(actual_value, expected_value):
                            self.logger.warning(f"Keyframe validation failed: {prop} expected {expected_value}, got {actual_value}")
                            return False
                            
            return True
        except Exception as e:
            self.logger.error(f"Keyframe validation error: {e}")
            return False

    def _compare_style_values(self, actual: str, expected: str) -> bool:
        """Compare CSS style values with tolerance"""
        if actual == expected:
            return True
            
        # For numeric values, allow small tolerance
        try:
            if "scale" in expected or "translateX" in expected:
                # Extract numeric values and compare with tolerance
                actual_num = float(''.join(filter(str.isdigit or '.-'.__contains__, actual)))
                expected_num = float(''.join(filter(str.isdigit or '.-'.__contains__, expected)))
                return abs(actual_num - expected_num) < 0.1
        except:
            pass
            
        return False

    async def test_particle_system(self, canvas_selector: str = "canvas.particle-system") -> TestResult:
        """Test particle system canvas updates"""
        start_time = time.time()
        
        try:
            if self.use_playwright:
                # Check if canvas element exists
                canvas = await self.page.wait_for_selector(canvas_selector)
                
                # Monitor canvas updates
                frame_data = await self.page.evaluate(f"""
                    const canvas = document.querySelector('{canvas_selector}');
                    const ctx = canvas.getContext('2d');
                    
                    // Capture initial frame
                    const initialImageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
                    
                    // Wait for animation frame
                    return new Promise((resolve) => {{
                        setTimeout(() => {{
                            const newImageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
                            
                            // Compare image data to detect changes
                            let pixelChanges = 0;
                            for (let i = 0; i < initialImageData.data.length; i += 4) {{
                                if (initialImageData.data[i] !== newImageData.data[i] ||
                                    initialImageData.data[i + 1] !== newImageData.data[i + 1] ||
                                    initialImageData.data[i + 2] !== newImageData.data[i + 2]) {{
                                    pixelChanges++;
                                }}
                            }}
                            
                            resolve({{
                                width: canvas.width,
                                height: canvas.height,
                                pixelChanges: pixelChanges,
                                totalPixels: canvas.width * canvas.height
                            }});
                        }}, 100);
                    }});
                """)
                
                # Validate particle system is active
                change_percentage = (frame_data["pixelChanges"] / frame_data["totalPixels"]) * 100
                particles_active = change_percentage > 0.1  # At least 0.1% pixel changes
                
                execution_time = time.time() - start_time
                return TestResult(
                    test_name="particle_system",
                    passed=particles_active,
                    execution_time=execution_time,
                    metrics={
                        "pixel_changes": frame_data["pixelChanges"],
                        "change_percentage": change_percentage
                    }
                )
                
        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                test_name="particle_system",
                passed=False,
                execution_time=execution_time,
                error_message=str(e)
            )

    async def test_interactive_elements(self, selectors: List[str]) -> TestResult:
        """Test interactive element response times"""
        start_time = time.time()
        response_times = []
        
        try:
            for selector in selectors:
                if self.use_playwright:
                    element = await self.page.wait_for_selector(selector)
                    
                    # Measure click response time
                    click_start = time.time()
                    await element.click()
                    
                    # Wait for visual feedback (e.g., class change, style update)
                    await self.page.wait_for_timeout(50)  # Small delay to capture response
                    
                    click_end = time.time()
                    response_times.append((click_end - click_start) * 1000)  # Convert to ms
            
            avg_response_time = sum(response_times) / len(response_times) if response_times else 0
            responsive = avg_response_time < 100  # Less than 100ms is considered responsive
            
            execution_time = time.time() - start_time
            return TestResult(
                test_name="interactive_elements",
                passed=responsive,
                execution_time=execution_time,
                metrics={
                    "avg_response_time_ms": avg_response_time,
                    "max_response_time_ms": max(response_times) if response_times else 0,
                    "elements_tested": len(selectors)
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                test_name="interactive_elements",
                passed=False,
                execution_time=execution_time,
                error_message=str(e)
            )

    async def capture_visual_snapshot(self, test_name: str, is_baseline: bool = False) -> str:
        """Capture visual snapshot for comparison"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if is_baseline:
            filename = f"{test_name}_baseline.png"
            filepath = self.baseline_dir / filename
        else:
            filename = f"{test_name}_{timestamp}.png"
            filepath = self.output_dir / filename
            
        if self.use_playwright:
            await self.page.screenshot(path=str(filepath), full_page=True)
        else:
            self.driver.save_screenshot(str(filepath))
            
        return str(filepath)

    def compare_visual_snapshots(self, baseline_path: str, current_path: str) -> float:
        """Compare two screenshots and return difference score"""
        if not IMAGE_COMPARISON_AVAILABLE:
            self.logger.warning("PIL not available for image comparison")
            return 0.0
            
        try:
            baseline = Image.open(baseline_path)
            current = Image.open(current_path)
            
            # Ensure same size
            if baseline.size != current.size:
                current = current.resize(baseline.size)
            
            # Calculate difference
            diff = ImageChops.difference(baseline, current)
            diff_array = np.array(diff)
            
            # Calculate percentage difference
            total_pixels = diff_array.size
            changed_pixels = np.count_nonzero(diff_array)
            
            return (changed_pixels / total_pixels) * 100
            
        except Exception as e:
            self.logger.error(f"Visual comparison failed: {e}")
            return 0.0

    async def run_comprehensive_test_suite(self, dashboard_url: str) -> Dict[str, Any]:
        """Run complete test suite for AtroZ dashboard"""
        await self.initialize_browser()
        await self.load_dashboard(dashboard_url)
        
        test_results = []
        
        # Test each animation
        for animation_name, config in self.animations.items():
            # Keyframe tests
            keyframe_result = await self.test_animation_keyframes(config)
            test_results.append(keyframe_result)
            
            # Visual snapshot
            snapshot_path = await self.capture_visual_snapshot(f"{animation_name}_animation")
            
            # Compare with baseline if exists
            baseline_path = self.baseline_dir / f"{animation_name}_animation_baseline.png"
            if baseline_path.exists():
                diff_score = self.compare_visual_snapshots(str(baseline_path), snapshot_path)
                keyframe_result.visual_diff_score = diff_score
                keyframe_result.screenshot_path = snapshot_path
        
        # Test particle systems
        particle_result = await self.test_particle_system()
        test_results.append(particle_result)
        
        # Test interactive elements
        interactive_selectors = [
            ".dashboard-button",
            ".menu-item",
            ".control-panel",
            ".data-widget"
        ]
        interactive_result = await self.test_interactive_elements(interactive_selectors)
        test_results.append(interactive_result)
        
        # Performance monitoring
        performance_result = await self.monitor_performance_metrics()
        test_results.append(performance_result)
        
        self.test_results.extend(test_results)
        
        # Generate comprehensive report
        report = self.generate_test_report()
        
        await self.cleanup_browser()
        
        return report

    async def monitor_performance_metrics(self) -> TestResult:
        """Monitor animation performance metrics"""
        start_time = time.time()
        
        try:
            if self.use_playwright:
                # Enable performance metrics
                await self.page.evaluate("""
                    window.performanceMetrics = {
                        frameRates: [],
                        memoryUsage: []
                    };
                    
                    // Monitor frame rate
                    let lastTime = performance.now();
                    function measureFPS() {
                        const now = performance.now();
                        const delta = now - lastTime;
                        const fps = 1000 / delta;
                        window.performanceMetrics.frameRates.push(fps);
                        lastTime = now;
                        
                        if (window.performanceMetrics.frameRates.length < 60) {
                            requestAnimationFrame(measureFPS);
                        }
                    }
                    measureFPS();
                    
                    // Monitor memory (if available)
                    if (performance.memory) {
                        window.performanceMetrics.memoryUsage.push({
                            used: performance.memory.usedJSHeapSize / 1024 / 1024,
                            total: performance.memory.totalJSHeapSize / 1024 / 1024
                        });
                    }
                """)
                
                # Wait for metrics collection
                await self.page.wait_for_timeout(2000)
                
                # Collect metrics
                metrics = await self.page.evaluate("""
                    const frameRates = window.performanceMetrics.frameRates;
                    const avgFPS = frameRates.reduce((a, b) => a + b, 0) / frameRates.length;
                    const minFPS = Math.min(...frameRates);
                    
                    return {
                        avgFPS: avgFPS,
                        minFPS: minFPS,
                        frameRateStability: (minFPS / avgFPS) * 100,
                        memoryUsage: window.performanceMetrics.memoryUsage
                    };
                """)
                
                # Validate performance thresholds
                performance_good = (
                    metrics["avgFPS"] >= 30 and
                    metrics["frameRateStability"] >= 80  # 80% stability
                )
                
                execution_time = time.time() - start_time
                return TestResult(
                    test_name="performance_monitoring",
                    passed=performance_good,
                    execution_time=execution_time,
                    metrics=metrics
                )
                
        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                test_name="performance_monitoring",
                passed=False,
                execution_time=execution_time,
                error_message=str(e)
            )

    def generate_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        passed_tests = [r for r in self.test_results if r.passed]
        failed_tests = [r for r in self.test_results if not r.passed]
        
        report = {
            "summary": {
                "total_tests": len(self.test_results),
                "passed": len(passed_tests),
                "failed": len(failed_tests),
                "success_rate": (len(passed_tests) / len(self.test_results)) * 100 if self.test_results else 0,
                "total_execution_time": sum(r.execution_time for r in self.test_results),
                "timestamp": datetime.now().isoformat()
            },
            "animation_tests": {
                "organicPulse": self._get_animation_test_summary("organicPulse"),
                "neuralFlow": self._get_animation_test_summary("neuralFlow"),
                "glitchEffect": self._get_animation_test_summary("glitchEffect")
            },
            "particle_system": self._get_test_summary("particle_system"),
            "interactive_elements": self._get_test_summary("interactive_elements"),
            "performance_metrics": self._get_test_summary("performance_monitoring"),
            "visual_regression": {
                "snapshots_captured": len([r for r in self.test_results if r.screenshot_path]),
                "visual_differences": [
                    {
                        "test": r.test_name,
                        "diff_score": r.visual_diff_score,
                        "screenshot": r.screenshot_path
                    }
                    for r in self.test_results if r.visual_diff_score is not None
                ]
            },
            "failed_tests": [
                {
                    "name": r.test_name,
                    "error": r.error_message,
                    "execution_time": r.execution_time
                }
                for r in failed_tests
            ]
        }
        
        # Save report
        report_path = self.report_dir / f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        return report

    def _get_animation_test_summary(self, animation_name: str) -> Dict[str, Any]:
        """Get test summary for specific animation"""
        animation_tests = [r for r in self.test_results if animation_name in r.test_name]
        
        if not animation_tests:
            return {"status": "not_tested"}
            
        return {
            "status": "passed" if all(r.passed for r in animation_tests) else "failed",
            "keyframe_validation": any("keyframes" in r.test_name for r in animation_tests),
            "visual_snapshot": any(r.screenshot_path for r in animation_tests),
            "performance": any(r.metrics for r in animation_tests),
            "execution_time": sum(r.execution_time for r in animation_tests)
        }

    def _get_test_summary(self, test_name: str) -> Dict[str, Any]:
        """Get summary for specific test"""
        test_results = [r for r in self.test_results if r.test_name == test_name]
        
        if not test_results:
            return {"status": "not_tested"}
            
        result = test_results[0]
        return {
            "status": "passed" if result.passed else "failed",
            "execution_time": result.execution_time,
            "metrics": result.metrics,
            "error": result.error_message
        }

    async def cleanup_browser(self):
        """Clean up browser resources"""
        if self.use_playwright and self.browser:
            await self.browser.close()
        elif self.driver:
            self.driver.quit()


# CLI Interface and Test Runner
async def run_visual_tests(dashboard_url: str = "http://localhost:3000"):
    """Run the complete visual testing suite"""
    framework = VisualTestingFramework(use_playwright=PLAYWRIGHT_AVAILABLE)
    
    print("🎭 Starting AtroZ Dashboard Visual Testing Framework")
    print(f"📱 Testing URL: {dashboard_url}")
    print(f"🛠️  Using: {'Playwright' if framework.use_playwright else 'Selenium'}")
    
    try:
        report = await framework.run_comprehensive_test_suite(dashboard_url)
        
        print("\n📊 Test Results Summary:")
        print(f"✅ Passed: {report['summary']['passed']}")
        print(f"❌ Failed: {report['summary']['failed']}")
        print(f"📈 Success Rate: {report['summary']['success_rate']:.1f}%")
        print(f"⏱️  Total Time: {report['summary']['total_execution_time']:.2f}s")
        
        if report['summary']['failed'] > 0:
            print("\n🚨 Failed Tests:")
            for failed in report['failed_tests']:
                print(f"  - {failed['name']}: {failed['error']}")
        
        print(f"\n📋 Full report saved to: visual_tests/reports/")
        return report
        
    except Exception as e:
        print(f"❌ Test suite failed: {e}")
        return None


if __name__ == "__main__":
    import sys
    
    dashboard_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:3000"
    
    if not PLAYWRIGHT_AVAILABLE and not SELENIUM_AVAILABLE:
        print("❌ Neither Playwright nor Selenium is available!")
        print("Install with: pip install playwright selenium")
        sys.exit(1)
    
    asyncio.run(run_visual_tests(dashboard_url))