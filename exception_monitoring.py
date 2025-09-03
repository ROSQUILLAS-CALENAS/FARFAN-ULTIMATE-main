"""Exception monitoring dashboard and analytics for telemetry data."""

import json
import time
# # # from datetime import datetime, timedelta  # Module not found  # Module not found  # Module not found
# # # from typing import Dict, List, Any  # Module not found  # Module not found  # Module not found
# # # from collections import defaultdict, Counter  # Module not found  # Module not found  # Module not found
import threading
import logging

# # # from exception_telemetry import get_telemetry_stats, clear_telemetry_stats  # Module not found  # Module not found  # Module not found


class ExceptionMonitor:
    """Monitor and analyze exception patterns for debugging and optimization."""
    
    def __init__(self):
        self.monitoring_active = False
        self._monitoring_thread = None
        self._exception_history = []
        self._lock = threading.Lock()
        self.logger = logging.getLogger("exception_monitor")
        
    def start_monitoring(self, interval_seconds: int = 60):
        """Start continuous exception monitoring."""
        if self.monitoring_active:
            return
            
        self.monitoring_active = True
        self._monitoring_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self._monitoring_thread.start()
        self.logger.info(f"Exception monitoring started with {interval_seconds}s interval")
    
    def stop_monitoring(self):
        """Stop exception monitoring."""
        self.monitoring_active = False
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5.0)
        self.logger.info("Exception monitoring stopped")
    
    def _monitor_loop(self, interval_seconds: int):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                stats = get_telemetry_stats()
                if stats:
                    with self._lock:
                        self._exception_history.append({
                            "timestamp": datetime.utcnow().isoformat(),
                            "stats": dict(stats)
                        })
                        
                        # Keep only last 24 hours of data
                        cutoff_time = datetime.utcnow() - timedelta(hours=24)
                        self._exception_history = [
                            entry for entry in self._exception_history
                            if datetime.fromisoformat(entry["timestamp"]) > cutoff_time
                        ]
                
                time.sleep(interval_seconds)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(interval_seconds)
    
    def get_exception_trends(self) -> Dict[str, Any]:
        """Get exception trends and patterns."""
        with self._lock:
            if not self._exception_history:
                return {"error": "No exception data available"}
            
            # Aggregate exception counts by component and type
            component_totals = defaultdict(Counter)
            hourly_counts = defaultdict(int)
            
            for entry in self._exception_history:
                timestamp = datetime.fromisoformat(entry["timestamp"])
                hour_key = timestamp.strftime("%Y-%m-%d %H:00")
                
                total_exceptions = 0
                for component, exc_types in entry["stats"].items():
                    for exc_type, count in exc_types.items():
                        component_totals[component][exc_type] += count
                        total_exceptions += count
                
                hourly_counts[hour_key] += total_exceptions
            
            return {
                "component_breakdown": dict(component_totals),
                "hourly_trends": dict(hourly_counts),
                "total_components": len(component_totals),
                "data_points": len(self._exception_history),
                "time_range": {
                    "start": self._exception_history[0]["timestamp"] if self._exception_history else None,
                    "end": self._exception_history[-1]["timestamp"] if self._exception_history else None
                }
            }
    
    def get_top_error_components(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get components with the most exceptions."""
        with self._lock:
            component_counts = defaultdict(int)
            
            for entry in self._exception_history:
                for component, exc_types in entry["stats"].items():
                    component_counts[component] += sum(exc_types.values())
            
            sorted_components = sorted(
                component_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            return [
                {"component": comp, "exception_count": count}
                for comp, count in sorted_components[:limit]
            ]
    
    def get_exception_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in exception types and frequencies."""
        with self._lock:
            if not self._exception_history:
                return {"error": "No exception data available"}
            
            # Most common exception types across all components
            all_exception_types = Counter()
            component_exception_patterns = defaultdict(Counter)
            
            for entry in self._exception_history:
                for component, exc_types in entry["stats"].items():
                    for exc_type, count in exc_types.items():
                        all_exception_types[exc_type] += count
                        component_exception_patterns[component][exc_type] += count
            
            # Find components with recurring exceptions
            recurring_patterns = {}
            for component, exc_types in component_exception_patterns.items():
                if len(exc_types) > 1:  # Multiple exception types
                    recurring_patterns[component] = dict(exc_types.most_common())
            
            return {
                "most_common_exceptions": dict(all_exception_types.most_common(10)),
                "recurring_exception_patterns": recurring_patterns,
                "unique_exception_types": len(all_exception_types),
                "components_with_multiple_exception_types": len(recurring_patterns)
            }
    
    def generate_monitoring_report(self) -> str:
        """Generate a comprehensive monitoring report."""
        trends = self.get_exception_trends()
        top_components = self.get_top_error_components()
        patterns = self.get_exception_patterns()
        
        report = []
        report.append("=" * 60)
        report.append("EXCEPTION MONITORING REPORT")
        report.append("=" * 60)
        report.append(f"Generated at: {datetime.utcnow().isoformat()}")
        report.append("")
        
        # Time range
        if trends.get("time_range"):
            report.append(f"Data Range: {trends['time_range']['start']} to {trends['time_range']['end']}")
            report.append(f"Data Points: {trends['data_points']}")
            report.append("")
        
        # Top error-prone components
        report.append("TOP ERROR-PRONE COMPONENTS:")
        report.append("-" * 30)
        for i, comp_data in enumerate(top_components[:5], 1):
            report.append(f"{i}. {comp_data['component']}: {comp_data['exception_count']} exceptions")
        report.append("")
        
        # Most common exceptions
        if patterns.get("most_common_exceptions"):
            report.append("MOST COMMON EXCEPTION TYPES:")
            report.append("-" * 35)
            for exc_type, count in list(patterns["most_common_exceptions"].items())[:5]:
                report.append(f"  {exc_type}: {count} occurrences")
            report.append("")
        
        # Recurring patterns
        if patterns.get("recurring_exception_patterns"):
            report.append("COMPONENTS WITH RECURRING EXCEPTION PATTERNS:")
            report.append("-" * 50)
            for component, exc_types in patterns["recurring_exception_patterns"].items():
                report.append(f"  {component}:")
                for exc_type, count in list(exc_types.items())[:3]:
                    report.append(f"    - {exc_type}: {count}")
            report.append("")
        
        return "\n".join(report)
    
    def export_telemetry_data(self, filepath: str):
        """Export telemetry data to JSON file."""
        with self._lock:
            export_data = {
                "exported_at": datetime.utcnow().isoformat(),
                "exception_history": self._exception_history,
                "trends": self.get_exception_trends(),
                "patterns": self.get_exception_patterns()
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
        
        self.logger.info(f"Telemetry data exported to {filepath}")


# Global monitor instance
_monitor = ExceptionMonitor()

def start_exception_monitoring(interval_seconds: int = 60):
    """Start global exception monitoring."""
    _monitor.start_monitoring(interval_seconds)

def stop_exception_monitoring():
    """Stop global exception monitoring."""
    _monitor.stop_monitoring()

def get_monitoring_report() -> str:
    """Get current monitoring report."""
    return _monitor.generate_monitoring_report()

def export_monitoring_data(filepath: str):
    """Export monitoring data to file."""
    _monitor.export_telemetry_data(filepath)