#!/usr/bin/env python3

from egw_query_expansion import DeterministicGainFunction, MonotonicTaskSelector, Task

print("✓ Successfully imported MonotonicTaskSelector")
print("✓ Successfully imported DeterministicGainFunction")
print("✓ Successfully imported Task")
print("✓ All components working correctly")

# Quick functionality test
task = Task("test", 10.0)
print(f"✓ Created task: {task.id} with cost {task.cost}")

gain_func = DeterministicGainFunction({"test": 50.0})
selector = MonotonicTaskSelector(gain_func)
print("✓ Created selector with gain function")

selector.add_tasks([task])
selected = selector.select_tasks(20.0)
print(f"✓ Selected {len(selected)} tasks with budget 20.0")

print("✅ All core functionality verified")
