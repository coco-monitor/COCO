import asyncio
import functools
import logging
import weakref
from typing import Any, Callable, Dict, Optional
from .config import MonitorConfig

logger = logging.getLogger(__name__)

# Delayed import to avoid circular imports
async_monitor_with_time_travel = None

import threading

# Thread-safe global variables - each thread has independent storage
_thread_local = threading.local()

# Process-level graph registry (supports multi-thread sharing)
_global_graph_registry = weakref.WeakValueDictionary()
_graph_registry_lock = threading.RLock()

def _get_graph_registry():
    """Get current thread's graph registry"""
    if not hasattr(_thread_local, 'graph_registry'):
        _thread_local.graph_registry = weakref.WeakValueDictionary()
    return _thread_local.graph_registry

def _get_current_config():
    """Get current thread's config"""
    return getattr(_thread_local, 'current_config', None)

def _set_current_config(config):
    """Set current thread's config"""
    _thread_local.current_config = config

def _get_coco_tasks():
    """Get current thread's coco tasks"""
    if not hasattr(_thread_local, 'coco_tasks'):
        _thread_local.coco_tasks = []
    return _thread_local.coco_tasks

def _get_workflow_control():
    """Get current thread's workflow control"""
    if not hasattr(_thread_local, 'workflow_control'):
        _thread_local.workflow_control = {"error_detected": False}
    return _thread_local.workflow_control

def set_execution_config(config: Dict[str, Any]):
    """Set the current execution config for CoCo monitoring."""
    _set_current_config(config)

def get_execution_config() -> Optional[Dict[str, Any]]:
    """Get the current execution config."""
    return _get_current_config()

def add_coco_task(task):
    """Add a CoCo monitoring task to the current thread's list."""
    _get_coco_tasks().append(task)

def get_coco_tasks():
    """Get all CoCo monitoring tasks for current thread."""
    return _get_coco_tasks().copy()

def clear_coco_tasks():
    """Clear and cancel all CoCo monitoring tasks for current thread."""
    coco_tasks = _get_coco_tasks()
    
    # Cancel all running CoCo monitoring tasks
    for task in coco_tasks:
        if not task.done():
            logger.debug(f"🚫 Cancelling CoCo monitoring task: {task}")
            task.cancel()
    
    # Clear task list
    coco_tasks.clear()
    logger.debug("✅ All CoCo monitoring tasks cleared and cancelled")

# Time travel state management - just for basic cleanup
def clear_time_travel_state():
    """Clear any time travel state"""
    logger.debug("Time travel state cleared")

def set_error_detected(detected: bool = True):
    """Set error detection status (for monitoring) - atomic operation to prevent race conditions"""
    workflow_control = _get_workflow_control()
    
    if detected:
        # Atomic operation: only the first task to set error_detected can succeed
        if workflow_control.get("error_detected", False):
            return False  # Another monitoring task is already handling the error
        workflow_control["error_detected"] = True
        logger.info("🚨 Error detected by CoCo")
        return True  # Successfully set error status
    else:
        workflow_control["error_detected"] = False
        return True

def reset_workflow_control():
    """Reset workflow control status"""
    workflow_control = _get_workflow_control()
    workflow_control.clear()
    workflow_control["error_detected"] = False

def register_graph(graph_id: str, graph):
    """Register a graph instance for CoCo monitoring (process-wide, thread-safe)."""
    with _graph_registry_lock:
        _global_graph_registry[graph_id] = graph
    logger.info(f"Registered graph {graph_id} for CoCo monitoring (process-wide)")

def get_graph(graph_id: str = "default"):
    """Get registered graph instance (process-wide, thread-safe)."""
    with _graph_registry_lock:
        return _global_graph_registry.get(graph_id)

def coco_monitor(
    node_id: str,
    model_config: MonitorConfig,
    prompt_template: Optional[str] = None,
    confidence_threshold: float = 0.9,
    graph_id: str = "default",
):
    """CoCo monitoring decorator with LangGraph time travel.
    
    Args:
        node_id: Unique identifier for the node
        prompt_template: Format string template that can reference any keys from state, e.g.:
                        "Task Description: {task}\nCriteria: {standard}\nNode Output: {node2_output}\n"
                        "Please check if the output meets the criteria."
                        Keys in {} will be automatically extracted from state.
                        Must be provided, will raise ValueError if None.
        confidence_threshold: Minimum confidence for error reporting
        graph_id: Graph instance identifier for time travel
        monitor_model: Custom model for CoCo monitoring
        monitor_api_key: API key for the monitoring model
        monitor_base_url: Base URL for the monitoring model API
    """
    # Parameter validation
    if prompt_template is None:
        raise ValueError(f"prompt_template is required for coco_monitor on node '{node_id}'")
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(state, **kwargs):
            # 1. Check and apply correction instructions (if exists)
            correction_key = f"coco_correction_{node_id}"
            if hasattr(state, 'get') and correction_key in state:
                print(f"🔄 CoCo: Applying correction to node {node_id}")
                state = apply_correction(state, state[correction_key], node_id)
            
            # 2. Execute node logic
            result = await func(state, **kwargs) if asyncio.iscoroutinefunction(func) else func(state, **kwargs)
            
            # 3. Start independent CoCo monitoring (non-blocking to current workflow)
            config = get_execution_config()
            if config:
                print(f"🔍 CoCo: Starting monitoring for node {node_id}")
                
                # Delayed import to avoid circular dependencies
                global async_monitor_with_time_travel
                if async_monitor_with_time_travel is None:
                    from .monitor import async_monitor_with_time_travel
                
                
                coco_task = asyncio.create_task(
                    async_monitor_with_time_travel(
                        graph_id=graph_id,
                        config=config,
                        node_id=node_id,
                        full_state=state,
                        prompt_template=prompt_template,
                        threshold=confidence_threshold,
                        monitor_config=model_config
                    )
                )
                
                # Track task but don't wait for its completion
                add_coco_task(coco_task)
            
            # 4. Return result immediately, don't wait for CoCo processing
            return result
        return wrapper
    return decorator


def enhance_prompt_with_correction(original_prompt: str, state:dict, node_id: str, output_key: str) -> str:
    """Generic prompt enhancement mechanism - add error hints when correction is detected"""
    correction_key = f"coco_correction_{node_id}"
    
    if correction_key not in state:
        return original_prompt
    
    correction_data = state[correction_key]
    error_description = correction_data.get("error_description", "")
    correction_hint = correction_data.get("correction_hint", "")
    
    # Get original error output from correction_data
    if "failed_output" not in correction_data:
        raise ValueError(f"Missing 'failed_output' in correction_data for node '{node_id}'. This indicates a system error.")
    
    raw_failed_output = correction_data["failed_output"]
    
    # If output_key is provided, try to extract specified field from failed_output
    if output_key:
        failed_output = str(raw_failed_output[output_key])
    else:
        failed_output = str(raw_failed_output)
    
    enhanced_prompt = f"""{original_prompt}

──────────────────────────────────────────────
[System Notice] A potential issue was detected in your previous response
──────────────────────────────────────────────

Context
- Previous output: \n{failed_output}
- Potential issue: \n{error_description}
- Suggested improvement: \n{correction_hint}

Instructions  
1. **Think according to the task** to judge whether the reported issue is right or not.
2. **If the issue is real**, revise your previous answer to resolve the issue and produce the correct solution.
3. **If the issue is not real**, keep your original answer and can add a brief (1-2 line) note explaining why it remains correct.  
4. You can (not have to) reveal your reasoning in a `<reason></reason>` block before the final answer.
"""
    
    return enhanced_prompt

def apply_correction(state, correction_data: Dict[str, Any], node_id: str):
    """Apply correction strategy to state based on error type.
    
    Args:
        state: Current workflow state
        correction_data: Correction instructions from CoCo
        node_id: Node being corrected
    
    Returns:
        Modified state with corrections applied
    """
    try:
        error_type = correction_data.get("error_type", "")
        correction_hint = correction_data.get("correction_hint", "")
        
        logger.info(f"Applying {error_type} correction to {node_id}: {correction_hint}")
        
        # Apply corrections based on error type
        if error_type == "format_error":
            state = apply_format_correction(state, correction_data)
        elif error_type == "logic_error":
            state = apply_logic_correction(state, correction_data)  
        elif error_type == "content_error":
            state = apply_content_correction(state, correction_data)
        
        return state
        
    except Exception as e:
        logger.error(f"Failed to apply correction for {node_id}: {e}")
        return state

def apply_format_correction(state, correction_data):
    """Apply format-related corrections."""
    # Example: Fix text formatting based on hints
    if hasattr(state, 'get') and 'input_text' in state:
        hint = correction_data.get("correction_hint", "").lower()
        if "uppercase" in hint:
            state['input_text'] = state['input_text'].upper()
        elif "lowercase" in hint:
            state['input_text'] = state['input_text'].lower()
        elif "title" in hint:
            state['input_text'] = state['input_text'].title()
    return state

def apply_logic_correction(state, correction_data):
    """Apply logic-related corrections."""
    # Example: Adjust processing parameters
    original_input = correction_data.get("original_input", {})
    if hasattr(state, 'update') and original_input:
        # Restore some original parameters that might have been processed incorrectly
        for key, value in original_input.items():
            if key.endswith('_param') or key.endswith('_setting'):
                state[key] = value
    return state

def apply_content_correction(state, correction_data):
    """Apply content-related corrections.""" 
    # Example: Add missing information or fix content
    hint = correction_data.get("correction_hint", "")
    if "missing" in hint.lower() and hasattr(state, 'get'):
        # Add placeholder for missing content
        state['coco_content_fixed'] = True
    return state


async def coco_execute_workflow(workflow, initial_state, config=None, graph_id: str = "default", max_wait_time: float = 60.0):
    """
    CoCo-enhanced workflow execution function - integrate CoCo monitoring with one line of code
    
    Usage:
        # Replace: result = await workflow.ainvoke(initial_state, config)
        # With: result = await coco_execute_workflow(workflow, initial_state, config)
    
    Args:
        workflow: Compiled LangGraph workflow
        initial_state: Initial state
        config: Runtime configuration (optional, will auto-generate if not provided)
        graph_id: Graph ID (for CoCo monitoring)
        max_wait_time: Maximum wait time (seconds)
    
    Returns:
        Workflow execution result (returns corrected result if CoCo performed correction)
    """
    import asyncio
    import time
    
    # 1. Prepare configuration
    if config is None:
        thread_id = f"coco_{int(time.time() * 1000)}"
        config = {"configurable": {"thread_id": thread_id}}
    
    thread_id = config.get("configurable", {}).get("thread_id")
    if not thread_id:
        raise ValueError("Config must contain thread_id")
    
    # 2. Ensure workflow is registered and set execution configuration
    # Check if already registered to avoid duplicate registration
    if get_graph(graph_id) is None:
        register_graph(graph_id, workflow)
        print(f"📝 Registering workflow to CoCo system: {graph_id}")
    else:
        print(f"✅ Workflow already exists in CoCo system: {graph_id}")
    
    set_execution_config(config)
    
    # 3. Reset state
    reset_workflow_control()
    clear_coco_tasks()
    clear_time_travel_state()
    
    try:
        # 4. Execute original workflow
        original_result = await workflow.ainvoke(initial_state, config=config)
        
        # 5. Smart wait for CoCo processing
        correction_result = await _wait_for_coco_smart(
            thread_id=thread_id, 
            max_wait_time=max_wait_time
        )
        
        # 6. Return final result
        if correction_result is not None:
            print("🎯 CoCo: Returning corrected result")
            return correction_result
        else:
            return original_result
            
    except Exception as e:
        logger.error(f"❌ CoCo workflow execution failed: {e}")
        raise
    finally:
        # Cleanup
        clear_coco_tasks()

async def _wait_for_coco_smart(thread_id: str, max_wait_time: float = 60.0):
    """Smart wait for CoCo processing completion (internal function)"""
    from .monitor import _get_running_tasks
    
    start_time = asyncio.get_event_loop().time()
    
    # Phase 1: Wait for CoCo error detection (increased wait time to match new detailed analysis prompt)
    detection_timeout = 40.0  # Increased to 40 seconds, giving Monitor sufficient time for detailed analysis
    
    while True:
        elapsed = asyncio.get_event_loop().time() - start_time
        
        if elapsed >= detection_timeout:
            print("⏰ CoCo: Monitoring wait timeout")
            break
            
        # Check if error was detected
        if _get_workflow_control().get("error_detected", False):
            logger.info("🚨 CoCo detected error, starting automatic correction")
            break
            
        # Check CoCo task status
        coco_tasks = get_coco_tasks()
        if coco_tasks:
            running = sum(1 for task in coco_tasks if not task.done())
            if running == 0:
                print("📋 CoCo: Monitoring tasks completed")
                return None
        else:
            print("📋 CoCo: No monitoring tasks running")
            return None
            
        await asyncio.sleep(0.1)
    
    # Phase 2: If error detected, wait for time travel correction
    if _get_workflow_control().get("error_detected", False):
        # Wait for time travel task to appear
        time_travel_task = None
        wait_start = asyncio.get_event_loop().time()
        
        while True:
            elapsed = asyncio.get_event_loop().time() - start_time
            
            # Check total timeout
            if elapsed >= max_wait_time:
                logger.warning(f"⏰ CoCo correction timeout ({elapsed:.1f}s)")
                return None
            
            # Look for time travel task
            running_tasks = _get_running_tasks()
            if thread_id in running_tasks:
                time_travel_task = running_tasks[thread_id]
                break
                
            # If waiting too long without time travel task, might be an issue
            if (asyncio.get_event_loop().time() - wait_start) > 10.0:
                logger.warning("⚠️ No correction task found, returning original result")
                return None
                
            await asyncio.sleep(0.2)
        
        # Wait for time travel task completion
        if time_travel_task:
            try:
                result = await time_travel_task
                print(f"✅ CoCo: Automatic correction completed ({elapsed:.1f}s)")
                return result
            except asyncio.CancelledError:
                logger.warning("⚠️ Correction task was cancelled")
                return None
            except Exception as e:
                logger.error(f"❌ Correction task error: {e}")
                return None
    
    return None