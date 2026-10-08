from __future__ import annotations
from typing import Any, Dict, Optional
import asyncio
import json
import logging
import os
import string
from datetime import datetime
from langchain_qwq import ChatQwen 
from langchain_core.messages import HumanMessage
from .config import MonitorConfig


logger = logging.getLogger(__name__)
from .decorator import (
    _get_workflow_control,
    set_error_detected,
    clear_coco_tasks,
    get_graph,
)


import threading as _threading
_monitor_thread_local = _threading.local()

def _get_running_tasks():
    """Get current thread's running tasks"""
    if not hasattr(_monitor_thread_local, 'running_tasks'):
        _monitor_thread_local.running_tasks = {}
    return _monitor_thread_local.running_tasks

def _register_task(thread_id: str, task: asyncio.Task) -> None:
    _get_running_tasks()[thread_id] = task

def _pop_task(thread_id: str) -> Optional[asyncio.Task]:
    return _get_running_tasks().pop(thread_id, None)

class ErrorDetectionEngine:
    """Generate structured error reports with LLM."""

    def __init__(self, monitor_config: MonitorConfig = None) -> None:
        if not isinstance(monitor_config, MonitorConfig):
            raise ValueError("monitor_config must be MonitorConfig Class")

        if monitor_config:
            model = monitor_config.model
            api_key = monitor_config.api_key
            base_url = monitor_config.base_url
            temperature = monitor_config.temperature
            top_p = monitor_config.top_p
            enable_thinking = monitor_config.enable_thinking
        else:
            model = os.getenv("MODEL_NAME")
            api_key = os.getenv("OPENAI_API_KEY")
            base_url = os.getenv("BASE_URL")
            temperature = 0.01
            top_p = 1.0
            enable_thinking = False
        
        self.llm = ChatQwen(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            top_p=top_p,
            enable_thinking=enable_thinking
        )
        
        # Log the monitoring configuration being used
        logger.debug(f"CoCo monitoring engine initialized: model={model}, base_url={base_url}")

    async def detect_error(
        self,
        *,
        node_id: str,
        full_state: Dict,
        prompt_template: str | None = None,
        timeout: float = 30.0,
    ) -> Dict[str, Any]:
        """Return dict with keys: has_error / confidence / …"""

        if prompt_template:
            # Parse required keys from template
            required_keys = self._extract_template_keys(prompt_template)
            # Extract corresponding values from state
            format_data = self._extract_values_from_state(full_state, required_keys)
            # Generic monitor prefix - emphasize reasoning process
            prefix = """You are a monitoring LLM. Your task is to carefully analyze the provided information to determine if there are any issues or errors. Think step by step. First conduct a thorough analysis, then provide your conclusion in JSON format. Following are the information: \n"""
            
            # Format task-specific template
            task_content = prompt_template.format(**format_data)
            
            # Generic output format guidance
            suffix = """Please follow this response format:

## Analysis
[Provide your detailed step-by-step analysis here. Explain your reasoning process, check test cases if applicable, and identify any specific issues.]

## Conclusion
Based on my analysis above:

```json
{
    "has_error": true/false,
    "confidence": 0.0-1.0,
    "error_type": "logic_error|format_error|content_error|other",
    "error_description": "detailed description of any specific issue found",
    "correction_hint": "specific actionable suggestion for improvement"
}
```"""
            
            prompt = prefix + task_content + suffix
        else:
            # template has to be provided
            raise ValueError(f"prompt_template is required for node {node_id}")

        try:
            resp = await asyncio.wait_for(
                self.llm.ainvoke([HumanMessage(content=prompt)]),
                timeout=timeout,
            )
            
            # Log monitor interaction
            await self._log_monitor_interaction(node_id, prompt, resp.content, format_data)
            
        except asyncio.TimeoutError:
            logger.warning("🔌 LLM timeout while detecting error for %s", node_id)
            return {
                "error": "Timeout",
                "text": "",
                "node_id": node_id,
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as exc:
            logger.error("LLM failure while detecting error for %s: %s", node_id, exc)
            return {
                "error": "other",
                "text": exc,
                "node_id": node_id,
                "timestamp": datetime.now().isoformat(),
            }

        return self._parse_response(resp.content, node_id)

    async def _log_monitor_interaction(self, node_id: str, prompt: str, response: str, format_data: Dict[str, str]):
        """Log monitor interaction data for performance analysis"""
        try:
            import os
            import json
            from datetime import datetime
            
            # Create log directory
            log_dir = "../results/monitor_logs"
            os.makedirs(log_dir, exist_ok=True)
            
            # Parse response to get structured data
            parsed_result = self._parse_response(response, node_id)
            
            # Build log entry
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "node_id": node_id,
                "prompt_length": len(prompt),
                "prompt": prompt,
                "response": response,
                "response_length": len(response),
                "format_data": format_data,
                "parsed_result": parsed_result,
                "has_error": parsed_result.get("has_error", False),
                "confidence": parsed_result.get("confidence", 0),
                "error_type": parsed_result.get("error_type", ""),
                "error_description": parsed_result.get("error_description", ""),
                "correction_hint": parsed_result.get("correction_hint", "")
            }
            
            # Save logs by date and node ID separately
            date_str = datetime.now().strftime("%Y%m%d")
            log_file = f"{log_dir}/monitor_{node_id}_{date_str}.jsonl"
            
            # Append to log file
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
                
        except Exception as e:
            logger.error(f"Failed to log monitor interaction: {e}")

    def _extract_template_keys(self, template: str) -> list:
        """Safely extract all keys from formatted string template using string.Formatter"""
        formatter = string.Formatter()
        keys = []
        try:
            for _, field_name, _, _ in formatter.parse(template):
                if field_name is not None:
                    keys.append(field_name)
        except ValueError as e:
            raise ValueError(f"Invalid template format: {e}")
        return keys
    
    def _extract_values_from_state(self, state: Dict, required_keys: list) -> Dict[str, str]:
        """Extract specific values from full state"""
        format_data = {}
        
        for key in required_keys:
            format_data[key] = state.get(key, None)

            if not format_data[key]:
                raise KeyError(f"Key '{key}' not found in state")
        
        return format_data

    def _parse_response(self, text: str, node_id: str) -> Dict[str, Any]:
        text = text.strip()
        parsed = {}
        try:
            # first, parse answer from ```json```
            json_start = text.lower().find("```json")
            if json_start != -1:
                json_content_start = text.find("\n", json_start) + 1
                json_end = text.find("```", json_content_start)
                
                if json_end != -1:
                    json_str = text[json_content_start:json_end].strip()
                    parsed = json.loads(json_str)
                else:
                    raise ValueError("Found ```json but no closing ```")
            else:
                # fallback to find `{}`
                logger.warning(f"⚠️  No ```json``` block found in monitor response for {node_id}, using fallback {{}} matching")
                
                last_brace_pos = text.rfind("}")
                if last_brace_pos == -1:
                    raise ValueError("No closing brace found")
                
                brace_count = 0
                start_pos = -1
                
                for i in range(last_brace_pos, -1, -1):
                    if text[i] == '}':
                        brace_count += 1
                    elif text[i] == '{':
                        brace_count -= 1
                        if brace_count == 0:
                            start_pos = i
                            break
                
                if start_pos == -1:
                    raise ValueError("No matching opening brace found")
                
                # extract json and parse it
                json_str = text[start_pos:last_brace_pos + 1]
                parsed = json.loads(json_str)
                
                print("⚠️  WARNING: Used fallback {} matching. Monitor should use ```json``` blocks for better parsing.")
                
        except Exception as e:
            logger.error(f"❌ Failed to parse monitor response for {node_id}: {e}")
            logger.error(f"❌ Response text: {text[:200]}...")
            parsed = {}

        if parsed:
            return {
                "has_error": bool(parsed.get("has_error")),
                "confidence": float(parsed.get("confidence", 0)),
                "error_type": parsed.get("error_type", "other"),
                "error_description": parsed.get("error_description", ""),
                "correction_hint": parsed.get("correction_hint", ""),
                "node_id": node_id,
                "timestamp": datetime.now().isoformat(),
            }

        return {
            "error": "parse_error",
            "text": text,
            "node_id": node_id,
            "timestamp": datetime.now().isoformat(),
        }

async def direct_time_travel(
    *,
    graph_id: str,
    config: Dict[str, Any],
    node_id: str,
    correction_data: Dict[str, Any],
) -> None:
    """Cancel current run ➜ patch checkpoint ➜ relaunch from `node_id`."""

    graph = get_graph(graph_id)
    if graph is None:
        logger.error("Graph %s not found – cannot time‑travel ✈️", graph_id)
        return

    thread_id: Optional[str] = config.get("configurable", {}).get("thread_id")

    # 1️⃣ cancel old run
    if thread_id:
        prev = _pop_task(thread_id)
        if prev and not prev.done():
            logger.info("🚫 Cancelling previous workflow for thread %s (task: %s)", thread_id, prev)
            prev.cancel()
            try:
                await prev
            except asyncio.CancelledError:
                logger.debug("Previous workflow cancelled cleanly")
        else:
            logger.debug("No active workflow to cancel for thread %s", thread_id)

    try:
        # Get the history of state, find out the checkpoint
        states = list(graph.get_state_history(config))
        target_checkpoint = None
        
        for i, state in enumerate(states):
            # If next field contains the error node, this is the checkpoint we want
            if state.next and node_id in state.next:
                target_checkpoint = state
                break
        
        if not target_checkpoint:
            # If suitable checkpoint not found, use the most recent state
            target_checkpoint = states[0] if states else None
            logger.warning(f"⚠️ No checkpoint found with next={node_id}, using most recent state")
            
        if not target_checkpoint:
            logger.error("No suitable checkpoint found for time-travel")
            return
            
        # 2️⃣ prepare patched checkpoint 
        patch_key = f"coco_correction_{node_id}"
        
        # Merge original state and correction info, ensure no loss of original data
        target_values = target_checkpoint.values if target_checkpoint.values else {}
        state_patch = {
            **target_values,  # Keep original state  
            patch_key: correction_data,  # Add correction info
            "coco_reexecution": True  # Add re-execution flag
        }
        
        logger.info(f"Preparing state_patch for {node_id}: {patch_key}")
        logger.info(f"Found target checkpoint before {node_id}, next: {target_checkpoint.next}")
        
        # Add correction info to target checkpoint
        new_config = await graph.aupdate_state(
            config=target_checkpoint.config,
            values=state_patch,
        )
        
        # Wait a short time to ensure write completion
        await asyncio.sleep(0.1) 
        
        # Verify if state was written correctly
        updated_state = graph.get_state(new_config)
        
        if patch_key not in updated_state.values:
            logger.error(f"❌ Correction key {patch_key} NOT found in updated state!")
            logger.error(f"❌ Available keys: {list(updated_state.values.keys())}")
        else:
            logger.info(f"✅ Correction data successfully written to state")
        
    except Exception as exc:
        logger.error("Failed to write checkpoint for time‑travel: %s", exc)
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return

    # 3️⃣ clear other monitoring tasks and launch new run
    clear_coco_tasks()  # Cancel all other CoCo monitoring tasks
    # Note: Don't immediately reset error_detected, let other monitoring tasks know correction is in progress

    try:
        # Continue executing workflow from specified configuration
        logger.info(f"🚀 About to create new workflow task with config: {new_config}")
        new_task = asyncio.create_task(graph.ainvoke(None, new_config))
        if thread_id:
            # Check if there are old tasks
            running_tasks = _get_running_tasks()
            old_task = running_tasks.get(thread_id)
            if old_task:
                logger.warning(f"⚠️ Overriding existing task for thread {thread_id}: {old_task}")
            _register_task(thread_id, new_task)
        logger.info("✅ Relaunched workflow (thread %s, new task: %s)", thread_id, new_task)
        
        # Add task completion callback to track task status
        def task_done_callback(task):
            if task.cancelled():
                logger.error(f"❌ Workflow task was CANCELLED: {task}")
            elif task.exception():
                logger.error(f"❌ Workflow task failed with exception: {task.exception()}")
            else:
                logger.info(f"✅ Workflow task completed successfully: {task}")
        
        new_task.add_done_callback(task_done_callback)
    except Exception as exc:
        logger.error("Failed to relaunch workflow after time‑travel: %s", exc)

async def async_monitor_with_time_travel(
    *,
    graph_id: str,
    config: Dict[str, Any],
    node_id: str,
    full_state: Any,
    prompt_template: str | None = None,
    threshold: float = 0.9,
    monitor_config: MonitorConfig = None,
) -> None:
    """Run the CoCo monitor; if an error is confirmed, trigger time‑travel.

    Parameters
    ----------
    graph_id
        The id you used when registering the compiled LangGraph.
    config
        The current RunnableConfig (must contain `thread_id`).
    node_id
        The name of the node just executed.
    full_state
        Complete state data to extract template variables from.
    prompt_template
        Format string template with {key} placeholders for state variables.
    threshold
        Minimum confidence required to act on the error report.
    monitor_config
        Configuration for the monitoring model (model, api_key, base_url, temperature, top_p).
        Must be a MonitorConfig object.
    """

    # Skip if another correction is already underway
    if _get_workflow_control().get("error_detected"):
        logger.debug("Skip monitor - another correction in progress")
        return

    # 1️⃣  Detect error ---------------------------------------------------------
    engine = ErrorDetectionEngine(monitor_config)
    
    try:
        report = await engine.detect_error(
            node_id=node_id,
            full_state=full_state,
            prompt_template=prompt_template,
        )
    except asyncio.CancelledError:
        print(f"🚫 CoCo: Monitoring cancelled ({node_id})")
        return
    except Exception as e:
        print(f"❌ CoCo: Monitoring error - {str(e)}")
        return

    # Check if task was cancelled
    if asyncio.current_task().cancelled():
        logger.debug(f"🚫 CoCo monitoring for {node_id} was cancelled before processing results")
        return

    if not (report["has_error"] and report["confidence"] >= threshold):
        print(f"✅ CoCo: Monitoring completed, no errors found (confidence: {report['confidence']:.2f})")
        return

    # Check again if another correction is in progress (might have started during LLM call)
    if _get_workflow_control().get("error_detected"):
        logger.debug(f"Skip monitor for {node_id} – another correction started during error detection")
        return

    # 2️⃣  Error confirmed – prepare correction ---------------------------------
    print(f"🚨 CoCo: Error detected - {report['error_description']}")
    print(f"🤖 CoCo: Monitor response - {report.get('correction_hint', 'No correction suggestion')}")
    # Atomic operation: only the first monitoring task that detects error can set error status
    if not set_error_detected(True):
        logger.debug(f"Skip time travel for {node_id} – another correction already in progress")
        return

    correction_patch = _build_correction_patch(report, full_state, full_state)

    # 3️⃣  Invoke direct time‑travel -------------------------------------------
    await direct_time_travel(
        graph_id=graph_id,
        config=config,
        node_id=node_id,
        correction_data=correction_patch,
    )

    # Log time travel statistics
    await _log_time_travel_event(node_id, report, correction_patch)

    print(f"🔄 CoCo: Initiating time travel correction for node {node_id}")

async def _log_time_travel_event(node_id: str, report: Dict[str, Any], correction_patch: Dict[str, Any]):
    """Log time travel correction events"""
    try:
        import os
        import json
        from datetime import datetime
        
        # Create log directory
        log_dir = "../results/monitor_logs"
        os.makedirs(log_dir, exist_ok=True)
        
        # Build time travel event record
        travel_event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "time_travel_triggered",
            "node_id": node_id,
            "error_type": report.get("error_type", ""),
            "confidence": report.get("confidence", 0),
            "error_description": report.get("error_description", ""),
            "correction_hint": report.get("correction_hint", ""),
            "correction_patch_keys": list(correction_patch.keys()) if correction_patch else []
        }
        
        # Save to time travel log file
        date_str = datetime.now().strftime("%Y%m%d")
        log_file = f"{log_dir}/time_travel_events_{date_str}.jsonl"
        
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(travel_event, ensure_ascii=False) + "\n")
            
    except Exception as e:
        logger.error(f"Failed to log time travel event: {e}")

def _build_correction_patch(
    report: Dict[str, Any],
    input_state: Any,
    output_result: Any,
) -> Dict[str, Any]:
    """Flatten report + context into a single dict to write into state."""
    # Save original output_result object, let enhance_prompt_with_correction extract based on output_key
    # This way it can support user-specified arbitrary output fields
    failed_output = output_result

    return {
        "error_type": report["error_type"],
        "error_description": report["error_description"],
        "correction_hint": report["correction_hint"],
        "confidence": report["confidence"],
        "original_input": input_state if isinstance(input_state, (dict, str)) else str(input_state),
        "failed_output": failed_output,
        "timestamp": report["timestamp"],
    }
