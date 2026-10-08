## Math node -> python code node -> summary node

import os
import asyncio
import logging
from typing import Dict, Any
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_qwq import ChatQwen

from src.decorator import coco_monitor, coco_execute_workflow, enhance_prompt_with_correction
from src.config import MonitorConfig

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

# State definition for the PI calculation workflow
PICalculationState = Dict[str, Any]


# LLM configuration
llm = ChatQwen(
    model="Qwen/Qwen2.5-7B-Instruct",
    api_key=os.getenv("MODELSCOPE_API_KEY"),
    base_url=os.getenv("MODELSCOPE_BASE_URL"),
    temperature=0.0
)

# CoCo monitoring configuration
monitor_config = MonitorConfig(
    model="Qwen/Qwen3-8B",
    api_key=os.getenv("MODELSCOPE_API_KEY"),
    base_url=os.getenv("MODELSCOPE_BASE_URL"),
    temperature=0.0,
    enable_thinking=False
)

# Math LLM Node - Provide mathematical formula for calculating PI
async def math_formula_node(state: PICalculationState) -> PICalculationState:
    """Math LLM provides a formula to calculate PI"""
    try:
        base_prompt = """You are a mathematics expert. Provide a clear, concise mathematical formula for calculating PI.

Requirements:
1. Choose one well-known method (Leibniz series, Gregory-Leibniz, Machin's formula, etc.)
2. Give the formula and recommended iterations (e.g., "Use 1000+ iterations for 4-6 decimal places")
3. Keep response brief and focused

Provide only the formula and iteration guidance."""
        
        response = await llm.ainvoke(base_prompt)
        formula = response.content if hasattr(response, 'content') else str(response)
        
        state["formula"] = formula.strip()
        state["step"] = "formula_provided"
        
        print(f"\n📐 Math Formula: {formula[:100]}...")
        print()
        return state
        
    except Exception as e:
        logger.error(f"Math formula node error: {e}")
        state["formula"] = f"Error in formula generation: {e}"
        return state

# Coder LLM Node - Implement the formula in code (monitored to prevent math.pi usage)
@coco_monitor(
    node_id="code_implementation", 
    prompt_template="""Task: Implement PI calculation code

Generated Code:
{code}

Evaluation Criteria:
1. Code must NOT use math.pi or any built-in library
2. Must implement mathematical calculations from scratch
3. Code should be executable Python code
4. Should handle convergence/iterations properly
""",
    confidence_threshold=0.8,
    graph_id="pi_calculation",
    model_config=monitor_config
)
async def code_implementation_node(state: PICalculationState) -> PICalculationState:
    """Coder LLM implements the PI calculation code (monitored to prevent math.pi usage)"""
    try:
        # Ensure formula exists in state
        if "formula" not in state or not state["formula"]:
            raise ValueError("No formula available in state")
        
        formula = state["formula"]
        
        base_prompt = f"""You are a Python programmer. Based on the following mathematical formula for calculating PI, write Python code to implement it.

Mathematical Formula:
{formula}

Requirements:
1. Write complete, executable Python code
2. DO NOT use math.pi, numpy.pi, or any built-in PI constants
3. Implement the formula from scratch
4. Use sufficient iterations for at least 4-6 decimal places accuracy
5. At the end, print the calculated PI value
6. Add comments explaining the implementation
7. Make sure the code is syntactically correct

Example structure:
```python
# Implementation of [formula name]
def calculate_pi():
    # Your implementation here
    return pi_value

# Calculate and output the result
result = calculate_pi()
print(result)
```

Provide only the Python code, no explanations outside the code comments."""
        
        # Check for CoCo correction
        if "coco_correction_code_implementation" in state:
            # Store original answer if it exists
            if "calculated_pi" in state:
                state["origin_answer"] = state["calculated_pi"]
            prompt = enhance_prompt_with_correction(base_prompt, state, "code_implementation", "code")
            print("\n🔄 Using CoCo correction for code implementation")
        else:
            prompt = base_prompt
        
        response = await llm.ainvoke(prompt)
        code = response.content if hasattr(response, 'content') else str(response)
        
        # Clean the code - remove markdown formatting if present
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0]
        elif "```" in code:
            code = code.split("```")[1].split("```")[0]
        
        state["code"] = code.strip()
        state["step"] = "code_generated"
        
        # Execute the code and capture output
        try:
            import io
            import sys
            
            # Capture stdout
            old_stdout = sys.stdout
            captured_output = io.StringIO()
            sys.stdout = captured_output
            
            # Execute the code
            exec_globals = {"__builtins__": {"range": range, "abs": abs, "float": float, "int": int, "len": len, "print": print}}
            exec(code, exec_globals)
            
            # Restore stdout and get the output
            sys.stdout = old_stdout
            output = captured_output.getvalue().strip()
            
            # Simply use the captured output as PI value
            if output:
                state["calculated_pi"] = float(output)
                print(f"\n💻 Calculated PI: {output}")
            else:
                state["calculated_pi"] = "No output captured"
                
        except Exception as e:
            state["calculated_pi"] = f"Execution error: {e}"
            print(f"Execution error: {state['calculated_pi']}")
        
        # Ensure calculated_pi is always set
        if "calculated_pi" not in state:
            state["calculated_pi"] = "Failed to calculate PI"
            print("Failed to calculate PI")
        
        print(f"💻 Generated Code: {len(code)} characters")
        print()
        return state
        
    except Exception as e:
        logger.error(f"Code implementation node error: {e}")
        state["code"] = f"Error in code generation: {e}"
        state["calculated_pi"] = f"Error in code generation: {e}"
        return state

# Summary LLM Node - Summarize the results
async def summary_results_node(state: PICalculationState) -> PICalculationState:
    """Summary LLM summarizes the results of math and code to give final answer"""
    try:
        # Ensure required fields exist in state
        if "formula" not in state or not state["formula"]:
            raise ValueError("No formula available in state")
        if "code" not in state or not state["code"]:
            raise ValueError("No code available in state")
        if "calculated_pi" not in state:
            # If calculated_pi doesn't exist, set a default value
            state["calculated_pi"] = "Calculation failed"
            
        formula = state["formula"]
        code = state["code"]
        calculated_pi = state["calculated_pi"]
        
        base_prompt = f"""Summarize the PI calculation results concisely:

Method: {formula[:100]}...
Code: {len(str(code))} characters generated
Result: {calculated_pi}

Provide a brief summary including:
1. What mathematical method was used
2. The calculated PI value
3. Accuracy assessment (compare to π ≈ 3.14159...)

Keep it short and to the point."""
        
        response = await llm.ainvoke(base_prompt)
        summary = response.content if hasattr(response, 'content') else str(response)
        
        state["summary"] = summary.strip()
        state["step"] = "completed"
        
        print(f"\n📋 Summary: {summary[:100]}...")
        print()  # Add spacing
        return state
        
    except Exception as e:
        logger.error(f"Summary node error: {e}")
        state["summary"] = f"Error in summary generation: {e}"
        # Ensure calculated_pi is set if missing
        if "calculated_pi" not in state:
            state["calculated_pi"] = f"Error in summary generation: {e}"
        return state

def create_pi_calculation_workflow():
    """Create the PI calculation workflow with three nodes"""
    graph = StateGraph(PICalculationState)
    
    # Add nodes
    graph.add_node("math_formula", math_formula_node)
    graph.add_node("code_implementation", code_implementation_node) 
    graph.add_node("summary_results", summary_results_node)
    
    # Define the flow
    graph.set_entry_point("math_formula")
    graph.add_edge("math_formula", "code_implementation")
    graph.add_edge("code_implementation", "summary_results")
    graph.add_edge("summary_results", END)
    
    # Add checkpoint support for CoCo time travel
    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)

async def main():
    """Main function - demonstrate PI calculation with CoCo monitoring"""
    
    # Set up minimal logging - only show warnings and errors
    logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')
    
    print("=" * 70)
    print("🥧 PI Calculation with CoCo Monitoring Demo")
    print("=" * 70)
    print("📋 Workflow: Math Formula → Code Implementation → Summary")
    print("🔍 CoCo will monitor each step for quality and correctness")
    print("⚠️  Special monitoring: Code must NOT use math.pi!")
    print()
    print("-" * 50)
    print("🚀 Starting workflow execution...")
    print("-" * 50)
    
    # Create workflow
    workflow = create_pi_calculation_workflow()
    
    # Initial state
    initial_state: PICalculationState = {
        "task": "Calculate PI using mathematical formula and code implementation",
        "step": "starting"
    }
    
    try:
        # Execute with CoCo monitoring
        result = await coco_execute_workflow(
            workflow=workflow,
            initial_state=initial_state,
            graph_id="pi_calculation",
            max_wait_time=60.0  # Allow more time for complex calculations
        )
        print("\n" + "=" * 50)
        print("📊 PI Calculation Results")
        print("=" * 50)
        
        # Check if result is None
        if result is None:
            print("❌ Error: Workflow execution returned None")
            print("This might be due to CoCo monitoring issues or workflow errors")
            return
        
        # Ensure all required fields are present
        required_fields = ["formula", "calculated_pi", "summary"]
        missing_fields = [field for field in required_fields if field not in result or not result[field]]
        
        if missing_fields:
            print(f"❌ Error: Missing required fields in result: {missing_fields}")
            print(f"Available fields: {list(result.keys())}")
            print("This indicates that some workflow nodes did not complete successfully")
            return
        
        print(f"\n📐 Mathematical Formula:")
        print(f"{result['formula']}")
        
        print(f"\n💻 Calculated PI Value:")
        print(f"{result['calculated_pi']}")
        
        print(f"\n📋 Summary:")
        print(f"{result['summary']}")
        
        # Final accuracy check
        try:
            if "origin_answer" in result:
                print(f"Origin calculated_pi: {result['origin_answer']}")
            calculated_pi_str = result['calculated_pi']
            calculated = float(calculated_pi_str)
            actual_pi = 3.14159265359
            error = abs(calculated - actual_pi)
            accuracy = max(0, 100 * (1 - error / actual_pi))
            
            print(f"\n🎯 Accuracy Analysis:")
            print(f"Calculated: {calculated}")
            print(f"Actual PI:  {actual_pi}")
            print(f"Error:      {error:.6f}")
            print(f"Accuracy:   {accuracy:.2f}%")
            
            if accuracy > 99:
                print("🎉 Excellent accuracy!")
            elif accuracy > 90:
                print("✅ Good accuracy!")
            else:
                print("⚠️ Consider more iterations for better accuracy")
                
        except (ValueError, TypeError) as e:
            print(f"⚠️ Could not perform accuracy analysis: {e}")
            
    except Exception as e:
        print(f"❌ Execution failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Starting PI calculation with CoCo monitoring...")
    asyncio.run(main())