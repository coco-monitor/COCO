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

GraphState = Dict[str, Any]

llm = ChatQwen(
    model="Qwen/Qwen3-8B",
    api_key=os.getenv("MODELSCOPE_API_KEY"),
    base_url=os.getenv("MODELSCOPE_BASE_URL"),
    enable_thinking=False
)

monitor_config = MonitorConfig(
    model = "Qwen/Qwen3-8B",
    api_key = os.getenv("MODELSCOPE_API_KEY"),
    base_url = os.getenv("MODELSCOPE_BASE_URL"),
    enable_thinking=False
)

# Define workflow nodes (just like usual)
@coco_monitor(
    node_id="calculator",
    prompt_template="""Question: 
{question}

Model Response:
{answer}

Evaluate Criteria:
Must directly answer without the process

""",
    confidence_threshold=0.9,
    graph_id="simple_calc",
    model_config=monitor_config
)
async def calculator_node(state: GraphState) -> Dict[str, Any]:
    """Calculator node"""
    try:
        question = state["question"]
        
        # Basic prompt
        base_prompt = f"Please calculate: {question}"
        
        if "coco_correction_calculator" in state:
            prompt = enhance_prompt_with_correction(base_prompt, state, "calculator", "answer")
            logger.info("🔄 Using CoCo correction prompt")
        else:
            prompt = base_prompt
            logger.info("🚀 Using original prompt")
        
        response = await llm.ainvoke(prompt)
        result = response.content if hasattr(response, 'content') else str(response)
        state["answer"] = result.strip()
        
        return state
        
    except Exception as e:
        logger.error(f"Calculator node error: {e}")
        state["answer"] = f"Calculator node error: {e}"
        return state

def create_simple_workflow():
    """Create simple calculation workflow"""
    graph = StateGraph(GraphState)
    
    # Add nodes
    graph.add_node("calculator", calculator_node)
    
    # Set entry and exit points
    graph.set_entry_point("calculator")
    graph.add_edge("calculator", END)
    
    # Add checkpoint support
    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)

async def main():
    """Main function - demonstrate simplified usage"""
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    print("=" * 60)
    print("🎯 CoCo Simplified Interface Demo")
    print("=" * 60)
    
    # Create workflow
    workflow = create_simple_workflow()
    
    # Test input (will trigger CoCo error detection: answer too long)
    test_question = "123 + 456 + 789 = ?"
    initial_state = {"question": test_question}
    
    print(f"📝 Test question: {test_question}")
    print("🎯 Expected: Long answer will trigger CoCo correction")
    print()
    
    try:
        # Integrate CoCo with one line of code!
        result = await coco_execute_workflow(
            workflow=workflow,
            initial_state=initial_state,
            graph_id="simple_calc",  # Optional, used to identify workflow
            max_wait_time=30.0       # Optional, maximum wait time
        )
        
        print("=" * 40)
        print("📊 Execution Results")
        print("=" * 40)
        
        answer = result.get("answer", "No answer obtained")
        print(f"🎯 Final answer: {answer}")
            
    except Exception as e:
        print(f"❌ Execution failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("CoCo interface example starting...")
    asyncio.run(main())