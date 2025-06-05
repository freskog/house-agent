#!/usr/bin/env python3
"""
Test script for evaluating search performance with timing and logging.
"""

import pytest
import asyncio
import time
import logging
from langchain_core.messages import HumanMessage
from agent import make_graph

# Configure logging for detailed timing information
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Test queries of different complexities
TEST_QUERIES = [
    # Simple factual questions
    "What is the capital of France?",
    "How many hours of sleep should a 4 year old get per day?",
    "What is the weather like today?",
    
    # More complex queries
    "What are the latest news about artificial intelligence?",
    "How do I fix a leaky faucet?",
    "What is the best programming language for machine learning?",
    
    # Time-sensitive queries
    "What time is it in Tokyo right now?",
    "When is the next solar eclipse?",
    
    # Research-oriented queries
    "Explain quantum computing in simple terms",
    "What are the benefits of renewable energy?"
]

async def time_query(graph, query: str, query_num: int) -> dict:
    """
    Execute a single query and measure timing.
    
    Returns timing information and response details.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Query {query_num}: {query}")
    logger.info(f"{'='*60}")
    
    # Record start time
    start_time = time.time()
    
    try:
        # Initialize state
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "next_action": "continue",
            "target_domain": None,
            "audio_server": None,
            "current_client": None
        }
        
        # Track timing for different phases
        routing_start = time.time()
        
        # Execute the query
        final_state = None
        step_count = 0
        step_timings = []
        
        async for step in graph.astream(initial_state, {"recursion_limit": 10}):
            step_start = time.time()
            step_count += 1
            
            for node_name, node_state in step.items():
                logger.info(f"Step {step_count} - Node: {node_name}")
                logger.info(f"  Next action: {node_state.get('next_action', 'N/A')}")
                logger.info(f"  Target domain: {node_state.get('target_domain', 'N/A')}")
                
                # Log message count and types
                messages = node_state.get("messages", [])
                logger.info(f"  Messages: {len(messages)} total")
                if messages:
                    last_msg = messages[-1]
                    msg_type = type(last_msg).__name__
                    content_preview = last_msg.content[:100] if hasattr(last_msg, 'content') else str(last_msg)[:100]
                    logger.info(f"  Last message type: {msg_type}")
                    logger.info(f"  Last message preview: {content_preview}...")
                
                final_state = node_state
            
            step_end = time.time()
            step_duration = step_end - step_start
            step_timings.append({
                'step': step_count,
                'duration': step_duration,
                'nodes': list(step.keys())
            })
            logger.info(f"  Step duration: {step_duration:.3f}s")
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        # Extract final response
        final_response = "No response"
        if final_state and "messages" in final_state:
            messages = final_state["messages"]
            if messages:
                last_message = messages[-1]
                if hasattr(last_message, 'content'):
                    final_response = last_message.content
        
        # Log summary
        logger.info(f"\n🎯 QUERY SUMMARY:")
        logger.info(f"   Query: {query}")
        logger.info(f"   Total time: {total_duration:.3f}s")
        logger.info(f"   Steps: {step_count}")
        logger.info(f"   Response length: {len(final_response)} chars")
        logger.info(f"   Response preview: {final_response[:200]}...")
        
        return {
            "query": query,
            "success": True,
            "total_duration": total_duration,
            "step_count": step_count,
            "step_timings": step_timings,
            "response_length": len(final_response),
            "response": final_response,
            "error": None
        }
        
    except Exception as e:
        end_time = time.time()
        total_duration = end_time - start_time
        
        logger.error(f"❌ Query failed: {e}")
        return {
            "query": query,
            "success": False,
            "total_duration": total_duration,
            "step_count": 0,
            "step_timings": [],
            "response_length": 0,
            "response": "",
            "error": str(e)
        }

async def run_performance_test():
    """Run the full performance test suite."""
    logger.info("🚀 Starting search performance test...")
    logger.info(f"Testing {len(TEST_QUERIES)} queries")
    
    # Initialize the agent graph
    async with make_graph() as graph:
        logger.info("✅ Agent graph initialized successfully")
        
        results = []
        total_test_start = time.time()
        
        for i, query in enumerate(TEST_QUERIES, 1):
            result = await time_query(graph, query, i)
            results.append(result)
            
            # Brief pause between queries
            await asyncio.sleep(0.5)
        
        total_test_duration = time.time() - total_test_start
        
        # Performance analysis
        logger.info(f"\n📊 PERFORMANCE ANALYSIS")
        logger.info(f"{'='*60}")
        
        successful_queries = [r for r in results if r["success"]]
        failed_queries = [r for r in results if not r["success"]]
        
        logger.info(f"Total queries: {len(results)}")
        logger.info(f"Successful: {len(successful_queries)}")
        logger.info(f"Failed: {len(failed_queries)}")
        logger.info(f"Success rate: {len(successful_queries)/len(results)*100:.1f}%")
        logger.info(f"Total test duration: {total_test_duration:.2f}s")
        
        if successful_queries:
            durations = [r["total_duration"] for r in successful_queries]
            step_counts = [r["step_count"] for r in successful_queries]
            response_lengths = [r["response_length"] for r in successful_queries]
            
            logger.info(f"\n⏱️  TIMING STATISTICS:")
            logger.info(f"   Average query time: {sum(durations)/len(durations):.3f}s")
            logger.info(f"   Fastest query: {min(durations):.3f}s")
            logger.info(f"   Slowest query: {max(durations):.3f}s")
            logger.info(f"   Average steps: {sum(step_counts)/len(step_counts):.1f}")
            logger.info(f"   Average response length: {sum(response_lengths)/len(response_lengths):.0f} chars")
        
        # Failed queries analysis
        if failed_queries:
            logger.info(f"\n❌ FAILED QUERIES:")
            for result in failed_queries:
                logger.info(f"   - {result['query']}")
                logger.info(f"     Error: {result['error']}")
        
        # Individual query breakdown
        logger.info(f"\n📋 INDIVIDUAL RESULTS:")
        for i, result in enumerate(results, 1):
            status = "✅" if result["success"] else "❌"
            logger.info(f"{status} Query {i}: {result['total_duration']:.3f}s - {result['query'][:50]}...")
        
        return results

@pytest.mark.performance
@pytest.mark.slow
@pytest.mark.asyncio
async def test_search_performance():
    """Pytest wrapper for the performance test."""
    results = await run_performance_test()
    
    # Basic assertions to ensure the test ran
    assert len(results) > 0, "No test results generated"
    
    successful_queries = [r for r in results if r["success"]]
    success_rate = len(successful_queries) / len(results)
    
    # We expect at least 70% success rate
    assert success_rate >= 0.7, f"Success rate too low: {success_rate:.1%}"
    
    # Average response time should be reasonable (under 10 seconds)
    if successful_queries:
        avg_time = sum(r["total_duration"] for r in successful_queries) / len(successful_queries)
        assert avg_time < 10.0, f"Average response time too slow: {avg_time:.2f}s"

if __name__ == "__main__":
    import pytest
    asyncio.run(run_performance_test()) 