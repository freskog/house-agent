# House Agent Tests 🧪

This directory contains the sanctioned test suite for the house agent project.

## **Test Structure**

### **Unit Tests**
- `test_search_node_unit.py` - Unit tests for SearchNode with mocking
  - Tests caller detection, response formatting, tool execution
  - Fast, isolated tests that don't require external APIs

### **Functional Tests** 
- `test_search_node_functional.py` - Functional tests for SearchNode with real API
  - Tests actual search functionality with Tavily API (requires `TAVILY_API_KEY`)
  - End-to-end verification that the simplified SearchNode works correctly

### **Integration Tests**
- `test_house_node_integration.py` - Integration test for house automation
  - Tests the full flow: query → router → house node → Home Assistant
  - Verifies real office lights status query

### **Performance Tests**
- `test_search_performance.py` - Performance monitoring for search operations
  - Tests search speed and timing across different query types
  - Generates performance reports with detailed timing analysis

## **Running Tests**

### **All Tests**
```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=nodes
```

### **Specific Test Categories**
```bash
# Unit tests only (fast, no external deps)
pytest tests/test_*_unit.py -v

# Functional tests (requires API keys)
pytest tests/test_*_functional.py -v

# Integration tests (requires full system)
pytest tests/test_*_integration.py -v

# Performance tests
pytest tests/test_*_performance.py -v
```

### **Individual Tests**
```bash
# Test SearchNode functionality
python tests/test_search_node_functional.py

# Test house automation
python tests/test_house_node_integration.py

# Run performance analysis
python tests/test_search_performance.py
```

## **Environment Setup**

Some tests require environment variables:

```bash
# For search functionality
export TAVILY_API_KEY="your-tavily-api-key"

# For Home Assistant integration  
export OPENAI_API_KEY="your-openai-api-key"
```

## **Test Guidelines**

### **What Belongs Here**
- ✅ Tests that verify current functionality
- ✅ Tests that prevent regressions  
- ✅ Performance monitoring tests
- ✅ Integration tests for key workflows

### **What Doesn't Belong Here**
- ❌ Debug scripts or one-off tests
- ❌ Tests for removed/obsolete functionality
- ❌ Experimental or exploratory code
- ❌ Tests that don't provide value

### **Naming Conventions**
- `test_<component>_unit.py` - Unit tests with mocking
- `test_<component>_functional.py` - Functional tests with real APIs  
- `test_<component>_integration.py` - Full system integration tests
- `test_<component>_performance.py` - Performance and timing tests

## **Maintenance**

These tests should be:
- **Kept up to date** with code changes
- **Run before major releases** to prevent regressions
- **Updated** when functionality changes
- **Removed** if they become obsolete

The goal is a small, focused test suite that provides confidence in the system without maintenance overhead. 