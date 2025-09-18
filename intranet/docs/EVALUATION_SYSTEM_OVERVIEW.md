# RAG System Semantic Evaluation - Complete Implementation

## Overview

I've completely refactored the RAG evaluation system to meet your specific requirements:

1. ✅ **QA Pairs Integration**: Loads questions from `qa_pairs.json`
2. ✅ **HTTP API Testing**: Pings the RAG agent at `http://0.0.0.0:8000/query`
3. ✅ **Semantic Validation**: Uses Unify o4-mini for semantic accuracy validation
4. ✅ **Detailed Reasoning**: LLM provides grounded reasoning for each validation decision
5. ✅ **Extensible Prompts**: Injectable and maintainable prompt system
6. ✅ **Dependency Injection**: Modular architecture with hot-swappable components
7. ✅ **Comprehensive Results**: Detailed JSON output with statistics and per-question analysis

## Architecture

### Core Components

#### 1. **SemanticValidator** (`intranet/core/semantic_validator.py`)
- Uses Unify o4-mini for LLM-based semantic validation
- Extensible prompt system with injectable examples
- Detailed scoring: semantic similarity, factual accuracy, completeness, relevance
- Grounded reasoning for every validation decision

#### 2. **RAGHTTPClient** (`intranet/core/rag_http_client.py`)
- HTTP client specifically for the RAG API at `http://0.0.0.0:8000/query`
- Proper request/response handling with the correct JSON format
- Batch processing with concurrency control
- Comprehensive error handling and retries

#### 3. **QAManager** (`intranet/core/qa_loader.py`)
- Loads QA pairs from `qa_pairs.json` using dependency injection
- Generates validation examples dynamically from loaded QA pairs
- Supports different QA formats through protocol-based loaders
- Hot-swappable QA sources and formats

#### 4. **EvaluationResults** (`intranet/core/evaluation_results.py`)
- Comprehensive result structures with detailed statistics
- Per-question analysis with reasoning
- Overall performance grading (A-F)
- Error analysis and recommendations
- JSON serialization for detailed reporting

### Updated Components

#### 5. **RAG Agent** (`intranet/core/rag_agent.py`)
- Enhanced with Pydantic model validation
- Updated system message to enforce JSON response format
- Comprehensive response structuring with validation

#### 6. **API** (`intranet/core/api.py`)
- Updated to handle new Pydantic response models
- Proper error handling and response validation

## Usage

### Basic Evaluation

```bash
# Run evaluation with default settings
python intranet/scripts/06_run_evaluation.py

# Custom QA pairs file
python intranet/scripts/06_run_evaluation.py --qa-pairs-file custom_qa.json

# Test subset of questions
python intranet/scripts/06_run_evaluation.py --num-questions 10

# Custom API endpoint
python intranet/scripts/06_run_evaluation.py --api-url http://localhost:8000
```

### Advanced Configuration

```bash
# Full configuration
python intranet/scripts/06_run_evaluation.py \
  --qa-pairs-file intranet/data/qa_pairs.json \
  --num-questions 20 \
  --api-url http://0.0.0.0:8000 \
  --max-concurrent 3 \
  --timeout 120 \
  --validator-model gpt-4o-mini@openai \
  --output-dir results \
  --examples-count 6
```

## Key Features

### 1. **Semantic Validation with Grounded Reasoning**

The system uses o4-mini to validate each response with detailed reasoning:

```json
{
  "is_correct": false,
  "confidence_score": 0.2,
  "reasoning": "The generated answer contains a factual error. It states 10 working days when the correct timeframe is 5 working days. This is a significant factual discrepancy that makes the answer incorrect.",
  "semantic_similarity": 0.7,
  "factual_accuracy": 0.0,
  "completeness": 1.0,
  "relevance": 1.0
}
```

### 2. **Extensible Prompt System**

Validation prompts are injectable and examples are auto-generated from QA pairs:

```python
# Custom validator with different model
validator = SemanticValidator(
    model_name="gpt-4@openai",
    examples=custom_examples
)

# Update examples dynamically
validator.update_examples(new_qa_examples)
```

### 3. **Hot-Swappable QA Sources**

```python
# Different QA loaders
csv_loader = CSVQALoader(required_fields=["question", "answer"])
qa_manager = QAManager(loader=csv_loader)

# Or load different formats
qa_manager.load_qa_pairs("custom_format.json")
```

### 4. **Comprehensive Results**

Output includes detailed analysis:

```json
{
  "evaluation_id": "uuid",
  "timestamp": "2024-01-01T12:00:00",
  "performance_metrics": {
    "accuracy_rate": 0.85,
    "average_confidence": 0.78,
    "score_distribution": {
      "excellent (0.8-1.0)": 12,
      "good (0.6-0.8)": 5,
      "fair (0.4-0.6)": 2,
      "poor (0.0-0.4)": 1
    }
  },
  "summary": {
    "overall_grade": "B",
    "strengths": ["High accuracy rate", "Fast responses"],
    "recommendations": ["Review low-scoring questions"]
  },
  "question_evaluations": [...]
}
```

### 5. **Detailed Per-Question Analysis**

Each question gets comprehensive evaluation:

```json
{
  "question_id": 1,
  "question": "What is the speed limit for Class 2 scooters?",
  "expected_answer": "Class 2 scooters are limited to 4 mph.",
  "generated_answer": "The speed limit is 4 mph for Class 2 scooters.",
  "is_correct": true,
  "confidence_score": 0.95,
  "reasoning": "Perfect semantic match with correct factual information",
  "semantic_similarity": 0.95,
  "factual_accuracy": 1.0,
  "completeness": 1.0,
  "relevance": 1.0,
  "response_time": 1.2
}
```

## Extensibility Examples

### Custom QA Loader

```python
class DatabaseQALoader:
    def load(self, connection_string: str) -> List[QAPair]:
        # Load from database
        pass

qa_manager = QAManager(loader=DatabaseQALoader())
```

### Custom Validation Prompts

```python
custom_prompt = """
Your validation prompt with {examples} and {question} placeholders...
"""

validator = SemanticValidator(prompt_template=custom_prompt)
```

### Different Validation Models

```python
# Use different models for validation
claude_validator = SemanticValidator("claude-3-sonnet@anthropic")
gpt4_validator = SemanticValidator("gpt-4@openai")
```

## Output Files

The system generates comprehensive JSON files:

- `semantic_evaluation_YYYYMMDD_HHMMSS.json` - Complete results
- Detailed statistics and per-question analysis
- Error analysis and recommendations
- Validation examples used
- System information and configuration

## Benefits

1. **Accurate Validation**: LLM-based semantic comparison instead of simple string matching
2. **Detailed Insights**: Understand exactly why responses are correct/incorrect
3. **Extensible Design**: Easy to swap QA sources, models, and validation logic
4. **Production Ready**: Comprehensive error handling and logging
5. **Actionable Results**: Clear recommendations for improvement

This implementation fully meets your requirements while providing a robust, extensible foundation for RAG system evaluation.
