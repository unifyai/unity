"""
Semantic Validator for RAG System Evaluation
============================================

This module provides semantic validation of RAG responses using LLM-based comparison
with extensible prompts and detailed reasoning for validation decisions.
"""

import json
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import unify


@dataclass
class ValidationResult:
    """Result of semantic validation for a single question-answer pair."""

    question: str
    expected_answer: str
    generated_answer: str
    sources_expected: List[str]
    sources_generated: List[Dict[str, Any]]

    is_correct: bool
    confidence_score: float  # 0.0 to 1.0
    reasoning: str

    # Additional metrics
    semantic_similarity: Optional[float] = None
    factual_accuracy: Optional[float] = None
    completeness: Optional[float] = None
    relevance: Optional[float] = None

    # Response metadata
    response_time: Optional[float] = None
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "question": self.question,
            "expected_answer": self.expected_answer,
            "generated_answer": self.generated_answer,
            "sources_expected": self.sources_expected,
            "sources_generated": self.sources_generated,
            "is_correct": self.is_correct,
            "confidence_score": self.confidence_score,
            "reasoning": self.reasoning,
            "semantic_similarity": self.semantic_similarity,
            "factual_accuracy": self.factual_accuracy,
            "completeness": self.completeness,
            "relevance": self.relevance,
            "response_time": self.response_time,
            "error_message": self.error_message,
        }


class SemanticValidator:
    """
    LLM-based semantic validator for RAG responses.

    Uses configurable prompts and detailed analysis to validate whether
    generated answers are semantically equivalent to expected answers.
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini@openai",
        prompt_template: Optional[str] = None,
        examples: Optional[List[Dict[str, Any]]] = None,
    ):
        """
        Initialize the semantic validator.

        Args:
            model_name: Unify model name to use for validation
            prompt_template: Custom prompt template (optional)
            examples: Example QA pairs to include in prompt (optional)
        """
        self.model_name = model_name
        self.client = unify.AsyncUnify(
            model_name,
            cache=json.loads(os.environ.get("UNIFY_CACHE", "true")),
            traced=json.loads(os.environ.get("UNIFY_TRACED", "true")),
        )

        # Set up extensible prompt system
        self.prompt_template = prompt_template or self._get_default_prompt_template()
        self.examples = examples or self._get_default_examples()

        print(f"🔍 Semantic validator initialized with {model_name}")

    def _get_default_prompt_template(self) -> str:
        """Get the default validation prompt template."""
        return """You are a semantic validation expert for a RAG (Retrieval-Augmented Generation) system evaluation.

Your task is to determine whether a generated answer is semantically correct when compared to the expected answer for a given question.

EVALUATION CRITERIA:
1. **Semantic Similarity**: Does the generated answer convey the same meaning as the expected answer?
2. **Factual Accuracy**: Are the facts, numbers, and specific details correct?
3. **Completeness**: Does the generated answer cover the key points from the expected answer?
4. **Relevance**: Does the generated answer directly address the question asked?

CRITICAL POLICY ON EXTRA INFORMATION:
• Extra information beyond the expected answer that is accurate and relevant MUST NOT reduce correctness or factual accuracy.
• Mark the generated answer as correct if it includes all key facts from the expected answer and those facts are accurate — even if the generated answer contains additional, non‑contradictory context.
• Only penalize when extra information contradicts the expected answer or introduces incorrect claims. Do not penalize for well‑scoped additional context, notes, or examples.
• Completeness is measured relative to the expected answer: missing expected key points reduces completeness; adding extra correct details does not.

SCORING GUIDELINES:
- Score 1.0: Perfect semantic match, all facts correct, complete coverage
- Score 0.8-0.9: Very good match, minor differences in wording or style
- Score 0.6-0.7: Good match, captures main points but may miss some details
- Score 0.4-0.5: Partial match, some correct information but significant gaps
- Score 0.2-0.3: Poor match, major factual errors or missing key information
- Score 0.0-0.1: Incorrect or completely irrelevant answer

EXAMPLES:
{examples}

EVALUATION TASK:
Question: {question}
Expected Answer: {expected_answer}
Generated Answer: {generated_answer}
Expected Sources: {expected_sources}

Please evaluate the generated answer and respond with a JSON object in this exact format:
{{
    "is_correct": true/false,
    "confidence_score": 0.0-1.0,
    "reasoning": "Detailed explanation of your evaluation decision",
    "semantic_similarity": 0.0-1.0,
    "factual_accuracy": 0.0-1.0,
    "completeness": 0.0-1.0,
    "relevance": 0.0-1.0
}}

CRITICAL: Your reasoning must be detailed and specific. Explain:
- What aspects are correct or incorrect
- Any factual discrepancies
- Missing information
- Why you assigned the specific scores
- How the generated answer compares to the expected answer
- Explicitly state whether any extra information is present and whether it is accurate and non‑contradictory (this should not lower correctness).

Respond only with the JSON object, no additional text."""

    def _get_default_examples(self) -> List[Dict[str, Any]]:
        """Get default example QA pairs for the prompt."""
        return [
            {
                "question": "What is the maximum speed limit for Class 2 mobility scooters?",
                "expected_answer": "Class 2 scooters are limited to 4 mph.",
                "generated_answer": "The speed limit for Class 2 mobility scooters is 4 mph.",
                "evaluation": {
                    "is_correct": True,
                    "confidence_score": 0.95,
                    "reasoning": "The generated answer correctly states the 4 mph speed limit for Class 2 scooters, conveying the same factual information as the expected answer with slightly different wording.",
                    "semantic_similarity": 0.95,
                    "factual_accuracy": 1.0,
                    "completeness": 1.0,
                    "relevance": 1.0,
                },
            },
            {
                "question": "What is the maximum speed limit for Class 2 mobility scooters?",
                "expected_answer": "Class 2 scooters are limited to 4 mph.",
                "generated_answer": "Class 2 scooters are limited to 4 mph. They are typically intended for footpaths.",
                "evaluation": {
                    "is_correct": True,
                    "confidence_score": 0.95,
                    "reasoning": "Core fact (4 mph) is correct and present. The extra note about footpaths is accurate and non‑contradictory, so it does not reduce correctness.",
                    "semantic_similarity": 0.95,
                    "factual_accuracy": 1.0,
                    "completeness": 1.0,
                    "relevance": 1.0,
                },
            },
            {
                "question": "How many working days must Midland Heart acknowledge a Stage 1 complaint?",
                "expected_answer": "Within 5 working days of receipt.",
                "generated_answer": "Midland Heart must acknowledge Stage 1 complaints within 10 working days.",
                "evaluation": {
                    "is_correct": False,
                    "confidence_score": 0.2,
                    "reasoning": "The generated answer contains a factual error. It states 10 working days when the correct timeframe is 5 working days. This is a significant factual discrepancy that makes the answer incorrect.",
                    "semantic_similarity": 0.7,
                    "factual_accuracy": 0.0,
                    "completeness": 1.0,
                    "relevance": 1.0,
                },
            },
            {
                "question": "What is the contact route for IT issues?",
                "expected_answer": "Raise a ticket via the IT Service Desk portal.",
                "generated_answer": "Raise a ticket via the IT Service Desk portal. You can also call after hours for critical incidents.",
                "evaluation": {
                    "is_correct": True,
                    "confidence_score": 0.9,
                    "reasoning": "The answer includes the exact expected action and adds accurate, non‑contradictory context about after‑hours support; extra information is allowed and should not reduce correctness.",
                    "semantic_similarity": 0.9,
                    "factual_accuracy": 1.0,
                    "completeness": 1.0,
                    "relevance": 1.0,
                },
            },
        ]

    def update_examples(self, new_examples: List[Dict[str, Any]]) -> None:
        """Update the examples used in the validation prompt."""
        self.examples = new_examples
        print(f"📝 Updated validation examples with {len(new_examples)} entries")

    def update_prompt_template(self, new_template: str) -> None:
        """Update the prompt template used for validation."""
        self.prompt_template = new_template
        print("📝 Updated validation prompt template")

    def _format_examples(self) -> str:
        """Format examples for inclusion in the prompt."""
        formatted_examples = []

        for i, example in enumerate(self.examples, 1):
            example_text = f"""
Example {i}:
Question: {example['question']}
Expected: {example['expected_answer']}
Generated: {example['generated_answer']}
Evaluation: {json.dumps(example['evaluation'], indent=2)}
"""
            formatted_examples.append(example_text)

        return "\n".join(formatted_examples)

    async def validate_response(
        self,
        question: str,
        expected_answer: str,
        generated_answer: str,
        expected_sources: List[str],
        generated_sources: List[Dict[str, Any]] = None,
        response_time: Optional[float] = None,
    ) -> ValidationResult:
        """
        Validate a generated response against the expected answer.

        Args:
            question: The original question
            expected_answer: The ground truth answer
            generated_answer: The RAG system's answer
            expected_sources: Expected source documents
            generated_sources: RAG system's source citations
            response_time: Time taken to generate the response

        Returns:
            ValidationResult with detailed analysis
        """
        try:
            # Format the validation prompt
            formatted_prompt = self.prompt_template.format(
                examples=self._format_examples(),
                question=question,
                expected_answer=expected_answer,
                generated_answer=generated_answer,
                expected_sources=(
                    ", ".join(expected_sources)
                    if expected_sources
                    else "None specified"
                ),
            )

            # Get LLM evaluation
            self.client.reset_messages()
            response = await self.client.generate(
                messages=[{"role": "user", "content": formatted_prompt}],
            )

            # Parse the JSON response
            try:
                eval_result = json.loads(response.strip())
            except json.JSONDecodeError:
                # If JSON parsing fails, try to extract JSON from response
                import re

                json_match = re.search(r"\{.*\}", response, re.DOTALL)
                if json_match:
                    eval_result = json.loads(json_match.group())
                else:
                    raise ValueError(f"Could not parse validation response: {response}")

            # Create validation result
            result = ValidationResult(
                question=question,
                expected_answer=expected_answer,
                generated_answer=generated_answer,
                sources_expected=expected_sources,
                sources_generated=generated_sources or [],
                is_correct=eval_result.get("is_correct", False),
                confidence_score=eval_result.get("confidence_score", 0.0),
                reasoning=eval_result.get("reasoning", "No reasoning provided"),
                semantic_similarity=eval_result.get("semantic_similarity"),
                factual_accuracy=eval_result.get("factual_accuracy"),
                completeness=eval_result.get("completeness"),
                relevance=eval_result.get("relevance"),
                response_time=response_time,
            )

            return result

        except Exception as e:
            print(f"❌ Validation error for question: {question[:50]}... - {e}")

            # Return error result
            return ValidationResult(
                question=question,
                expected_answer=expected_answer,
                generated_answer=generated_answer,
                sources_expected=expected_sources,
                sources_generated=generated_sources or [],
                is_correct=False,
                confidence_score=0.0,
                reasoning=f"Validation failed due to error: {str(e)}",
                response_time=response_time,
                error_message=str(e),
            )

    async def validate_batch(
        self,
        qa_pairs: List[Dict[str, Any]],
        generated_responses: List[Dict[str, Any]],
        max_concurrent: int = 3,
    ) -> List[ValidationResult]:
        """
        Validate a batch of responses with concurrency control.

        Args:
            qa_pairs: List of ground truth QA pairs
            generated_responses: List of RAG system responses
            max_concurrent: Maximum concurrent validations

        Returns:
            List of ValidationResult objects
        """
        import asyncio

        async def validate_single(
            qa_pair: Dict[str, Any],
            response: Dict[str, Any],
        ) -> ValidationResult:
            return await self.validate_response(
                question=qa_pair["question"],
                expected_answer=qa_pair["answer"],
                generated_answer=response.get("answer", ""),
                expected_sources=qa_pair.get("sources", []),
                generated_sources=response.get("sources", []),
                response_time=response.get("response_time"),
            )

        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)

        async def validate_with_semaphore(
            qa_pair: Dict[str, Any],
            response: Dict[str, Any],
        ) -> ValidationResult:
            async with semaphore:
                return await validate_single(qa_pair, response)

        # Run validations with concurrency control
        tasks = [
            validate_with_semaphore(qa_pair, response)
            for qa_pair, response in zip(qa_pairs, generated_responses)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions
        validated_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"❌ Validation {i+1} failed: {result}")
                # Create error result
                qa_pair = qa_pairs[i]
                response = generated_responses[i]
                error_result = ValidationResult(
                    question=qa_pair["question"],
                    expected_answer=qa_pair["answer"],
                    generated_answer=response.get("answer", ""),
                    sources_expected=qa_pair.get("sources", []),
                    sources_generated=response.get("sources", []),
                    is_correct=False,
                    confidence_score=0.0,
                    reasoning=f"Validation failed: {str(result)}",
                    error_message=str(result),
                )
                validated_results.append(error_result)
            else:
                validated_results.append(result)

        return validated_results
