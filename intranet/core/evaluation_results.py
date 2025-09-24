"""
Evaluation Results Structure
============================

This module provides comprehensive result structures for RAG evaluation
with detailed statistics, analysis, and JSON serialization support.
"""

import json
import uuid
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

from .semantic_validator import ValidationResult
from .rag_http_client import RAGResponse
from .qa_loader import QAPair


@dataclass
class QuestionEvaluation:
    """Detailed evaluation result for a single question."""

    # Question metadata
    question_id: int
    question: str
    expected_answer: str
    expected_sources: List[str]

    # RAG system response
    generated_answer: str
    generated_sources: List[Dict[str, Any]]
    response_time: float
    conversation_id: Optional[str] = None

    # Validation results
    is_correct: bool = False
    confidence_score: float = 0.0
    reasoning: str = ""

    # Detailed metrics
    semantic_similarity: Optional[float] = None
    factual_accuracy: Optional[float] = None
    completeness: Optional[float] = None
    relevance: Optional[float] = None

    # Error information
    rag_error: Optional[str] = None
    validation_error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "question_id": self.question_id,
            "question": self.question,
            "expected_answer": self.expected_answer,
            "expected_sources": self.expected_sources,
            "generated_answer": self.generated_answer,
            "generated_sources": self.generated_sources,
            "response_time": self.response_time,
            "conversation_id": self.conversation_id,
            "is_correct": self.is_correct,
            "confidence_score": self.confidence_score,
            "reasoning": self.reasoning,
            "semantic_similarity": self.semantic_similarity,
            "factual_accuracy": self.factual_accuracy,
            "completeness": self.completeness,
            "relevance": self.relevance,
            "rag_error": self.rag_error,
            "validation_error": self.validation_error,
        }

    @classmethod
    def from_qa_and_responses(
        cls,
        qa_pair: QAPair,
        rag_response: RAGResponse,
        validation_result: ValidationResult,
    ) -> "QuestionEvaluation":
        """Create QuestionEvaluation from QA pair and responses."""
        return cls(
            question_id=qa_pair.id,
            question=qa_pair.question,
            expected_answer=qa_pair.answer,
            expected_sources=qa_pair.sources,
            generated_answer=rag_response.answer,
            generated_sources=rag_response.sources,
            response_time=rag_response.response_time,
            conversation_id=rag_response.conversation_id,
            is_correct=validation_result.is_correct,
            confidence_score=validation_result.confidence_score,
            reasoning=validation_result.reasoning,
            semantic_similarity=validation_result.semantic_similarity,
            factual_accuracy=validation_result.factual_accuracy,
            completeness=validation_result.completeness,
            relevance=validation_result.relevance,
            rag_error=rag_response.error,
            validation_error=validation_result.error_message,
        )


@dataclass
class PerformanceMetrics:
    """Performance metrics for the evaluation."""

    # Overall metrics
    total_questions: int = 0
    successful_questions: int = 0
    failed_questions: int = 0
    success_rate: float = 0.0

    # Accuracy metrics
    correct_answers: int = 0
    incorrect_answers: int = 0
    accuracy_rate: float = 0.0

    # Confidence metrics
    average_confidence: float = 0.0
    median_confidence: float = 0.0
    min_confidence: float = 0.0
    max_confidence: float = 0.0

    # Detailed metrics averages
    average_semantic_similarity: Optional[float] = None
    average_factual_accuracy: Optional[float] = None
    average_completeness: Optional[float] = None
    average_relevance: Optional[float] = None

    # Response time metrics
    total_response_time: float = 0.0  # aggregated; may be overridden by wall_clock_time
    average_response_time: float = 0.0
    median_response_time: float = 0.0
    min_response_time: float = 0.0
    max_response_time: float = 0.0
    wall_clock_time: float = 0.0  # actual elapsed time of the evaluation run

    # Distribution metrics
    score_distribution: Dict[str, int] = field(default_factory=dict)
    performance_categories: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "total_questions": self.total_questions,
            "successful_questions": self.successful_questions,
            "failed_questions": self.failed_questions,
            "success_rate": self.success_rate,
            "correct_answers": self.correct_answers,
            "incorrect_answers": self.incorrect_answers,
            "accuracy_rate": self.accuracy_rate,
            "average_confidence": self.average_confidence,
            "median_confidence": self.median_confidence,
            "min_confidence": self.min_confidence,
            "max_confidence": self.max_confidence,
            "average_semantic_similarity": self.average_semantic_similarity,
            "average_factual_accuracy": self.average_factual_accuracy,
            "average_completeness": self.average_completeness,
            "average_relevance": self.average_relevance,
            "total_response_time": self.total_response_time,
            "average_response_time": self.average_response_time,
            "median_response_time": self.median_response_time,
            "min_response_time": self.min_response_time,
            "max_response_time": self.max_response_time,
            "wall_clock_time": self.wall_clock_time,
            "score_distribution": self.score_distribution,
            "performance_categories": self.performance_categories,
        }


@dataclass
class ErrorAnalysis:
    """Analysis of errors encountered during evaluation."""

    rag_errors: List[Dict[str, Any]] = field(default_factory=list)
    validation_errors: List[Dict[str, Any]] = field(default_factory=list)
    common_error_patterns: Dict[str, int] = field(default_factory=dict)
    error_rate: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "rag_errors": self.rag_errors,
            "validation_errors": self.validation_errors,
            "common_error_patterns": self.common_error_patterns,
            "error_rate": self.error_rate,
        }


@dataclass
class EvaluationSummary:
    """High-level evaluation summary with key insights."""

    overall_grade: str = "F"  # A, B, C, D, F
    grade_explanation: str = ""

    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    key_insights: List[str] = field(default_factory=list)
    confidence_in_results: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "overall_grade": self.overall_grade,
            "grade_explanation": self.grade_explanation,
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "recommendations": self.recommendations,
            "key_insights": self.key_insights,
            "confidence_in_results": self.confidence_in_results,
        }


@dataclass
class EvaluationResults:
    """Comprehensive evaluation results with all analysis and metrics."""

    # Metadata
    evaluation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    evaluation_type: str = "semantic_validation"

    # Configuration
    config: Dict[str, Any] = field(default_factory=dict)
    qa_source: Optional[str] = None
    api_endpoint: Optional[str] = None

    # Results
    question_evaluations: List[QuestionEvaluation] = field(default_factory=list)
    performance_metrics: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    error_analysis: ErrorAnalysis = field(default_factory=ErrorAnalysis)
    summary: EvaluationSummary = field(default_factory=EvaluationSummary)

    # Raw data for further analysis
    validation_examples_used: List[Dict[str, Any]] = field(default_factory=list)
    system_info: Dict[str, Any] = field(default_factory=dict)

    def add_question_evaluation(self, question_eval: QuestionEvaluation) -> None:
        """Add a question evaluation result."""
        self.question_evaluations.append(question_eval)
        self._update_metrics()

    def _update_metrics(self) -> None:
        """Update performance metrics based on current question evaluations."""
        if not self.question_evaluations:
            return

        # Basic counts
        self.performance_metrics.total_questions = len(self.question_evaluations)
        self.performance_metrics.successful_questions = sum(
            1
            for q in self.question_evaluations
            if not q.rag_error and not q.validation_error
        )
        self.performance_metrics.failed_questions = (
            self.performance_metrics.total_questions
            - self.performance_metrics.successful_questions
        )
        self.performance_metrics.success_rate = (
            self.performance_metrics.successful_questions
            / self.performance_metrics.total_questions
        )

        # Accuracy metrics
        self.performance_metrics.correct_answers = sum(
            1 for q in self.question_evaluations if q.is_correct
        )
        self.performance_metrics.incorrect_answers = (
            self.performance_metrics.total_questions
            - self.performance_metrics.correct_answers
        )
        self.performance_metrics.accuracy_rate = (
            self.performance_metrics.correct_answers
            / self.performance_metrics.total_questions
        )

        # Confidence metrics
        confidence_scores = [q.confidence_score for q in self.question_evaluations]
        if confidence_scores:
            self.performance_metrics.average_confidence = sum(confidence_scores) / len(
                confidence_scores,
            )
            sorted_scores = sorted(confidence_scores)
            self.performance_metrics.median_confidence = sorted_scores[
                len(sorted_scores) // 2
            ]
            self.performance_metrics.min_confidence = min(confidence_scores)
            self.performance_metrics.max_confidence = max(confidence_scores)

        # Detailed metrics
        def safe_average(values):
            valid_values = [v for v in values if v is not None]
            return sum(valid_values) / len(valid_values) if valid_values else None

        self.performance_metrics.average_semantic_similarity = safe_average(
            [q.semantic_similarity for q in self.question_evaluations],
        )
        self.performance_metrics.average_factual_accuracy = safe_average(
            [q.factual_accuracy for q in self.question_evaluations],
        )
        self.performance_metrics.average_completeness = safe_average(
            [q.completeness for q in self.question_evaluations],
        )
        self.performance_metrics.average_relevance = safe_average(
            [q.relevance for q in self.question_evaluations],
        )

        # Response time metrics
        response_times = [q.response_time for q in self.question_evaluations]
        if response_times:
            aggregated_time = sum(response_times)
            self.performance_metrics.average_response_time = sum(response_times) / len(
                response_times,
            )
            sorted_times = sorted(response_times)
            self.performance_metrics.median_response_time = sorted_times[
                len(sorted_times) // 2
            ]
            self.performance_metrics.min_response_time = min(response_times)
            self.performance_metrics.max_response_time = max(response_times)
            # Prefer real elapsed (wall clock) if available over naive sum
            self.performance_metrics.total_response_time = (
                self.performance_metrics.wall_clock_time
                if self.performance_metrics.wall_clock_time > 0.0
                else aggregated_time
            )

        # Score distribution
        score_ranges = {
            "excellent (0.8-1.0)": 0,
            "good (0.6-0.8)": 0,
            "fair (0.4-0.6)": 0,
            "poor (0.0-0.4)": 0,
        }

        for q in self.question_evaluations:
            score = q.confidence_score
            if score >= 0.8:
                score_ranges["excellent (0.8-1.0)"] += 1
            elif score >= 0.6:
                score_ranges["good (0.6-0.8)"] += 1
            elif score >= 0.4:
                score_ranges["fair (0.4-0.6)"] += 1
            else:
                score_ranges["poor (0.0-0.4)"] += 1

        self.performance_metrics.score_distribution = score_ranges

        # Performance categories
        self.performance_metrics.performance_categories = {
            "high_accuracy": sum(
                1 for q in self.question_evaluations if q.confidence_score >= 0.8
            ),
            "medium_accuracy": sum(
                1 for q in self.question_evaluations if 0.4 <= q.confidence_score < 0.8
            ),
            "low_accuracy": sum(
                1 for q in self.question_evaluations if q.confidence_score < 0.4
            ),
            "fast_responses": sum(
                1 for q in self.question_evaluations if q.response_time < 2.0
            ),
            "slow_responses": sum(
                1 for q in self.question_evaluations if q.response_time > 5.0
            ),
        }

    def finalize(self) -> None:
        """Finalize the evaluation by generating summary and analysis."""
        self._update_metrics()
        self._analyze_errors()
        self._generate_summary()

    def _analyze_errors(self) -> None:
        """Analyze errors from the evaluation."""
        rag_errors = []
        validation_errors = []
        error_patterns = {}

        for q in self.question_evaluations:
            if q.rag_error:
                rag_errors.append(
                    {
                        "question_id": q.question_id,
                        "question": (
                            q.question[:100] + "..."
                            if len(q.question) > 100
                            else q.question
                        ),
                        "error": q.rag_error,
                    },
                )

                # Extract error patterns
                error_type = (
                    q.rag_error.split(":")[0] if ":" in q.rag_error else q.rag_error
                )
                error_patterns[error_type] = error_patterns.get(error_type, 0) + 1

            if q.validation_error:
                validation_errors.append(
                    {
                        "question_id": q.question_id,
                        "question": (
                            q.question[:100] + "..."
                            if len(q.question) > 100
                            else q.question
                        ),
                        "error": q.validation_error,
                    },
                )

        total_errors = len(rag_errors) + len(validation_errors)
        error_rate = (
            total_errors / len(self.question_evaluations)
            if self.question_evaluations
            else 0.0
        )

        self.error_analysis = ErrorAnalysis(
            rag_errors=rag_errors,
            validation_errors=validation_errors,
            common_error_patterns=error_patterns,
            error_rate=error_rate,
        )

    def _generate_summary(self) -> None:
        """Generate high-level summary and recommendations."""
        metrics = self.performance_metrics

        # Determine overall grade
        accuracy = metrics.accuracy_rate
        if accuracy >= 0.9:
            grade = "A"
        elif accuracy >= 0.8:
            grade = "B"
        elif accuracy >= 0.7:
            grade = "C"
        elif accuracy >= 0.6:
            grade = "D"
        else:
            grade = "F"

        grade_explanation = f"Based on {accuracy:.1%} accuracy rate and {metrics.average_confidence:.2f} average confidence score."

        # Identify strengths and weaknesses
        strengths = []
        weaknesses = []
        recommendations = []

        if metrics.accuracy_rate >= 0.8:
            strengths.append("High accuracy rate indicates good knowledge retrieval")

        if metrics.average_response_time < 3.0:
            strengths.append("Fast response times provide good user experience")

        if metrics.average_confidence >= 0.8:
            strengths.append("High confidence scores suggest reliable answers")

        if metrics.accuracy_rate < 0.7:
            weaknesses.append("Accuracy rate needs improvement")
            recommendations.append("Review document ingestion and search strategies")

        if metrics.average_response_time > 5.0:
            weaknesses.append("Response times are slow")
            recommendations.append("Optimize query processing and document retrieval")

        if self.error_analysis.error_rate > 0.1:
            weaknesses.append("High error rate affects reliability")
            recommendations.append("Investigate and fix common error patterns")

        # Generate insights
        insights = []

        if (
            metrics.performance_categories.get("high_accuracy", 0)
            > metrics.total_questions * 0.5
        ):
            insights.append("System performs well on majority of questions")

        if metrics.score_distribution.get("poor (0.0-0.4)", 0) > 0:
            insights.append(
                f"{metrics.score_distribution['poor (0.0-0.4)']} questions need attention",
            )

        self.summary = EvaluationSummary(
            overall_grade=grade,
            grade_explanation=grade_explanation,
            strengths=strengths,
            weaknesses=weaknesses,
            recommendations=recommendations,
            key_insights=insights,
            confidence_in_results=min(metrics.average_confidence, metrics.success_rate),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "evaluation_id": self.evaluation_id,
            "timestamp": self.timestamp,
            "evaluation_type": self.evaluation_type,
            "config": self.config,
            "qa_source": self.qa_source,
            "api_endpoint": self.api_endpoint,
            "question_evaluations": [q.to_dict() for q in self.question_evaluations],
            "performance_metrics": self.performance_metrics.to_dict(),
            "error_analysis": self.error_analysis.to_dict(),
            "summary": self.summary.to_dict(),
            "validation_examples_used": self.validation_examples_used,
            "system_info": self.system_info,
        }

    def save_to_file(self, output_path: str) -> None:
        """Save evaluation results to a JSON file."""
        output_data = self.to_dict()

        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"💾 Evaluation results saved to {output_path}")

    def get_failed_questions(self) -> List[QuestionEvaluation]:
        """Get questions that failed evaluation."""
        return [
            q
            for q in self.question_evaluations
            if not q.is_correct or q.rag_error or q.validation_error
        ]

    def get_top_questions(self, n: int = 5) -> List[QuestionEvaluation]:
        """Get top N best-performing questions."""
        return sorted(
            self.question_evaluations,
            key=lambda q: q.confidence_score,
            reverse=True,
        )[:n]

    def get_worst_questions(self, n: int = 5) -> List[QuestionEvaluation]:
        """Get top N worst-performing questions."""
        return sorted(
            self.question_evaluations,
            key=lambda q: q.confidence_score,
        )[:n]
