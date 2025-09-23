#!/usr/bin/env python3
"""
RAG System Semantic Evaluation
==============================

This script runs comprehensive semantic evaluation of the RAG system including:
- QA pairs from qa_pairs.json
- HTTP endpoint evaluation with semantic validation
- LLM-based validation using o4-mini for semantic accuracy
- Detailed reasoning and analysis for each question
- Extensible prompt system with dependency injection
- Comprehensive JSON results with statistics and per-question analysis

Usage:
    python scripts/06_run_evaluation.py [OPTIONS]

Options:
    --qa-pairs-file      Path to QA pairs JSON file (default: intranet/data/qa_pairs.json)
    --num-questions      Number of test questions to evaluate (default: all)
    --api-url           API base URL for HTTP evaluation (default: http://0.0.0.0:8000)
    --max-concurrent    Maximum concurrent HTTP requests (default: 5)
    --timeout           HTTP timeout in seconds (default: 120)
    --validator-model   Model for semantic validation (default: gpt-4o-mini@openai)
    --output-dir        Output directory for results (default: intranet/evals)
    --examples-count    Number of validation examples to generate (default: 4)
"""

import asyncio
import sys
import argparse
import time
from pathlib import Path
from datetime import datetime

# Import utilities and setup environment
from utils import initialize_script_environment

# Initialize environment and setup paths
if not initialize_script_environment():
    sys.exit(1)

# Import new evaluation components
from intranet.core.semantic_validator import SemanticValidator
from intranet.core.rag_http_client import RAGHTTPClient
from intranet.core.qa_loader import QAManager, JSONQALoader
from intranet.core.evaluation_results import (
    EvaluationResults,
    QuestionEvaluation,
)


async def run_semantic_evaluation(
    qa_pairs_file="intranet/data/qa_pairs.json",
    num_questions=None,
    api_url="http://0.0.0.0:8000",
    max_concurrent=5,
    timeout=1000,
    validator_model="gpt-4o-mini@openai",
    output_dir="intranet/evals",
    examples_count=4,
    batch_size: int | None = None,
):
    """Run comprehensive semantic evaluation of the RAG system."""

    print("🔬 Starting RAG System Semantic Evaluation...")
    print("=" * 70)

    start_time = time.time()

    try:
        # Initialize evaluation results
        results = EvaluationResults()
        results.config = {
            "qa_pairs_file": qa_pairs_file,
            "num_questions": num_questions,
            "api_url": api_url,
            "max_concurrent": max_concurrent,
            "timeout": timeout,
            "validator_model": validator_model,
            "examples_count": examples_count,
        }
        results.qa_source = qa_pairs_file
        results.api_endpoint = api_url

        # Pre-compute results file path for rolling writes
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        results_file = output_path / f"semantic_evaluation_{timestamp}.json"

        # Load QA pairs
        print(f"📚 Loading QA pairs from {qa_pairs_file}...")
        qa_manager = QAManager(loader=JSONQALoader())

        try:
            qa_pairs = qa_manager.load_qa_pairs(qa_pairs_file)
            if not qa_pairs:
                print("❌ No QA pairs loaded!")
                return False
        except FileNotFoundError:
            print(f"❌ QA pairs file not found: {qa_pairs_file}")
            print("   Make sure the QA pairs file exists")
            return False
        except Exception as e:
            print(f"❌ Error loading QA pairs: {e}")
            return False

        # Get subset if requested
        qa_subset = qa_manager.get_subset(num_questions)
        questions = qa_manager.get_questions()[: len(qa_subset)]

        print(f"📊 QA Pairs loaded: {len(qa_subset)} questions")

        # Initialize HTTP client
        print(f"🌐 Initializing HTTP client for {api_url}...")
        http_client = RAGHTTPClient(
            base_url=api_url,
            timeout=timeout,
            max_retries=2,
        )

        # Test API connectivity
        print("🔍 Testing API connectivity...")
        health_status = await http_client.health_check()

        if health_status.get("status") != "healthy":
            print(f"⚠️ API health check failed: {health_status}")
            print("   Proceeding with evaluation anyway...")

        # Generate validation examples from QA pairs
        print(f"📝 Generating {examples_count} validation examples...")
        validation_examples = qa_manager.generate_validation_examples(
            num_examples=examples_count,
            include_negative=True,
        )
        results.validation_examples_used = validation_examples

        # Initialize semantic validator with examples
        print(f"🔍 Initializing semantic validator with {validator_model}...")
        validator = SemanticValidator(
            model_name=validator_model,
            examples=validation_examples,
        )

        # Get system info
        print("📊 Getting system information...")
        system_stats = await http_client.get_system_stats()
        results.system_info = system_stats

        print(f"\n🎯 Evaluation Configuration:")
        print(f"   📚 QA Pairs: {len(qa_subset)} questions")
        print(f"   🌐 API Endpoint: {api_url}")
        print(f"   🔍 Validator Model: {validator_model}")
        print(f"   ⚡ Max Concurrent: {max_concurrent}")
        print(f"   ⏱️ Timeout: {timeout}s")
        print(f"   📝 Validation Examples: {len(validation_examples)}")
        if batch_size:
            print(f"   📦 Batch Size: {batch_size}")

        if system_stats.get("total_documents", 0) > 0:
            print(f"   📖 Documents in System: {system_stats['total_documents']}")

        # Step 1: Query RAG API
        print(f"\n🚀 Step 1: Querying RAG API & validating in batches…")
        print("-" * 50)

        rag_responses: list = []
        validation_results_all: list = []

        # helper to process one slice
        async def _process_batch(batch_qs, batch_qa_pairs, batch_idx):
            print(
                f"   • Batch {batch_idx+1}: Q{batch_idx*batch_size+1}–Q{batch_idx*batch_size + len(batch_qs)}",
            )
            batch_rag = await http_client.query_batch(
                questions=batch_qs,
                max_concurrent=max_concurrent,
                conversation_prefix=f"eval_b{batch_idx}",
            )
            # validation
            validation_data = [
                {
                    "answer": r.answer,
                    "sources": r.sources,
                    "response_time": r.response_time,
                }
                for r in batch_rag
            ]
            batch_val = await validator.validate_batch(
                qa_pairs=[qa.to_dict() for qa in batch_qa_pairs],
                generated_responses=validation_data,
                max_concurrent=3,
            )
            # aggregate
            rag_responses.extend(batch_rag)
            validation_results_all.extend(batch_val)
            # push into EvaluationResults
            for qa_pair, rag_resp, val_res in zip(batch_qa_pairs, batch_rag, batch_val):
                q_eval = QuestionEvaluation.from_qa_and_responses(
                    qa_pair,
                    rag_resp,
                    val_res,
                )
                results.add_question_evaluation(q_eval)
            # stream save
            try:
                results.save_to_file(str(results_file))
            except Exception as e:
                print(f"⚠️ Could not stream batch results: {e}")

        if batch_size is None or batch_size <= 0:
            await _process_batch(
                batch_qs=questions,
                batch_qa_pairs=qa_subset,
                batch_idx=0,
            )
        else:
            for b_idx, start in enumerate(range(0, len(questions), batch_size)):
                end = start + batch_size
                await _process_batch(
                    batch_qs=questions[start:end],
                    batch_qa_pairs=qa_subset[start:end],
                    batch_idx=b_idx,
                )

        # Step 3: Compile results (already added incrementally)
        print(f"\n📊 Compiling final results…")
        print("-" * 50)

        # Finalize results
        results.finalize()

        total_time = time.time() - start_time

        # Display comprehensive results
        print(f"\n{'='*70}")
        print(f"📊 SEMANTIC EVALUATION RESULTS")
        print(f"{'='*70}")

        metrics = results.performance_metrics
        summary = results.summary

        # Overall performance
        print(f"🎯 Overall Grade: {summary.overall_grade}")
        print(f"   {summary.grade_explanation}")
        print(f"")
        print(f"📈 Key Metrics:")
        print(f"   • Total Questions: {metrics.total_questions}")
        print(
            f"   • Accuracy Rate: {metrics.accuracy_rate:.1%} ({metrics.correct_answers}/{metrics.total_questions})",
        )
        print(f"   • Average Confidence: {metrics.average_confidence:.3f}")
        print(f"   • Success Rate: {metrics.success_rate:.1%}")
        print(f"   • Average Response Time: {metrics.average_response_time:.2f}s")
        print(f"   • Total Evaluation Time: {total_time:.1f}s")

        # Detailed metrics
        if metrics.average_semantic_similarity:
            print(f"\n🔍 Detailed Analysis:")
            print(
                f"   • Semantic Similarity: {metrics.average_semantic_similarity:.3f}",
            )
            print(f"   • Factual Accuracy: {metrics.average_factual_accuracy:.3f}")
            print(f"   • Completeness: {metrics.average_completeness:.3f}")
            print(f"   • Relevance: {metrics.average_relevance:.3f}")

        # Score distribution
        print(f"\n📊 Score Distribution:")
        for category, count in metrics.score_distribution.items():
            percentage = (
                count / metrics.total_questions * 100
                if metrics.total_questions > 0
                else 0
            )
            print(f"   • {category.title()}: {count} ({percentage:.1f}%)")

        # Error analysis
        if results.error_analysis.error_rate > 0:
            print(f"\n❌ Error Analysis:")
            print(f"   • Error Rate: {results.error_analysis.error_rate:.1%}")
            print(f"   • RAG Errors: {len(results.error_analysis.rag_errors)}")
            print(
                f"   • Validation Errors: {len(results.error_analysis.validation_errors)}",
            )

            if results.error_analysis.common_error_patterns:
                print(f"   • Common Error Patterns:")
                for (
                    pattern,
                    count,
                ) in results.error_analysis.common_error_patterns.items():
                    print(f"     - {pattern}: {count}")

        # Sample results
        print(f"\n📝 Sample Results:")
        top_questions = results.get_top_questions(2)
        worst_questions = results.get_worst_questions(2)

        if top_questions:
            print(f"\n   🏆 Best Performing Questions:")
            for i, q in enumerate(top_questions, 1):
                print(
                    f"   {i}. Q{q.question_id}: {q.confidence_score:.3f} - {q.question[:80]}...",
                )
                print(f"      Reasoning: {q.reasoning[:100]}...")

        if worst_questions:
            print(f"\n   ⚠️ Questions Needing Attention:")
            for i, q in enumerate(worst_questions, 1):
                print(
                    f"   {i}. Q{q.question_id}: {q.confidence_score:.3f} - {q.question[:80]}...",
                )
                print(f"      Reasoning: {q.reasoning[:100]}...")

        # Strengths and recommendations
        if summary.strengths:
            print(f"\n✅ Strengths:")
            for strength in summary.strengths:
                print(f"   • {strength}")

        if summary.weaknesses:
            print(f"\n⚠️ Areas for Improvement:")
            for weakness in summary.weaknesses:
                print(f"   • {weakness}")

        if summary.recommendations:
            print(f"\n🎯 Recommendations:")
            for rec in summary.recommendations:
                print(f"   • {rec}")

        # Final save (ensure fully written)
        results.save_to_file(str(results_file))

        print(f"\n💾 Results saved to: {results_file}")
        print(f"📊 Unify logs available for detailed analysis")

        # Final assessment
        print(f"\n{'='*70}")
        if summary.overall_grade in ["A", "B"]:
            print(f"🎉 EXCELLENT PERFORMANCE! Grade: {summary.overall_grade}")
            print(f"   The system demonstrates strong semantic understanding")
        elif summary.overall_grade == "C":
            print(f"✅ GOOD PERFORMANCE! Grade: {summary.overall_grade}")
            print(f"   The system is working well with room for improvement")
        elif summary.overall_grade == "D":
            print(f"⚠️ FAIR PERFORMANCE - Grade: {summary.overall_grade}")
            print(f"   The system needs optimization")
        else:
            print(f"❌ POOR PERFORMANCE - Grade: {summary.overall_grade}")
            print(f"   The system requires significant improvement")

        print(f"\n🔗 Next Steps:")
        print(f"   📊 Review detailed results in: {results_file}")
        print(f"   🔍 Focus on questions with low confidence scores")
        print(f"   🛠️ Address common error patterns if present")
        print(f"   📈 Monitor performance trends over time")

        return summary.overall_grade in ["A", "B", "C"]

    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run comprehensive RAG system semantic evaluation",
    )
    parser.add_argument(
        "--qa-pairs-file",
        type=str,
        default="intranet/data/qa_pairs.json",
        help="Path to QA pairs JSON file (default: intranet/data/qa_pairs.json)",
    )
    parser.add_argument(
        "--num-questions",
        type=int,
        default=None,
        help="Number of test questions to evaluate (default: all)",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://0.0.0.0:8000",
        help="API base URL for HTTP evaluation (default: http://0.0.0.0:8000)",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=5,
        help="Maximum concurrent HTTP requests (default: 5)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="HTTP timeout in seconds (default: 120)",
    )
    parser.add_argument(
        "--validator-model",
        type=str,
        default="gpt-4o-mini@openai",
        help="Model for semantic validation (default: gpt-4o-mini@openai)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="intranet/evals",
        help="Output directory for results (default: intranet/evals)",
    )
    parser.add_argument(
        "--examples-count",
        type=int,
        default=4,
        help="Number of validation examples to generate (default: 4)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Number of questions per API batch (default: all in one batch)",
    )

    args = parser.parse_args()

    print(f"🔬 RAG System Semantic Evaluation")
    print(f"📚 QA Pairs: {args.qa_pairs_file}")
    print(f"🌐 API URL: {args.api_url}")
    print(f"🔍 Validator: {args.validator_model}")
    print()

    success = asyncio.run(
        run_semantic_evaluation(
            qa_pairs_file=args.qa_pairs_file,
            num_questions=args.num_questions,
            api_url=args.api_url,
            max_concurrent=args.max_concurrent,
            timeout=args.timeout,
            validator_model=args.validator_model,
            output_dir=args.output_dir,
            examples_count=args.examples_count,
            batch_size=args.batch_size,
        ),
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
