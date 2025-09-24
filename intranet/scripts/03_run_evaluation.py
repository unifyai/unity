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
    --timeout           HTTP timeout in seconds (default: 600)
    --validator-model   Model for semantic validation (default: o4-mini@openai)
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
from intranet.core.qa_loader import QAManager, PolicyJSONLoader
from intranet.core.evaluation_results import (
    EvaluationResults,
    QuestionEvaluation,
)


async def run_semantic_evaluation(
    data_dir: str | None = "intranet/data",
    num_questions=None,
    api_url="http://0.0.0.0:8000",
    max_concurrent=5,
    timeout=600,
    validator_model="o4-mini@openai",
    output_dir="intranet/evals",
    examples_count=4,
    batch_size: int | None = 1,
    per_difficulty: int = 2,
    save_after: int = 6,
    per_difficulty_map: dict | None = None,
    retreival_mode: str = "tool_loop",
):
    """Run comprehensive semantic evaluation of the RAG system."""

    print("🔬 Starting RAG System Semantic Evaluation...")
    print("=" * 70)

    start_time = time.time()

    try:
        # Initialize evaluation results
        results = EvaluationResults()
        results.config = {
            "data_dir": data_dir,
            "num_questions": num_questions,
            "api_url": api_url,
            "max_concurrent": max_concurrent,
            "timeout": timeout,
            "validator_model": validator_model,
            "examples_count": examples_count,
            "per_difficulty": per_difficulty,
            "batch_size": batch_size,
        }
        results.qa_source = data_dir
        results.api_endpoint = api_url

        # Pre-compute results file path for rolling writes
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        results_file = output_path / f"semantic_evaluation_{timestamp}.json"

        # Load QA pairs (supports single file or directory)
        print(f"📚 Loading QA pairs from data dir/file: {data_dir} (unified format)...")
        data_path = Path(data_dir or "intranet/data")
        loader = PolicyJSONLoader()
        grouped = loader.load_grouped(str(data_path))
        if not grouped:
            print("❌ No policy files loaded!")
            return False

        # Single-file mode → ignore file batching
        is_single_file = data_path.is_file()
        if is_single_file and (batch_size is None or batch_size != 1):
            print(
                f"ℹ️ Single-file input detected; ignoring --batch-size={batch_size} and using 1.",
            )
            batch_size = 1

        # Get subset if requested
        # Build per-file subsets: pick N per difficulty (easy/medium/hard) for each policy
        def _select_per_difficulty(pairs: list, n: int) -> list:
            by_diff = {"easy": [], "medium": [], "hard": []}
            for p in pairs:
                diff = (p.metadata or {}).get("difficulty")
                if diff in by_diff:
                    by_diff[diff].append(p)
            # Allow granular per-difficulty override
            if per_difficulty_map:
                easy_n = max(0, int(per_difficulty_map.get("easy", n)))
                med_n = max(0, int(per_difficulty_map.get("medium", n)))
                hard_n = max(0, int(per_difficulty_map.get("hard", n)))
            else:
                easy_n = med_n = hard_n = max(0, n)

            selected: list = []
            selected.extend(by_diff["easy"][:easy_n])
            selected.extend(by_diff["medium"][:med_n])
            selected.extend(by_diff["hard"][:hard_n])
            return selected

        file_batches: list[tuple[str, list]] = []
        total_selected = 0
        for policy_name, pairs in grouped:
            sel = _select_per_difficulty(pairs, per_difficulty)
            if sel:
                file_batches.append((policy_name, sel))
                total_selected += len(sel)

        print(
            f"📊 Built evaluation set: {len(file_batches)} files, {total_selected} questions",
        )

        # Initialize HTTP client
        print(f"🌐 Initializing HTTP client for {api_url}...")
        http_client = RAGHTTPClient(
            base_url=api_url,
            timeout=timeout,
            max_retries=2,
        )

        # Test API connectivity
        print("🔍 Testing API connectivity...")
        try:
            health_status = await http_client.health_check()
        except Exception as e:
            print(f"⚠️ API health check request failed: {e}")
            health_status = {"status": "unknown", "error": str(e)}

        if health_status.get("status") != "healthy":
            print(f"⚠️ API health check failed: {health_status}")
            print("   Proceeding with evaluation anyway...")

        # Generate validation examples from QA pairs (optional synthetic examples)
        print(f"📝 Generating {examples_count} validation examples...")
        # Keep examples purely synthetic and independent of grouped selection
        qa_manager = QAManager(loader=PolicyJSONLoader())
        # Flat load to reuse the existing example generation path
        _flat_pairs = qa_manager.load_qa_pairs(data_dir or "intranet/data")
        validation_examples = qa_manager.generate_validation_examples(
            num_examples=max(0, int(examples_count or 0)),
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
        print(f"   📚 Selected Questions: {total_selected}")
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

        # helper to process one file worth of questions
        async def _process_file_batch(
            policy_name: str,
            qa_pairs_file: list,
            file_idx: int,
        ):
            print(
                f"   • File {file_idx+1}/{len(file_batches)}: {policy_name} ({len(qa_pairs_file)} qs)",
            )

            # Aggregate into a fresh EvaluationResults for this file
            file_results = EvaluationResults()
            file_results.config = results.config
            file_results.qa_source = policy_name
            file_results.api_endpoint = results.api_endpoint

            # Stream queries with concurrency and evaluate/snapshot every `save_after`
            semaphore = asyncio.Semaphore(max_concurrent)

            async def _query_one(idx: int, question: str):
                async with semaphore:
                    return idx, await http_client.query(
                        question,
                        conversation_id=f"eval_{policy_name}_{idx:04d}",
                        user_id=f"eval_user_{idx:04d}",
                        retreival_mode=retreival_mode,
                    )

            tasks = [
                asyncio.create_task(_query_one(i, qa.question))
                for i, qa in enumerate(qa_pairs_file)
            ]

            pending_indices: list[int] = []
            pending_responses: list = []

            async def _process_chunk(indices: list[int], responses: list):
                if not indices:
                    return
                # Prepare matching QA pairs and validation inputs
                qa_chunk = [qa_pairs_file[i] for i in indices]
                # Only validate successful responses with non-empty answers
                validation_pairs = []
                validation_data = []
                for qa_p, r in zip(qa_chunk, responses):
                    if getattr(r, "success", False) and (
                        r.answer is not None and str(r.answer).strip() != ""
                    ):
                        validation_pairs.append(qa_p)
                        validation_data.append(
                            {
                                "answer": r.answer,
                                "sources": r.sources,
                                "response_time": r.response_time,
                            },
                        )
                if validation_pairs:
                    val_results = await validator.validate_batch(
                        qa_pairs=[qa.to_dict() for qa in validation_pairs],
                        generated_responses=validation_data,
                        max_concurrent=3,
                    )
                else:
                    val_results = []
                # Aggregate into file and global results
                v_idx = 0
                for qa_pair, rag_resp in zip(qa_chunk, responses):
                    if getattr(rag_resp, "success", False) and (
                        rag_resp.answer is not None
                        and str(rag_resp.answer).strip() != ""
                    ):
                        val_res = val_results[v_idx]
                        v_idx += 1
                    else:
                        # Create a minimal validation record for failed responses
                        val_res = {
                            "score": 0.0,
                            "details": {
                                "reason": rag_resp.error or "unsuccessful response",
                            },
                        }
                    q_eval = QuestionEvaluation.from_qa_and_responses(
                        qa_pair,
                        rag_resp,
                        val_res,
                    )
                    file_results.add_question_evaluation(q_eval)
                    results.add_question_evaluation(q_eval)
                # Rolling save of global results
                try:
                    results.finalize()
                    results.save_to_file(str(results_file))
                    print(
                        f"   🗂️ Rolling update saved ({len(results.question_evaluations)} total questions so far)",
                    )
                except Exception as e:
                    print(f"⚠️ Rolling save failed: {e}")

            # Consume completions as they arrive
            for fut in asyncio.as_completed(tasks):
                idx, rag_resp = await fut
                pending_indices.append(idx)
                pending_responses.append(rag_resp)
                if save_after and len(pending_responses) >= save_after:
                    await _process_chunk(pending_indices, pending_responses)
                    pending_indices = []
                    pending_responses = []

            # Process any remaining responses
            if pending_responses:
                await _process_chunk(pending_indices, pending_responses)

            # Finalize and save per-file results
            file_results.finalize()
            file_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_out = output_path / f"semantic_eval_{policy_name}_{file_ts}.json"
            try:
                file_results.save_to_file(str(file_out))
                print(f"   💾 Saved: {file_out}")
            except Exception as e:
                print(f"⚠️ Could not save file results for {policy_name}: {e}")

        # Process files in batches of 'batch_size' files (default 1)
        if batch_size is None or batch_size <= 0:
            batch_size = 1

        for b_idx, start in enumerate(range(0, len(file_batches), batch_size)):
            chunk = file_batches[start : start + batch_size]
            # Run each file in this chunk sequentially to avoid mixing outputs
            for idx_in, (pname, pqs) in enumerate(chunk):
                await _process_file_batch(pname, pqs, start + idx_in)

            # After each batch, update and save the global results incrementally
            try:
                results.finalize()
                results.save_to_file(str(results_file))
                print(f"   🗂️ Updated global results: {results_file}")
            except Exception as e:
                print(f"⚠️ Could not update global results: {e}")

        # Step 3: Compile results (already added incrementally)
        print(f"\n📊 Compiling final results…")
        print("-" * 50)

        # Finalize results
        results.finalize()

        total_time = time.time() - start_time
        # Record wall clock time so totals are not an overcount in parallel runs
        results.performance_metrics.wall_clock_time = total_time

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
        "--data-dir",
        type=str,
        default="intranet/data",
        help="Directory or file with unified policy eval JSONs (easy/medium/hard)",
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
        default=600,
        help="HTTP timeout in seconds (default: 600)",
    )
    parser.add_argument(
        "--validator-model",
        type=str,
        default="o4-mini@openai",
        help="Model for semantic validation (default: o4-mini@openai)",
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
        default=1,
        help="Number of files per evaluation batch (default: 1 file per batch)",
    )
    parser.add_argument(
        "--per-difficulty",
        type=int,
        default=2,
        help="Number of questions to sample from each difficulty per file (default: 2)",
    )
    parser.add_argument(
        "--save-after",
        type=int,
        default=6,
        help="Evaluate and persist after this many completed queries (rolling save)",
    )
    parser.add_argument(
        "--per-difficulty-map",
        type=str,
        default=None,
        help='JSON string to override counts per difficulty, e.g. {"easy":10,"medium":10,"hard":5}',
    )
    parser.add_argument(
        "--retreival-mode",
        type=str,
        default="tool_loop",
        choices=["tool_loop", "llm"],
        help='Retrieval path: "tool_loop" (default) or "llm"',
    )

    args = parser.parse_args()

    print(f"🔬 RAG System Semantic Evaluation")
    print(f"📚 Data Dir/File: {args.data_dir}")
    print(f"🌐 API URL: {args.api_url}")
    print(f"🔍 Validator: {args.validator_model}")
    print()

    success = asyncio.run(
        run_semantic_evaluation(
            data_dir=args.data_dir,
            num_questions=args.num_questions,
            api_url=args.api_url,
            max_concurrent=args.max_concurrent,
            timeout=args.timeout,
            validator_model=args.validator_model,
            output_dir=args.output_dir,
            examples_count=args.examples_count,
            batch_size=args.batch_size,
            per_difficulty=args.per_difficulty,
            save_after=args.save_after,
            per_difficulty_map=(
                json.loads(args.per_difficulty_map) if args.per_difficulty_map else None
            ),
            retreival_mode=args.retreival_mode,
        ),
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
