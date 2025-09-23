"""
QA Pairs Loader for RAG Evaluation
==================================

This module provides flexible loading and management of QA pairs for evaluation
with support for different formats and extensible validation examples.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Protocol
from dataclasses import dataclass


@dataclass
class QAPair:
    """Standardized QA pair structure."""

    id: int
    question: str
    answer: str
    sources: List[str]
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "question": self.question,
            "answer": self.answer,
            "sources": self.sources,
            "metadata": self.metadata or {},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QAPair":
        """Create QAPair from dictionary."""
        return cls(
            id=data.get("id", 0),
            question=data["question"],
            answer=data["answer"],
            sources=data.get("sources", []),
            metadata=data.get("metadata"),
        )


class QALoader(Protocol):
    """Protocol for QA pair loaders."""

    def load(self, source: str) -> List[QAPair]:
        """Load QA pairs from a source."""
        ...

    def validate_format(self, data: Any) -> bool:
        """Validate the format of the loaded data."""
        ...


class PolicyJSONLoader:
    """Loader for the unified policy eval format under intranet/data.

    Expects JSON files with top-level keys: "easy", "medium", "hard",
    each mapping to a list of objects with {"question": str, "answer": str}.
    Can load a single JSON file or a directory of JSON files.
    """

    def __init__(self) -> None:
        pass

    def _load_one_file(self, file_path: Path, next_id: int) -> tuple[list[QAPair], int]:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        qa_pairs: list[QAPair] = []
        policy_name = file_path.stem

        # Unified keys and their order
        for difficulty in ("easy", "medium", "hard"):
            items = data.get(difficulty, []) or []
            if not isinstance(items, list):
                continue
            for item in items:
                q = item.get("question")
                a = item.get("answer")
                if not isinstance(q, str) or not isinstance(a, str):
                    continue
                qa_pairs.append(
                    QAPair(
                        id=next_id,
                        question=q,
                        answer=a,
                        sources=[policy_name],
                        metadata={"difficulty": difficulty, "policy": policy_name},
                    ),
                )
                next_id += 1
        return qa_pairs, next_id

    def load(self, source: str) -> List[QAPair]:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Policy data path not found: {source}")

        all_pairs: list[QAPair] = []
        next_id = 1

        if path.is_file() and path.suffix.lower() == ".json":
            pairs, next_id = self._load_one_file(path, next_id)
            all_pairs.extend(pairs)
        elif path.is_dir():
            for fp in sorted(path.glob("*.json")):
                pairs, next_id = self._load_one_file(fp, next_id)
                all_pairs.extend(pairs)
        else:
            raise ValueError(
                "Source must be a .json file or a directory of .json files",
            )

        print(
            f"📚 Loaded {len(all_pairs)} QA pairs from {source} (unified policy format)",
        )
        return all_pairs

    def validate_format(self, data: Any) -> bool:  # Not used; validation done per file
        return True

    def load_grouped(self, source: str) -> list[tuple[str, List[QAPair]]]:
        """Load QA pairs grouped per file (policy_name, pairs).

        Args:
            source: JSON file path or directory path

        Returns:
            List of tuples: (policy_name, list[QAPair]) preserving per-file grouping
        """
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Policy data path not found: {source}")

        groups: list[tuple[str, List[QAPair]]] = []
        next_id = 1

        def _append_group(fp: Path, next_id_in: int) -> int:
            pairs, next_after = self._load_one_file(fp, next_id_in)
            groups.append((fp.stem, pairs))
            return next_after

        if path.is_file() and path.suffix.lower() == ".json":
            next_id = _append_group(path, next_id)
        elif path.is_dir():
            for fp in sorted(path.glob("*.json")):
                next_id = _append_group(fp, next_id)
        else:
            raise ValueError(
                "Source must be a .json file or a directory of .json files",
            )

        return groups


class QAManager:
    """
    Manager for QA pairs with extensible loading and validation example generation.
    """

    def __init__(self, loader: QALoader = None):
        """
        Initialize the QA manager.

        Args:
            loader: QA loader instance (defaults to PolicyJSONLoader)
        """
        self.loader = loader or PolicyJSONLoader()
        self.qa_pairs: List[QAPair] = []
        self.loaded_from: Optional[str] = None

    def load_qa_pairs(self, source: str) -> List[QAPair]:
        """
        Load QA pairs from a source.

        Args:
            source: Source path or identifier

        Returns:
            List of loaded QA pairs
        """
        self.qa_pairs = self.loader.load(source)
        self.loaded_from = source

        # Log summary
        if self.qa_pairs:
            avg_q_len = sum(len(qa.question) for qa in self.qa_pairs) / len(
                self.qa_pairs,
            )
            avg_a_len = sum(len(qa.answer) for qa in self.qa_pairs) / len(self.qa_pairs)

            print(f"📊 QA Pairs Summary:")
            print(f"   • Total pairs: {len(self.qa_pairs)}")
            print(f"   • Average question length: {avg_q_len:.1f} chars")
            print(f"   • Average answer length: {avg_a_len:.1f} chars")

            # Show source distribution
            sources = {}
            for qa in self.qa_pairs:
                for source in qa.sources:
                    sources[source] = sources.get(source, 0) + 1

            if sources:
                print(f"   • Unique sources: {len(sources)}")
                top_sources = sorted(sources.items(), key=lambda x: x[1], reverse=True)[
                    :3
                ]
                for source, count in top_sources:
                    print(f"     - {source}: {count} references")

        return self.qa_pairs

    def get_subset(
        self,
        num_questions: Optional[int] = None,
        start_index: int = 0,
    ) -> List[QAPair]:
        """
        Get a subset of QA pairs.

        Args:
            num_questions: Number of questions to return (None for all)
            start_index: Starting index

        Returns:
            Subset of QA pairs
        """
        if not self.qa_pairs:
            return []

        end_index = len(self.qa_pairs)
        if num_questions:
            end_index = min(start_index + num_questions, len(self.qa_pairs))

        subset = self.qa_pairs[start_index:end_index]

        if subset != self.qa_pairs:
            print(f"📋 Using subset: {len(subset)}/{len(self.qa_pairs)} QA pairs")

        return subset

    def generate_validation_examples(
        self,
        num_examples: int = 3,
        include_negative: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Generate validation examples from the loaded QA pairs.

        Args:
            num_examples: Number of examples to generate
            include_negative: Whether to include negative examples

        Returns:
            List of validation examples for the semantic validator
        """
        if not self.qa_pairs:
            print("⚠️ No QA pairs loaded, using default examples")
            return []

        examples = []

        # Positive examples from actual QA pairs
        positive_count = num_examples // 2 if include_negative else num_examples
        for i in range(min(positive_count, len(self.qa_pairs))):
            qa = self.qa_pairs[i]

            # Create a positive example with slight rephrasing
            example = {
                "question": qa.question,
                "expected_answer": qa.answer,
                "generated_answer": self._rephrase_answer(qa.answer),
                "evaluation": {
                    "is_correct": True,
                    "confidence_score": 0.95,
                    "reasoning": f"The generated answer correctly conveys the same information as the expected answer, with appropriate rephrasing that maintains factual accuracy.",
                    "semantic_similarity": 0.95,
                    "factual_accuracy": 1.0,
                    "completeness": 1.0,
                    "relevance": 1.0,
                },
            }
            examples.append(example)

        # Negative examples with factual errors
        if include_negative and len(self.qa_pairs) > 1:
            negative_count = num_examples - positive_count
            for i in range(min(negative_count, len(self.qa_pairs) - 1)):
                qa = self.qa_pairs[i]

                # Create a negative example with factual error
                wrong_answer = self._create_wrong_answer(qa.answer)

                example = {
                    "question": qa.question,
                    "expected_answer": qa.answer,
                    "generated_answer": wrong_answer,
                    "evaluation": {
                        "is_correct": False,
                        "confidence_score": 0.2,
                        "reasoning": f"The generated answer contains factual errors or incorrect information that does not match the expected answer.",
                        "semantic_similarity": 0.3,
                        "factual_accuracy": 0.0,
                        "completeness": 0.5,
                        "relevance": 0.8,
                    },
                }
                examples.append(example)

        print(f"📝 Generated {len(examples)} validation examples from QA pairs")
        return examples

    def _rephrase_answer(self, answer: str) -> str:
        """Create a rephrased version of an answer for positive examples."""
        # Simple rephrasing strategies
        rephrased = answer

        # Replace "must" with "should" or "needs to"
        if "must" in rephrased.lower():
            rephrased = rephrased.replace("must", "needs to").replace(
                "Must",
                "Needs to",
            )

        # Add clarifying phrases
        if len(rephrased) < 100:
            rephrased = f"According to the policy, {rephrased.lower()}"

        # Replace "within" with "in"
        rephrased = rephrased.replace("within", "in").replace("Within", "In")

        return rephrased

    def _create_wrong_answer(self, correct_answer: str) -> str:
        """Create a factually incorrect version of an answer for negative examples."""
        wrong = correct_answer

        # Change numbers if present
        import re

        numbers = re.findall(r"\d+", wrong)
        for num in numbers:
            # Double the number or add 5
            new_num = str(int(num) * 2) if int(num) < 20 else str(int(num) + 5)
            wrong = wrong.replace(num, new_num, 1)
            break  # Only change first number

        # Change time units
        wrong = wrong.replace("working days", "calendar days")
        wrong = wrong.replace("days", "weeks")
        wrong = wrong.replace("mph", "kmh")

        # Change quantities
        wrong = wrong.replace("Class 2", "Class 3")
        wrong = wrong.replace("Stage 1", "Stage 2")

        return wrong

    def export_qa_pairs(
        self,
        output_path: str,
        qa_pairs: Optional[List[QAPair]] = None,
    ) -> None:
        """
        Export QA pairs to a JSON file.

        Args:
            output_path: Path to save the JSON file
            qa_pairs: QA pairs to export (defaults to loaded pairs)
        """
        pairs_to_export = qa_pairs or self.qa_pairs

        if not pairs_to_export:
            print("⚠️ No QA pairs to export")
            return

        output_data = [qa.to_dict() for qa in pairs_to_export]

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"💾 Exported {len(pairs_to_export)} QA pairs to {output_path}")

    def get_questions(self) -> List[str]:
        """Get list of questions from loaded QA pairs."""
        return [qa.question for qa in self.qa_pairs]

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the loaded QA pairs."""
        if not self.qa_pairs:
            return {"total_pairs": 0}

        questions = [qa.question for qa in self.qa_pairs]
        answers = [qa.answer for qa in self.qa_pairs]

        # Collect sources
        all_sources = []
        for qa in self.qa_pairs:
            all_sources.extend(qa.sources)

        source_counts = {}
        for source in all_sources:
            source_counts[source] = source_counts.get(source, 0) + 1

        return {
            "total_pairs": len(self.qa_pairs),
            "avg_question_length": sum(len(q) for q in questions) / len(questions),
            "avg_answer_length": sum(len(a) for a in answers) / len(answers),
            "unique_sources": len(source_counts),
            "total_source_references": len(all_sources),
            "source_distribution": source_counts,
            "loaded_from": self.loaded_from,
        }
