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


class JSONQALoader:
    """Loader for JSON format QA pairs."""

    def __init__(self, required_fields: Optional[List[str]] = None):
        """
        Initialize the JSON QA loader.

        Args:
            required_fields: List of required fields in each QA pair
        """
        self.required_fields = required_fields or ["question", "answer"]

    def load(self, file_path: str) -> List[QAPair]:
        """
        Load QA pairs from a JSON file.

        Args:
            file_path: Path to the JSON file

        Returns:
            List of QAPair objects
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"QA pairs file not found: {file_path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not self.validate_format(data):
            raise ValueError(f"Invalid QA pairs format in {file_path}")

        qa_pairs = []
        for item in data:
            try:
                qa_pair = QAPair.from_dict(item)
                qa_pairs.append(qa_pair)
            except KeyError as e:
                print(
                    f"⚠️ Skipping invalid QA pair (missing {e}): {item.get('id', 'unknown')}",
                )
                continue

        print(f"📚 Loaded {len(qa_pairs)} QA pairs from {file_path}")
        return qa_pairs

    def validate_format(self, data: Any) -> bool:
        """
        Validate the format of QA pairs data.

        Args:
            data: The loaded data to validate

        Returns:
            True if format is valid, False otherwise
        """
        if not isinstance(data, list):
            print("❌ QA pairs data must be a list")
            return False

        if not data:
            print("❌ QA pairs list is empty")
            return False

        # Check first few items for required fields
        for i, item in enumerate(data[:3]):
            if not isinstance(item, dict):
                print(f"❌ QA pair {i} is not a dictionary")
                return False

            for field in self.required_fields:
                if field not in item:
                    print(f"❌ QA pair {i} missing required field: {field}")
                    return False

        return True


class QAManager:
    """
    Manager for QA pairs with extensible loading and validation example generation.
    """

    def __init__(self, loader: QALoader = None):
        """
        Initialize the QA manager.

        Args:
            loader: QA loader instance (defaults to JSONQALoader)
        """
        self.loader = loader or JSONQALoader()
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
