from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import unify

from ..common.filter_utils import normalize_filter_expr
from ..common.search_utils import (
    is_plain_identifier,
    ensure_vector_for_source,
    fetch_top_k_by_terms_with_score,
    fetch_scores_for_ids,
)
from ..common.semantic_search import extract_placeholders
from .types.message import Message
from ..contact_manager.types.contact import Contact
from ..common.embed_utils import ensure_vector_column

# Module-level tuning knobs for ranking/backfill behaviour
OVERSAMPLE_FACTOR = 5
BASE_OVERSAMPLE_MIN = 50
SEARCH_BATCH_SIZE = 1000


def _classify_terms(
    self,
    references: Dict[str, str],
    *,
    transcript_context: str,
    contact_context: str,
) -> Tuple[list[tuple[str, str]], list[tuple[str, str]], list[tuple[str, str]]]:
    """Return message, sender-contact, and receiver-contact embedding terms.

    Each term is a pair of (embed_column_name, ref_text).
    """
    msg_fields = set(Message.model_fields.keys())
    contact_fields = set(Contact.model_fields.keys())

    msg_embed_columns: list[tuple[str, str]] = []
    sender_contact_embed_columns: list[tuple[str, str]] = []
    receiver_contact_embed_columns: list[tuple[str, str]] = []

    # 1) message-side terms
    for source_expr, ref_text in references.items():
        placeholders = (
            extract_placeholders(source_expr)
            if not is_plain_identifier(source_expr)
            else []
        )
        is_message_side = False
        if is_plain_identifier(source_expr):
            is_message_side = (
                source_expr in msg_fields
                and not source_expr.startswith("sender_")
                and not source_expr.startswith("receiver_")
            )
        else:
            is_message_side = any(ph in msg_fields for ph in placeholders)
        if is_message_side:
            embed_column_name = ensure_vector_for_source(
                transcript_context,
                source_expr,
            )
            msg_embed_columns.append((embed_column_name, ref_text))

    # 2) contact-side terms (sender/receiver role)
    for source_expr, ref_text in references.items():
        placeholders = (
            extract_placeholders(source_expr)
            if not is_plain_identifier(source_expr)
            else []
        )
        role: Optional[str] = None
        base_expr = source_expr
        if is_plain_identifier(source_expr):
            if source_expr.startswith("sender_"):
                role = "sender"
                base_expr = source_expr[len("sender_") :]
            elif source_expr.startswith("receiver_"):
                role = "receiver"
                base_expr = source_expr[len("receiver_") :]
            elif (source_expr in contact_fields) and (source_expr not in msg_fields):
                role = "sender"
                base_expr = source_expr
        else:
            if (
                (len(placeholders) > 0)
                and all(ph in contact_fields for ph in placeholders)
                and not any(ph in msg_fields for ph in placeholders)
            ):
                if base_expr.startswith("sender_"):
                    role = "sender"
                    base_expr = base_expr[len("sender_") :]
                elif base_expr.startswith("receiver_"):
                    role = "receiver"
                    base_expr = base_expr[len("receiver_") :]
                else:
                    role = "sender"
        if role is not None:
            embed_column_name = ensure_vector_for_source(
                contact_context,
                base_expr,
            )
            if role == "sender":
                sender_contact_embed_columns.append((embed_column_name, ref_text))
            else:
                receiver_contact_embed_columns.append((embed_column_name, ref_text))

    return (
        msg_embed_columns,
        sender_contact_embed_columns,
        receiver_contact_embed_columns,
    )


def _contact_context_for_transcript_context(transcript_context: str) -> str:
    """Return the sibling Contacts context for a concrete Transcripts context."""

    root_context = transcript_context.rsplit("/", 1)[0]
    return f"{root_context}/Contacts"


def format_contacts_and_messages(self, messages: List[Message]) -> Dict[str, Any]:
    """Return a combined payload for contacts and messages (stable shape)."""
    if not messages:
        return {"messages": []}

    unique_ids: set[int] = set()
    for m in messages:
        try:
            if m.sender_id is not None:
                unique_ids.add(int(m.sender_id))
        except Exception:
            pass
        if isinstance(m.receiver_ids, list):
            for rid in m.receiver_ids:
                try:
                    if rid is not None:
                        unique_ids.add(int(rid))
                except Exception:
                    pass

    contacts_payload: Dict[str, Any] = {}
    if unique_ids:
        ids_expr = ", ".join(str(i) for i in sorted(unique_ids))
        flt = f"contact_id in [{ids_expr}]"
        try:
            contacts_payload = self._contact_manager.filter_contacts(
                filter=flt,
                limit=len(unique_ids),
            )
        except Exception:
            contacts_payload = {}

    message_keys_to_shorthand: dict[str, str] = Message.shorthand_map()
    shorthand_to_message_keys: dict[str, str] = Message.shorthand_inverse_map()

    return {
        **contacts_payload,
        "message_keys_to_shorthand": message_keys_to_shorthand,
        "messages": messages,
        "shorthand_to_message_keys": shorthand_to_message_keys,
    }


def filter_messages(
    self,
    *,
    filter: Optional[str] = None,
    offset: int = 0,
    limit: int | None = 100,
) -> Dict[str, Any]:
    normalized = normalize_filter_expr(filter)
    collected: list[dict] = []
    fetch_limit = (offset + limit) if limit is not None else 1000
    for context in self._read_transcript_contexts():
        logs = unify.get_logs(
            context=context,
            filter=normalized,
            offset=0,
            limit=fetch_limit,
            sorting={"timestamp": "descending"},
            from_fields=list(Message.model_fields.keys()),
        )
        collected.extend(dict(lg.entries) for lg in logs)
    collected.sort(key=lambda row: row.get("timestamp"), reverse=True)
    window = collected[offset : (offset + limit) if limit is not None else None]
    results = [Message(**row) for row in window]
    return format_contacts_and_messages(self, results)


def search_messages(
    self,
    *,
    references: Optional[Dict[str, str]] = None,
    k: int = 10,
) -> Dict[str, Any]:
    # Default behaviour: when references is None/empty, return most recent
    if not references:
        collected: list[dict] = []
        for context in self._read_transcript_contexts():
            logs = unify.get_logs(
                context=context,
                limit=k,
                from_fields=list(Message.model_fields.keys()),
                sorting={"timestamp": "descending"},
            )
            collected.extend(dict(lg.entries) for lg in logs)
        collected.sort(key=lambda row: row.get("timestamp"), reverse=True)
        results = [Message(**row) for row in collected[:k]]
        return format_contacts_and_messages(self, results)

    transcript_contexts = self._read_transcript_contexts()
    terms_by_context: list[
        tuple[str, list[tuple[str, str]], list[tuple[str, str]], list[tuple[str, str]]]
    ] = []
    for context in transcript_contexts:
        terms_by_context.append(
            (
                context,
                *_classify_terms(
                    self,
                    references,
                    transcript_context=context,
                    contact_context=_contact_context_for_transcript_context(context),
                ),
            ),
        )

    has_contact_terms = any(
        sender_terms or receiver_terms
        for _, _, sender_terms, receiver_terms in terms_by_context
    )
    oversample = max(k * OVERSAMPLE_FACTOR, BASE_OVERSAMPLE_MIN)

    if not has_contact_terms:
        scored_rows: list[tuple[float, dict]] = []
        for context, msg_terms, _, _ in terms_by_context:
            if not msg_terms:
                ensure_vector_column(context, "_content_emb", "content")
                msg_terms = [("_content_emb", next(iter(references.values())))]
            context_rows, score_key = fetch_top_k_by_terms_with_score(
                context,
                msg_terms,
                k=oversample,
                allowed_fields=list(Message.model_fields.keys()),
            )
            for row in context_rows:
                try:
                    score = float(row.get(score_key, 0.0))
                except Exception:
                    score = 0.0
                scored_rows.append((score, row))
        scored_rows.sort(key=lambda item: item[0])
        results = []
        for _, row in scored_rows[:k]:
            try:
                results.append(Message(**row))
            except Exception:
                continue
        return format_contacts_and_messages(self, results)

    scored_rows: list[tuple[float, dict]] = []
    for context, msg_terms, sender_terms, receiver_terms in terms_by_context:
        all_messages = []
        offset = 0
        while True:
            batch = unify.get_logs(
                context=context,
                offset=offset,
                limit=SEARCH_BATCH_SIZE,
                from_fields=list(Message.model_fields.keys()),
                sorting={"timestamp": "descending"},
            )
            if not batch:
                break
            all_messages.extend(batch)
            if len(batch) < SEARCH_BATCH_SIZE:
                break
            offset += SEARCH_BATCH_SIZE
        if not all_messages:
            continue

        msg_ids: list[int] = []
        sender_ids: set[int] = set()
        receiver_ids: set[int] = set()
        for lg in all_messages:
            entries = lg.entries
            try:
                if entries.get("message_id") is not None:
                    msg_ids.append(int(entries["message_id"]))
                if entries.get("sender_id") is not None:
                    sender_ids.add(int(entries["sender_id"]))
                for receiver_id in entries.get("receiver_ids", []) or []:
                    receiver_ids.add(int(receiver_id))
            except Exception:
                continue

        msg_scores: dict[int, float] = {}
        if msg_terms and msg_ids:
            msg_scores, _ = fetch_scores_for_ids(
                context,
                msg_terms,
                id_field="message_id",
                ids=msg_ids,
            )

        contact_context = _contact_context_for_transcript_context(context)
        sender_scores: dict[int, float] = {}
        if sender_terms and sender_ids:
            sender_scores, _ = fetch_scores_for_ids(
                contact_context,
                sender_terms,
                id_field="contact_id",
                ids=sorted(sender_ids),
            )

        receiver_scores: dict[int, float] = {}
        if receiver_terms and receiver_ids:
            receiver_scores, _ = fetch_scores_for_ids(
                contact_context,
                receiver_terms,
                id_field="contact_id",
                ids=sorted(receiver_ids),
            )

        for lg in all_messages:
            entries = dict(lg.entries)
            mid = entries.get("message_id")
            if mid is None:
                continue
            try:
                mid_int = int(mid)
            except Exception:
                continue

            total_score = msg_scores.get(mid_int, 2.0) if msg_terms else 0.0

            if sender_terms:
                try:
                    sender_id = int(entries.get("sender_id"))
                except Exception:
                    sender_id = None
                total_score += (
                    sender_scores.get(sender_id, 2.0) if sender_id is not None else 2.0
                )

            if receiver_terms:
                receiver_score = 2.0
                for receiver_id in entries.get("receiver_ids", []) or []:
                    try:
                        candidate_score = receiver_scores.get(int(receiver_id))
                    except Exception:
                        candidate_score = None
                    if candidate_score is not None and candidate_score < receiver_score:
                        receiver_score = candidate_score
                total_score += receiver_score

            scored_rows.append((total_score, entries))

    scored_rows.sort(key=lambda item: item[0])
    results: List[Message] = []
    for _, row in scored_rows:
        if len(results) >= k:
            break
        try:
            results.append(Message(**row))
        except Exception:
            continue

    return format_contacts_and_messages(self, results)
