from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import unify

from ..common.filter_utils import normalize_filter_expr
from ..common.search_utils import (
    is_plain_identifier,
    ensure_vector_for_source,
    fetch_top_k_by_terms,
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
RECEIVER_OVERSAMPLE_MIN = 100
TOP_CONTACTS_FACTOR = 10
TOP_CONTACTS_MIN = 200
BATCH_OR_SIZE = 50


def _classify_terms(
    self,
    references: Dict[str, str],
) -> Tuple[list[tuple[str, str]], list[tuple[str, str]], list[tuple[str, str]], str]:
    """Return (msg_terms, sender_terms, receiver_terms, query_hash).

    Each term is a pair of (embed_column_name, ref_text).
    """
    msg_fields = set(Message.model_fields.keys())
    contact_fields = set(Contact.model_fields.keys())

    msg_embed_columns: list[tuple[str, str]] = []
    sender_contact_embed_columns: list[tuple[str, str]] = []
    receiver_contact_embed_columns: list[tuple[str, str]] = []

    canonical = "|".join(f"{k}=>{references[k]}" for k in sorted(references.keys()))
    import hashlib as _hashlib

    query_hash = _hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:12]

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
                self._transcripts_ctx,
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
                self._contact_manager._ctx,
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
        query_hash,
    )


def _aggregate_receiver_min(
    candidate_rows: list[dict],
    receiver_scores_map: Dict[int, float],
    *,
    base_score_key: str = "",
) -> list[tuple[int, float]]:
    combined: list[tuple[int, float]] = []
    for row in candidate_rows:
        mid = row.get("message_id")
        if mid is None:
            continue
        base_score = 0.0
        if base_score_key and (base_score_key in row):
            try:
                base_score = float(row.get(base_score_key, 0))
            except Exception:
                base_score = 0.0
        min_recv = 2.0
        rids = row.get("receiver_ids", [])
        if isinstance(rids, list) and rids:
            for rid in rids:
                try:
                    rv = receiver_scores_map.get(int(rid))
                    if rv is not None and rv < min_recv:
                        min_recv = rv
                except Exception:
                    continue
        combined.append((int(mid), base_score + min_recv))
    return combined


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
    logs = unify.get_logs(
        context=self._transcripts_ctx,
        filter=normalized,
        offset=offset,
        limit=limit,
        sorting={"timestamp": "descending"},
        from_fields=list(Message.model_fields.keys()),
    )
    results = [Message(**lg.entries) for lg in logs]
    return format_contacts_and_messages(self, results)


def search_messages(
    self,
    *,
    references: Optional[Dict[str, str]] = None,
    k: int = 10,
) -> Dict[str, Any]:
    # Default behaviour: when references is None/empty, return most recent
    if not references:
        logs = unify.get_logs(
            context=self._transcripts_ctx,
            limit=k,
            from_fields=list(Message.model_fields.keys()),
            sorting={"timestamp": "descending"},
        )
        results = [Message(**lg.entries) for lg in logs]
        return format_contacts_and_messages(self, results)

    (
        msg_embed_columns,
        sender_contact_embed_columns,
        receiver_contact_embed_columns,
        query_hash,
    ) = _classify_terms(self, references)

    # 3) No contact-side terms → single-table ranking
    if not sender_contact_embed_columns and not receiver_contact_embed_columns:
        if not msg_embed_columns:
            ensure_vector_column(self._transcripts_ctx, "_content_emb", "content")
            msg_embed_columns = [("_content_emb", next(iter(references.values())))]
        rows = fetch_top_k_by_terms(self._transcripts_ctx, msg_embed_columns, k=k)
        results = [Message(**lg) for lg in rows]
        return format_contacts_and_messages(self, results)

    # 4) Compute scores without join - fetch messages and scores separately
    left_ctx = self._transcripts_ctx
    right_ctx = self._contact_manager._ctx

    candidate_rows: list[dict]
    candidate_score_key = ""
    if msg_embed_columns or sender_contact_embed_columns:
        # Avoid problematic join - compute scores from original contexts directly
        oversample = max(k * OVERSAMPLE_FACTOR, BASE_OVERSAMPLE_MIN)

        # Step 1: Get all messages from Transcripts
        all_messages = unify.get_logs(
            context=left_ctx,
            limit=oversample * 3,  # Fetch more to have room after filtering
            from_fields=list(Message.model_fields.keys()),
            sorting={"timestamp": "descending"},
        )

        # Step 2: Compute message-based scores if needed
        msg_scores: dict[int, float] = {}
        if msg_embed_columns:
            msg_ids = [
                int(lg.entries.get("message_id"))
                for lg in all_messages
                if lg.entries.get("message_id") is not None
            ]
            if msg_ids:
                msg_scores, _ = fetch_scores_for_ids(
                    left_ctx,
                    msg_embed_columns,
                    id_field="message_id",
                    ids=msg_ids,
                )

        # Step 3: Compute sender contact scores if needed
        sender_scores: dict[int, float] = {}
        if sender_contact_embed_columns:
            # Get unique sender_ids from messages
            sender_ids = list(
                set(
                    int(lg.entries.get("sender_id"))
                    for lg in all_messages
                    if lg.entries.get("sender_id") is not None
                ),
            )
            if sender_ids:
                sender_scores, _ = fetch_scores_for_ids(
                    right_ctx,
                    sender_contact_embed_columns,
                    id_field="contact_id",
                    ids=sender_ids,
                )

        # Step 4: Combine scores for each message
        scored_rows: list[tuple[dict, float]] = []
        for lg in all_messages:
            entries = lg.entries
            mid = entries.get("message_id")
            sender_id = entries.get("sender_id")
            if mid is None:
                continue
            try:
                mid_int = int(mid)
                sender_id_int = int(sender_id) if sender_id is not None else None
            except Exception:
                continue
            # Sum the scores (lower is better for cosine distance)
            total_score = msg_scores.get(mid_int, 0.0)
            if sender_id_int is not None:
                total_score += sender_scores.get(sender_id_int, 0.0)
            scored_rows.append((entries, total_score))

        # Sort by combined score (ascending - lower distance is better)
        scored_rows.sort(key=lambda x: x[1])
        candidate_rows = [row for row, _ in scored_rows[:oversample]]
    else:
        # Receiver-only: rank contacts then pull messages containing them
        top_contacts_limit = max(k * TOP_CONTACTS_FACTOR, TOP_CONTACTS_MIN)
        top_contact_rows, recv_score_key = fetch_top_k_by_terms_with_score(
            right_ctx,
            receiver_contact_embed_columns,
            k=top_contacts_limit,
        )
        contact_scores: dict[int, float] = {}
        for contact_row in top_contact_rows:
            cid = contact_row.get("contact_id")
            if cid is None:
                continue
            try:
                cid_int = int(cid)
            except Exception:
                continue
            try:
                c_score = float(contact_row.get(recv_score_key, 0))
            except Exception:
                c_score = 0.0
            contact_scores[cid_int] = c_score

        msg_to_score: dict[int, float] = {}
        oversample_target = max(k * OVERSAMPLE_FACTOR, RECEIVER_OVERSAMPLE_MIN)
        contact_ids: list[int] = list(contact_scores.keys())
        for i in range(0, len(contact_ids), BATCH_OR_SIZE):
            batch = contact_ids[i : i + BATCH_OR_SIZE]
            if not batch:
                continue
            or_expr = " or ".join(f"{cid} in receiver_ids" for cid in batch)
            rows = unify.get_logs(
                context=left_ctx,
                filter=or_expr,
                from_fields=["message_id", "receiver_ids"],
                limit=oversample_target,
            )
            for lg in rows:
                entries = lg.entries
                mid = entries.get("message_id")
                rids = entries.get("receiver_ids", [])
                if mid is None or not isinstance(rids, list):
                    continue
                try:
                    mid_int = int(mid)
                except Exception:
                    continue
                min_recv = 2.0
                for rid in rids:
                    try:
                        sc = contact_scores.get(int(rid))
                        if sc is not None and sc < min_recv:
                            min_recv = sc
                    except Exception:
                        continue
                prev = msg_to_score.get(mid_int)
                if (prev is None) or (min_recv < prev):
                    msg_to_score[mid_int] = min_recv
            if len(msg_to_score) >= oversample_target:
                break

        candidate_rows = []
        if msg_to_score:
            ids_expr = ", ".join(str(i) for i in msg_to_score.keys())
            rows = unify.get_logs(
                context=left_ctx,
                filter=f"message_id in [{ids_expr}]",
                from_fields=["message_id", "receiver_ids"],
                limit=len(msg_to_score),
            )
            for lg in rows:
                row = dict(lg.entries)
                row["_receiver_only_base"] = 0.0
                candidate_rows.append(row)

    if not receiver_contact_embed_columns:
        results: List[Message] = []
        taken = 0
        msg_field_keys = set(Message.model_fields.keys())
        for row in candidate_rows:
            if taken >= k:
                break
            msg_payload = {k: row.get(k) for k in msg_field_keys if k in row}
            try:
                results.append(Message(**msg_payload))
                taken += 1
            except Exception:
                continue
        return format_contacts_and_messages(self, results)

    # Receiver terms present: compute receiver scores and combine
    receiver_id_set: set[int] = set()
    for row in candidate_rows:
        rids = row.get("receiver_ids", [])
        if isinstance(rids, list):
            for rid in rids:
                try:
                    receiver_id_set.add(int(rid))
                except Exception:
                    continue

    receiver_scores_map, receiver_score_key = fetch_scores_for_ids(
        right_ctx,
        receiver_contact_embed_columns,
        id_field="contact_id",
        ids=sorted(receiver_id_set),
    )

    combined = _aggregate_receiver_min(
        candidate_rows,
        receiver_scores_map,
        base_score_key=candidate_score_key,
    )

    combined.sort(key=lambda t: t[1])
    top_ids = [mid for mid, _ in combined[:k]]
    if not top_ids:
        return format_contacts_and_messages(self, [])

    ids_expr = ", ".join(str(i) for i in top_ids)
    full_rows = unify.get_logs(
        context=left_ctx,
        filter=f"message_id in [{ids_expr}]",
        from_fields=list(Message.model_fields.keys()),
        limit=len(top_ids),
    )
    full_by_id: dict[int, dict] = {}
    for lg in full_rows:
        try:
            mid_val = int(lg.entries.get("message_id"))
        except Exception:
            continue
        full_by_id[mid_val] = dict(lg.entries)

    results: List[Message] = []
    for mid in top_ids:
        payload = full_by_id.get(mid)
        if not payload:
            continue
        try:
            results.append(Message(**payload))
        except Exception:
            continue

    return format_contacts_and_messages(self, results)
