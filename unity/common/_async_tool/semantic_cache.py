import unify
import json

_USER_MESSAGE_EMBEDDING_FIELD_NAME = "user_message_emb"
_EMBED_MODEL = "text-embedding-3-small"


def store_tool_trajectory(user_message, tool_trajectory):
    from unity import ASSISTANT_CONTEXT

    store_context = f"{ASSISTANT_CONTEXT}/Cache"

    # Ensure context exists
    context_exist = store_context in unify.get_contexts(prefix=store_context)
    if not context_exist:
        unify.create_context(store_context)

    log_id = unify.log(
        context=store_context,
        user_message=user_message,
        tool_trajectory=json.dumps(tool_trajectory),
    )

    embed_expr = f"embed({{logs:user_message}}, model='{_EMBED_MODEL}')"
    unify.create_derived_logs(
        context=store_context,
        key=_USER_MESSAGE_EMBEDDING_FIELD_NAME,
        equation=embed_expr,
        referenced_logs={"logs": [log_id.id]},
    )

    return log_id


def get_tool_trajectory(user_message):
    from unity import ASSISTANT_CONTEXT

    store_context = f"{ASSISTANT_CONTEXT}/Cache"

    # Ensure context exists
    context_exist = store_context in unify.get_contexts(prefix=store_context)
    if not context_exist:
        unify.create_context(store_context)

    logs = unify.get_logs(context=store_context, return_ids_only=True)
    unify.create_derived_logs(
        key="temp_score",
        equation=f"cosine({{logs:user_message_emb}}, embed('{user_message}', model={_EMBED_MODEL}))",
        # TODO: weird behavior?
        # equation=f"cosine(embed('{user_message}', model={_EMBED_MODEL}), {{logs:user_message_emb}})",
        # equation=f"cosine({{logs:user_message_emb}}, {{logs:user_message_emb}})",
        referenced_logs={
            "logs": logs,
        },
        context=store_context,
    )

    threshold = 0.4
    limit = 1
    logs = unify.get_logs(
        filter=f"temp_score <= {threshold}",
        sorting={
            "temp_score": "descending",
        },
        exclude_fields=["user_message_emb", "temp_score"],
        limit=limit,
        context=store_context,
    )

    return logs
