from typing import List, Optional

from pydantic import BaseModel, Field


# thanks to o3 (probably not needed if pred is going to happen outside of livekit)
class FlatSchemaModel(BaseModel):
    """
    Base class that post-processes Pydantic’s JSON-schema so that
    all $refs are expanded in-place and the top-level $defs block
    is removed.  Call `YourModel.model_json_schema()` as usual.
    """

    @classmethod
    def model_json_schema(cls, **kwargs):  # type: ignore[override]
        schema: dict = super().model_json_schema(**kwargs)

        # Pull out the central definitions store (if any)
        definitions = schema.pop("$defs", None)
        if not definitions:
            return schema  # nothing to flatten

        # Recursively walk the schema tree and replace every $ref
        def _inline_refs(node):
            if isinstance(node, dict):
                ref = node.pop("$ref", None)
                if ref:
                    # "#/$defs/Foo"  ->  "Foo"
                    def_key = ref.rsplit("/", 1)[-1]
                    node.update(definitions[def_key])
                    _inline_refs(node)  # the in-lined part may itself contain refs
                else:
                    for value in node.values():
                        _inline_refs(value)
            elif isinstance(node, list):
                for item in node:
                    _inline_refs(item)

        _inline_refs(schema)
        return schema


from typing import List, Literal, Union, Optional
from pydantic import BaseModel, Field


# -------- action variants --------
class SendWhatsAppMessageAction(BaseModel):
    type: Literal["whatsapp"]
    message: str


class SendTelegramMessageAction(BaseModel):
    type: Literal["telegram"]
    message: str


class SendSMSMessageAction(BaseModel):
    type: Literal["sms"]
    message: str


class SendEmailAction(BaseModel):
    type: Literal["email"]
    subject: str
    body: str


# -------- discriminated union --------
# actually, call and non-call modes will probably have some difference between their actions (as in more actions probably for the voice one),
# but for now lets keep them the same
ActionModel = Union[
    SendEmailAction,
    SendWhatsAppMessageAction,
    SendTelegramMessageAction,
    SendSMSMessageAction,
]


# -------- assistant output --------
class CallAssistantOutput(FlatSchemaModel):
    phone_utterance: str = Field(
        ...,
        description="Your response to the user over the phone",
    )
    actions: Optional[List[ActionModel]] = Field(
        ...,
        description="Actions the assistant should perform",
    )


class AssistantOutput(FlatSchemaModel):
    actions: List[ActionModel] = Field(
        ...,
        description="Actions the assistant should perform",
    )
