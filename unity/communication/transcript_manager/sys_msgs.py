import json

from ...communication.transcript_manager.transcript_manager import TranscriptManager
from ...communication.types.contact import Contact
from ...communication.types.message import Message
from ..types.message_exchange_summary import MessageExchangeSummary
from ...communication.transcript_manager.transcript_manager import TranscriptManager

SUMMARIZE = """
You will be given a series of exchanges, and you need to summarize these exchanges, based on the following guidance.
{guidance}
Please extract the most important information across all of the exchanges, without preferential treatment to any one of them.
"""

ANSWER = f"""
Your task is to answer the user question, and you should continue using the tools available until you are satisfied that you either have the correct answer, or you are confident it cannot be answered correctly. Firstly, you can summarize any exchange or group of exchanges, creating an overall explanatory paragraphs of said exchange(s). This tool is straightforward to use. You can also search contacts, messages and summaries (which you may or may not have created yourself). All three search tools include pagination with `offset` and `limit` to control the number of returned items and their offset in the list, and all include `filter` which accepts arbitrary Python logical expressions which evaluate to `bool`, and can include any of the relevant `Message` or `Summary` fields in the expressions (depending on the tool).

As a recap, the schemas for contacts, messages and summaries are as follows:

{json.dumps(Contact.model_json_schema(), indent=4)}

{json.dumps(Message.model_json_schema(), indent=4)}

{json.dumps(MessageExchangeSummary.model_json_schema(), indent=4)}

Available tools:
• {TranscriptManager.summarize.__name__}(exchange_ids, guidance?): summarise one or more exchanges.
• {TranscriptManager._search_contacts.__name__.lstrip("_")}(filter?, offset=0, limit=100) → List[Contact] – flexible boolean filtering.
• {TranscriptManager._search_messages.__name__.lstrip("_")}(filter?, offset=0, limit=100) → List[Message] – flexible boolean filtering.
• {TranscriptManager._search_summaries.__name__.lstrip("_")}(filter?, offset=0, limit=100) → List[MessageExchangeSummary] – flexible boolean filtering.
• {TranscriptManager._nearest_messages.__name__.lstrip("_")}(text: str, k: int = 10) → List[Message] – returns the top-k messages semantically similar to the given text.

Example usage:
# Find top-3 messages semantically similar to "banking and budgeting"
nearest_messages(text="banking and budgeting", k=3)

Some example filter expressions (`filter: str`) for the tools are as follows.

{TranscriptManager._search_contacts.__name__.lstrip("_")}:

- Sender's first name is John:  `filter="first_name == 'John'"`
- email address is gmail: `filter="'@gmail' in email"`
- WhatsApp number is american: `filter="'+1' in whatsapp_number"`
- Surname begins with "L": `filter="surname[0] == 'L'"`
- Flexible logical expressions and nesting. John L or has gmail: `filter="(first_name == 'John' and surname[0] == 'L') or '@gmail' in email"`

{TranscriptManager._search_messages.__name__.lstrip("_")}:

- Sender contact id is even:  `filter="contact_id % 2 == 0"`
- Medium is email: `filter="medium == 'email'"`
- Medium is email or whatsapp message: `filter="medium in ['email', 'whatsapp_message']"`
- Message contains the phrase Hello: `filter="'Hello' in content"`
- Flexible logical expressions and nesting. Email Greeting from contact 0: `filter="(('Hello' in content) or ('Goodbye' in content)) and medium == 'email' and contact_id == 0"`

{TranscriptManager._search_summaries.__name__.lstrip("_")}:

- Summary includes the substrings "sale" and "stapler":  `filter="sale" in summary and "stapler" in summary"`
- Summary includes either exchange id 0 or 1: `filter="0 in exchange_ids or 1 in exchange_ids"`
- Flexible logical expressions and nesting. Exchange id 0 or 1 and "sale" or "stapler" in summary: `filter="(0 in exchange_ids or 1 in exchange_ids) and ("sale" in summary or "stapler" in summary")"`

Remember that while filter-based search is useful for exact matches, the `nearest_messages` tool is more effective for finding semantically related content when you don't know the exact wording.
"""
