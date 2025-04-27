SUMMARIZE = """
You will be given a series of exchanges, and you need to summarize these exchanges, based on the following guidance.
{guidance}
Please extract the most important information across all of the exchanges, without preferential treatment to any one of them.
"""

MANAGER = """
Your task is to answer the user question, and you should continue using the tools available until you are satisfied that you either have the correct answer, or you are confident it cannot be answered correctly. All three tools available include pagination with `offset` and `limit` to control the number of returned items and their offset in the list, and all include `filter` which accepts arbitrary Python logical expressions which evaluate to `bool`, and can include any of the relevant `Message` or `Summary` fields in the expressions (depending on the tool).

As a recap, the schemas for contacts, messages and summaries are as follows:

{contact_schema}

{message_schema}

{summary_scheme}

Some example filter expressions (`filter: str`) for the three tools are as follows.

{search_contacts_tool_name}:

- Sender's first name is John:  `filter="first_name == 'John'"`
- email address is gmail: `filter="'@gmail' in email"`
- WhatsApp number is american: `filter="'+1' in whatsapp_number"`
- Surname begins with "L": `filter="surname[0] == 'L'"`
- Flexible logical expressions and nesting. John L or has gmail: `filter="(first_name == 'John' and surname[0] == 'L') or '@gmail' in email"`

{search_messages_tool_name}:

- Sender contact id is even:  `filter="contact_id % 2 == 0"`
- Medium is email: `filter="medium == 'email'"`
- Medium is email or whatsapp message: `filter="medium in ['email', 'whatsapp_message']"`
- Message contains the phrase Hello: `filter="'Hello' in content"`
- Flexible logical expressions and nesting. Email Greeting from contact 0: `filter="(('Hello' in content) or ('Goodbye' in content)) and medium == 'email' and contact_id == 0"`

{search_summaries_tool_name}:

- Summary includes the substrings "sale" and "stapler":  `filter="sale" in summary and "stapler" in summary"`
- Summary includes either exchange id 0 or 1: `filter="0 in exchange_ids or 1 in exchange_ids"`
- Flexible logical expressions and nesting. Exchange id 0 or 1 and "sale" or "stapler" in summary: `filter="(0 in exchange_ids or 1 in exchange_ids) and ("sale" in summary or "stapler" in summary")"`
"""
