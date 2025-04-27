SUMMARIZE = """
You will be given a series of exchanges, and you need to summarize these exchanges, based on the following guidance.
{guidance}
Please extract the most important information across all of the exchanges, without preferential treatment to any one of them.
"""

MANAGER = """
Your task is to ansewr the user question, and you should continue using the tools available until you are satisfied that you either have the correct answer, or you are confident it cannot be answered correctly. Both tools available include pagination with `offset` and `limit` to control the number of returned items and their offset in the list, and both include `filter` which accepts arbitrary Python logical expressions which evaluate to `bool`, and can include any of the relevant `Message` or `Summary` fields in the expressions (depending on the tool).

As a recap, the schemas for messages and summaries are as follows:

{message_schema}

{summary_scheme}

Some example filter expressions for {search_message_tool_name}:

`` -> A

`` -> B

`` -> C

`` -> D

Some example filter expressions for {search_message_tool_name}:

`` -> A

`` -> B

`` -> C

`` -> D


"""
