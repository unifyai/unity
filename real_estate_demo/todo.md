Problem: 
Frequent responses that are robotic and unnatural, example:
- Thanking the user when automatically filling fields
- Repetitive responses
- Unnatural or robotic tone

Ideas to try:
- re-name `response` field to `phone_utterance`: see if changing the field name helps in anyway
- Add two new actions `PromptUser`, `UpdateUser` (Prompt User will ask them for data or Respond to them while UpdateUser will be mainly used to give them update cueues on what you are doing at the moment)

putting everything in one field might be complicated

okay each turn the agent so decide:
- prompt the user for information
- take an agent_script_action
- end the session/call

Do not start gen while a generation in going on