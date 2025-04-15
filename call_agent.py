from vapi_python import Vapi
vapi = Vapi(api_key='31071044-4dc6-454d-8ed9-c6a1851e726a')
assistant = {
  'firstMessage': 'Hey, how are you?',
  'context': 'You are an employee at a drive thru...',
  'model': 'gpt-3.5-turbo',
  'voice': 'jennifer-playht',
  "recordingEnabled": True,
  "interruptionsEnabled": False
}
vapi.start(assistant=assistant)
