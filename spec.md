# Unity Technical Specification

## ComsManager

Manages all text-based and voice-based communcation, whether it be with the user, another person, or another AI assistant. When you call your assistant, it is the ComsManager that picks up the phone, when you send a text, the coms manager parses it and decided whether or not to respond.

### Tools

#### TranscriptManager

The `ComsManager` can send text-based queries to the `TranscriptManager`, asking anything related to the prior communications.

#### KnowledgeManager

The `ComsManager` can send text-based queries to the `KnowledgeManager`, asking anything related to useful knowledge the assistant has accumulated.

### Publishes To

#### Transcript Queue

Every time the user sends a message or ends their turn during a call, the `transcript_q` is populated with all messages *since the last user message*.

### Subscribed To

#### Task Completion Queue

Every time a task is completed, the `ComsManager` is informed, enabling this to be expressed to the user if appropriate.
