from __future__ import annotations

from typing import Any, Dict

from unity.actor.environments.base import BaseEnvironment, ToolMetadata
from unity.function_manager.primitives import PRIMITIVE_SOURCES, Primitives


class StateManagerEnvironment(BaseEnvironment):
    """State manager environment backed by `unity.function_manager.primitives.Primitives`.

    Exposes state manager methods like `primitives.contacts.ask(...)` for use inside
    generated plan code.
    """

    def __init__(self, primitives: Primitives):
        self._primitives = primitives

    @property
    def namespace(self) -> str:
        return "primitives"

    def get_instance(self) -> Primitives:
        return self._primitives

    def get_prompt_context(self) -> str:
        """Return Markdown-formatted rules/examples for using state managers."""
        return ""

    def get_tools(self) -> Dict[str, ToolMetadata]:
        # The public surface for state managers is driven by the shared primitives registry
        # (`PRIMITIVE_SOURCES`) to avoid hardcoding manager/method lists in multiple places.
        #
        # IMPORTANT: We are intentionally conservative with purity:
        # - Only clearly read-only methods are treated as pure (cacheable).
        # - Unknown methods default to impure to avoid incorrectly caching side effects.
        pure_methods = {
            "ask",
            "ask_about_file",  # FileManager read-only
            "get",
            "list",
            "search",
            "exists",
            "parse",
            "preview",
            "reduce",  # FileManager read-only
            "filter_files",  # FileManager read-only
            "search_files",  # FileManager read-only
            "visualize",  # FileManager read-only (generates plots, no mutation)
        }

        def _infer_primitives_attr_name(class_path: str) -> str | None:
            class_name = class_path.rsplit(".", 1)[-1]
            # Strip common suffixes used by managers.
            for suffix in ("Manager", "Scheduler", "Searcher"):
                if class_name.endswith(suffix):
                    class_name = class_name[: -len(suffix)]
                    break

            base = class_name[:1].lower() + class_name[1:]

            # Prefer plural if present on Primitives (common convention: contacts, tasks, secrets).
            plural = f"{base}s"
            if hasattr(Primitives, plural):
                return plural
            if hasattr(Primitives, base):
                return base

            # Fallback for irregular cases.
            special = {
                "Task": "tasks",
                "Tasks": "tasks",
                "Contact": "contacts",
                "Transcript": "transcripts",
                "Secret": "secrets",
                "Web": "web",
                "File": "files",
            }
            for k, v in special.items():
                if class_name == k and hasattr(Primitives, v):
                    return v
            return None

        tools: Dict[str, ToolMetadata] = {}
        for class_path, method_names in PRIMITIVE_SOURCES:
            # Skip ComputerPrimitives; those belong to the `computer_primitives` environment.
            if class_path.endswith(".ComputerPrimitives"):
                continue

            manager_attr = _infer_primitives_attr_name(class_path)
            if not manager_attr:
                # If the runtime `Primitives` interface doesn't expose this manager, skip it.
                continue

            for method_name in method_names:
                fq_name = f"{self.namespace}.{manager_attr}.{method_name}"
                tools[fq_name] = ToolMetadata(
                    name=fq_name,
                    is_impure=(method_name not in pure_methods),
                    is_steerable=True,
                    docstring=None,
                    signature=None,
                )

        return tools

    def get_prompt_context(self) -> str:
        """Markdown-formatted guidance for using state-manager primitives in plans."""

        return (
            "### State manager primitives (`primitives.*`)\n"
            "\n"
            "Each manager owns a specific domain of the assistant's durable state. Choose the right manager for your task:\n"
            "\n"
            "**Facts/Policies & Domain Knowledge** → `primitives.knowledge`\n"
            "- **Domain**: Organizational facts, policies, procedures, reference material, documentation, stored information\n"
            "- `.ask(...)`: Query stored knowledge - company policies (return/refund/warranty/HR), procedures, facts, historical records\n"
            "- `.update(...)`: Add/change facts, ingest structured data, update policies\n"
            "- `.refactor(...)`: Restructure knowledge schemas (advanced)\n"
            '- **Use when**: Questions about company policies, operational procedures, reference docs, "what is our X policy?", "summarize Y procedure"\n'
            '- **Examples**: "What\'s our return policy?", "Summarize onboarding procedure", "Office hours?", "Warranty terms for X?"\n'
            "\n"
            "**People & Relationships** → `primitives.contacts`\n"
            "- **Domain**: People, organizations, contact records (names, emails, phones, roles, locations)\n"
            "- `.ask(...)`: Find contacts by name/email/attribute, query relationships, get contact details\n"
            "- `.update(...)`: Create, edit, delete, or merge contact records\n"
            '- **Use when**: Questions about specific people, contact info, "who is X?", "find contact in Y location"\n'
            '- **Examples**: "Who is our contact at Acme Corp?", "Find Alice\'s email", "Contacts in Berlin?"\n'
            "\n"
            "**Durable Work & Tracking** → `primitives.tasks`\n"
            "- **Domain**: Task management, work queues, assignments, deadlines, priorities\n"
            "- `.ask(...)`: Query task status, what's due/scheduled, assignments, priorities\n"
            "- `.update(...)`: Create, edit, delete, or reorder tasks (NOT for starting work)\n"
            "- `.execute(...)`: Start durable, tracked execution (use this to run tasks, not `.update(...)`)\n"
            '- **Use when**: Questions about tasks/work items, "what\'s due?", "tasks assigned to X?", "high-priority items?"\n'
            '- **Examples**: "What tasks are due today?", "Show Alice\'s open tasks", "List high-priority items"\n'
            "\n"
            "**Conversation History** → `primitives.transcripts`\n"
            "- **Domain**: Past messages, conversation history, communication records (chat/SMS/email)\n"
            "- `.ask(...)`: Search messages, find what someone said, retrieve conversation context\n"
            '- **Use when**: Questions about past communications, "what did X say?", "last message about Y?", "conversation with Z?"\n'
            '- **Examples**: "What did Bob say yesterday?", "Last SMS with Alice?", "Messages mentioning budget?"\n'
            "\n"
            "**Time-Sensitive & Web** → `primitives.web`\n"
            '- **Domain**: Current events, real-time information, external research, "today/latest/now" queries\n'
            "- `.ask(...)`: Web search for current information, news, weather, public data\n"
            "- **Use when**: Questions requiring up-to-date external information, current events, weather, news\n"
            '- **Examples**: "Weather in Berlin today?", "Latest AI news?", "Current stock price?", "Recent announcements?"\n'
            "\n"
            "**Function & Task Guidance** → `primitives.guidance`\n"
            "- **Domain**: Execution instructions, runbooks, how-to guides for functions/tasks\n"
            "- `.ask(...)`: Query execution instructions, runbooks, best practices for specific operations\n"
            "- `.update(...)`: Create, edit, or delete guidance entries linked to functions\n"
            "- **Use when**: Questions about HOW to execute something, operational runbooks, incident response procedures\n"
            '- **Examples**: "How do I handle DB failover?", "Incident response for API outage?"\n'
            "\n"
            "**Files & Documents** → `primitives.files`\n"
            "- **Domain**: Received/downloaded files, document parsing, file metadata\n"
            "- `.ask(...)`: Query about specific files, parse document contents, extract information from files\n"
            "- `.organize(...)`: File management operations\n"
            "- **Use when**: Questions about specific files/documents the user shared or system has\n"
            '- **Examples**: "Parse the attached PDF", "What\'s in document X?", "Find files about Y"\n'
            "\n"
            "**Credentials & Secrets** → `primitives.secrets`\n"
            "- **Domain**: API keys, passwords, tokens, credentials\n"
            "- `.ask(...)`: Get metadata/placeholders only (never returns actual secret values)\n"
            "- `.update(...)`: Create, edit, or delete secrets\n"
            "- **Use when**: Managing credentials, API keys, secrets (rarely used in plans)\n"
            "\n"
            "**Manager Selection Priorities**:\n"
            "1. **knowledge** takes priority for organizational policies, procedures, company facts, internal documentation\n"
            "2. **transcripts** for historical communications (what was said/written)\n"
            "3. **contacts** for people/relationship information\n"
            "4. **tasks** for work items, deadlines, assignments\n"
            "5. **web** for current external information (weather, news, real-time data)\n"
            "6. **guidance** for execution instructions and runbooks\n"
            "7. **files** when dealing with specific documents\n"
            "\n"
            "**General Rules**:\n"
            "- All manager calls return a steerable handle; await `.result()` to get the final answer\n"
            "- If a manager asks for clarification, wait for the user response and answer via the handle's API\n"
            "- Prefer `ask(...)` for read-only queries; only use `update(...)`/`execute(...)` when mutations are needed\n"
            "- When in doubt between managers, prefer the most specific domain match\n"
        )

    async def capture_state(self) -> Dict[str, Any]:
        """State manager \"state\" is primarily evidenced via return values."""
        return {
            "type": "return_value",
            "note": "State manager evidence is captured via function return values.",
        }
