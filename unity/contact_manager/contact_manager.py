from typing import List, Dict, Optional

import unify
from .types.contact import Contact
from ..events.event_bus import EventBus


class ContactManager:

    def __init__(self, event_bus: EventBus, *, traced: bool = True) -> None:
        """
        Responsible managing the list of contact details stored upstream.
        """
        self._event_bus = event_bus

        ctxs = unify.get_active_context()
        read_ctx, write_ctx = ctxs["read"], ctxs["write"]
        assert (
            read_ctx == write_ctx
        ), "read and write contexts must be the same when instantiating a TranscriptManager."
        event_bus.register_event_types(["Contacts"])
        self._ctx = event_bus.ctxs["Contacts"]

        # Define tools for ask and update methods
        self._ask_tools: Dict[str, Callable] = {
            self._search_contacts.__name__: self._search_contacts,
        }
        self._update_tools: Dict[str, Callable] = {
            self._create_contact.__name__: self._create_contact,
            self._update_contact.__name__: self._update_contact,
            self._search_contacts.__name__: self._search_contacts,
        }
        # Add tracing
        if traced:
            self = unify.traced(self)

        # Public #
    # -------#
    def ask(
        self,
        text: str,
        *,
        return_reasoning_steps: bool = False,
        parent_chat_context: Optional[List[Dict[str, Any]]] = None,
        clarification_up_q: Optional[asyncio.Queue[str]] = None,
        clarification_down_q: Optional[asyncio.Queue[str]] = None,
    ) -> AsyncToolLoopHandle:
        """
        Ask any question as a text command about contacts.

        Args:
            text (str): The text-based question to answer.
            return_reasoning_steps (bool): Whether to return the reasoning steps along with the answer.
            parent_chat_context (list[dict]): A list of parent context messages to pass down into the tool use loop.
            clarification_up_q (asyncio.Queue[str]): A queue to send clarification questions up to the caller.
            clarification_down_q (asyncio.Queue[str]): A queue to send clarification answers down to the model.

        Returns:
            AsyncToolLoopHandle: A handle to the running conversation.
        """
        client = unify.AsyncUnify(
            "o4-mini@openai",  # Consider making model configurable
            cache=json.loads(os.environ.get("UNIFY_CACHE", "true")),
            traced=json.loads(os.environ.get("UNIFY_TRACED", "true")),
        )
        client.set_system_message(ASK_CONTACTS)

        tools = dict(self._ask_tools)
        if clarification_up_q is not None and clarification_down_q is not None:

            async def request_clarification(question: str) -> str:
                if clarification_up_q is None or clarification_down_q is None:
                    raise RuntimeError(
                        "Clarification queues not properly initialized for ask.",
                    )
                await clarification_up_q.put(question)
                return await clarification_down_q.get()

            tools["request_clarification"] = request_clarification

        handle = start_async_tool_use_loop(
            client,
            text,
            tools,
            parent_chat_context=parent_chat_context,
        )

        if return_reasoning_steps:
            original_result = handle.result

            async def wrapped_result():
                answer = await original_result()
                return answer, client.messages

            handle.result = wrapped_result  # type: ignore

        return handle

    def update(
        self,
        text: str,
        *,
        return_reasoning_steps: bool = False,
        parent_chat_context: Optional[List[Dict[str, Any]]] = None,
        clarification_up_q: Optional[asyncio.Queue[str]] = None,
        clarification_down_q: Optional[asyncio.Queue[str]] = None,
    ) -> AsyncToolLoopHandle:
        """
        Handle any plain-text english command to create or update contacts.

        Args:
            text (str): The text-based request.
            return_reasoning_steps (bool): Whether to return the reasoning steps.
            parent_chat_context (list[dict]): A list of parent context messages.
            clarification_up_q (asyncio.Queue[str]): Queue for clarification questions.
            clarification_down_q (asyncio.Queue[str]): Queue for clarification answers.

        Returns:
            AsyncToolLoopHandle: A handle to the running conversation.
        """
        client = unify.AsyncUnify(
            "o4-mini@openai",
            cache=json.loads(os.environ.get("UNIFY_CACHE", "true")),
            traced=json.loads(os.environ.get("UNIFY_TRACED", "true")),
        )
        client.set_system_message(UPDATE_CONTACTS)

        tools = dict(self._update_tools)
        if clarification_up_q is not None and clarification_down_q is not None:

            async def request_clarification(question: str) -> str:
                if clarification_up_q is None or clarification_down_q is None:
                    raise RuntimeError(
                        "Clarification queues not properly initialized for update.",
                    )
                await clarification_up_q.put(question)
                return await clarification_down_q.get()

            tools["request_clarification"] = request_clarification

        handle = start_async_tool_use_loop(
            client,
            text,
            tools,
            parent_chat_context=parent_chat_context,
        )

        if return_reasoning_steps:
            original_result = handle.result

            async def wrapped_result():
                answer = await original_result()
                return answer, client.messages

            handle.result = wrapped_result  # type: ignore

        return handle

    # Private #
    # --------#

    def _create_contact(
        self,
        *,
        first_name: Optional[str] = None,
        surname: Optional[str] = None,
        email_address: Optional[str] = None,
        phone_number: Optional[str] = None,
        whatsapp_number: Optional[str] = None,
    ) -> int:
        """
        Creates a new contact with the following contact details, as available.

        Args:
            first_name (str): The first name of the contact.
            surname (str): The surname of the contact.
            email_address (str): The email address of the contact.
            phone_number (str): The phone number of the contact.
            whatsapp_number (str): The WhatsApp number of the contact.
        Returns:
            int: The contact_id of the newly created contact.
        """

        # Prune None values
        contact_details = {
            "first_name": first_name,
            "surname": surname,
            "email_address": email_address,
            "phone_number": phone_number,
            "whatsapp_number": whatsapp_number,
        }
        assert any(
            contact_details.values(),
        ), "At least one contact detail must be provided."

        # If it's the first contact, create immediately
        if not unify.get_logs(context=self._ctx):
            return unify.log(
                context=self._ctx,
                **contact_details,
                contact_id=0,
                new=True,
            ).id

        # Verify uniqueness
        for key, value in contact_details.items():
            if key in ["first_name", "surname"] or value is None:
                continue
            logs = unify.get_logs(
                context=self._ctx,
                filter=f"{key} == '{value}'",
            )
            assert (
                len(logs) == 0
            ), f"Invalid, contact with {key} {value} already exists."

        # ToDo: filter only for contact_id once supported in the Python utility function
        logs = unify.get_logs(
            context=self._ctx,
        )
        largest_id = max([lg.entries["contact_id"] for lg in logs])
        this_id = largest_id + 1

        # Create the new contact
        return unify.log(
            context=self._ctx,
            **contact_details,
            contact_id=this_id,
            new=True,
        ).id

    def _update_contact(
        self,
        *,
        contact_id: int,
        first_name: Optional[str] = None,
        surname: Optional[str] = None,
        email_address: Optional[str] = None,
        phone_number: Optional[str] = None,
        whatsapp_number: Optional[str] = None,
    ) -> int:
        """
        Update the contact details of a contact.

        Args:
            contact_id (int): The id of the contact to update.
            first_name (Optional[str]): The first name of the contact.
            surname (Optional[str]): The surname of the contact.
            email_address (Optional[str]): The email address of the contact.
            phone_number (Optional[str]): The phone number of the contact.
            whatsapp_number (Optional[str]): The WhatsApp number of the contact.

        Returns:
            int: The contact_id of the updated contact.
        """
        # Prune None values
        contact_details = {
            "first_name": first_name,
            "surname": surname,
            "email_address": email_address,
            "phone_number": phone_number,
            "whatsapp_number": whatsapp_number,
        }
        assert any(
            contact_details.values(),
        ), "At least one contact detail must be provided."

        # Verify uniqueness
        for key, value in contact_details.items():
            if key in ["first_name", "surname"] or value is None:
                continue
            logs = unify.get_logs(
                context=self._ctx,
                filter=f"{key} == '{value}'",
            )
            assert (
                len(logs) == 0
            ), f"Invalid, contact with {key} {value} already exists."

        # get log id
        logs = unify.get_logs(
            context=self._ctx,
            filter=f"contact_id == {contact_id}",
        )
        assert len(logs) == 1
        log: unify.Log = logs[0]
        log.update_entries(
            **contact_details,
            contact_id=contact_id,
        )

    def _search_contacts(
        self,
        *,
        filter: Optional[str] = None,
        offset: int = 0,
        limit: int = 100,
    ) -> List[Contact]:
        """
        Retrieve contact details, based on flexible filtering for first name, surname, email address, WhatsApp number, phone number, or anything else.

        Args:
            filter (Optional[str]): The filter to apply to the contacts.
            offset (int): The offset to start the retrieval from.
            limit (int): The maximum number of contacts to retrieve.

        Returns:
            List[Contact]: A list of contacts.
        """
        logs = unify.get_logs(
            context=self._ctx,
            filter=filter,
            offset=offset,
            limit=limit,
        )
        return [Contact(**lg.entries) for lg in logs]
