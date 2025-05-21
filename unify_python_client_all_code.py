Project Path: unify

Source Tree:

```
unify
├── universal_api
│   ├── clients
│   │   ├── __init__.py
│   │   ├── multi_llm.py
│   │   ├── helpers.py
│   │   ├── base.py
│   │   └── uni_llm.py
│   ├── types
│   │   ├── __init__.py
│   │   └── prompt.py
│   ├── __init__.py
│   ├── utils
│   │   ├── custom_api_keys.py
│   │   ├── __init__.py
│   │   ├── endpoint_metrics.py
│   │   ├── custom_endpoints.py
│   │   ├── supported_endpoints.py
│   │   ├── queries.py
│   │   └── credits.py
│   ├── casting.py
│   ├── chatbot.py
│   └── usage.py
├── __init__.py
├── utils
│   ├── __init__.py
│   ├── map.py
│   ├── _caching.py
│   ├── helpers.py
│   └── _requests.py
├── logging
│   ├── __init__.py
│   ├── utils
│   │   ├── datasets.py
│   │   ├── __init__.py
│   │   ├── async_logger.py
│   │   ├── logs.py
│   │   ├── artifacts.py
│   │   ├── contexts.py
│   │   ├── projects.py
│   │   └── compositions.py
│   ├── dataset.py
│   └── logs.py
└── interfaces
    └── utils

```

`/Users/yushaarif/Unify/unify/unify/universal_api/clients/__init__.py`:

```py
from . import base, multi_llm, uni_llm
from .base import _Client
from .multi_llm import AsyncMultiUnify, MultiUnify, _MultiClient
from .uni_llm import AsyncUnify, Unify, _UniClient

```

`/Users/yushaarif/Unify/unify/unify/universal_api/clients/multi_llm.py`:

```py
# global
import abc
import asyncio
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type, Union

import requests
# local
import unify
# noinspection PyProtectedMember
from openai._types import Headers, Query
from openai.types.chat import (ChatCompletion, ChatCompletionMessageParam,
                               ChatCompletionStreamOptionsParam,
                               ChatCompletionToolChoiceOptionParam,
                               ChatCompletionToolParam)
from pydantic import BaseModel
from typing_extensions import Self
from unify import BASE_URL
from unify.utils import _requests
# noinspection PyProtectedMember
from unify.utils.helpers import _default, _validate_api_key

from ..clients import AsyncUnify, _Client, _UniClient
from ..utils.endpoint_metrics import Metrics


class _MultiClient(_Client, abc.ABC):
    def __init__(
        self,
        endpoints: Optional[Union[str, Iterable[str]]] = None,
        *,
        system_message: Optional[str] = None,
        messages: Optional[
            Union[
                List[ChatCompletionMessageParam],
                Dict[str, List[ChatCompletionMessageParam]],
            ]
        ] = None,
        frequency_penalty: Optional[float] = None,
        logit_bias: Optional[Dict[str, int]] = None,
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None,
        max_completion_tokens: Optional[int] = None,
        n: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        response_format: Optional[Union[Type[BaseModel], Dict[str, str]]] = None,
        seed: Optional[int] = None,
        stop: Union[Optional[str], List[str]] = None,
        temperature: Optional[float] = 1.0,
        top_p: Optional[float] = None,
        tools: Optional[Iterable[ChatCompletionToolParam]] = None,
        tool_choice: Optional[ChatCompletionToolChoiceOptionParam] = None,
        parallel_tool_calls: Optional[bool] = None,
        # platform arguments
        use_custom_keys: bool = False,
        tags: Optional[List[str]] = None,
        drop_params: Optional[bool] = True,
        region: Optional[str] = None,
        log_query_body: Optional[bool] = True,
        log_response_body: Optional[bool] = True,
        api_key: Optional[str] = None,
        # python client arguments
        stateful: bool = False,
        return_full_completion: bool = False,
        traced: bool = False,
        cache: Union[bool, str] = None,
        local_cache: bool = True,
        # passthrough arguments
        extra_headers: Optional[Headers] = None,
        extra_query: Optional[Query] = None,
        **kwargs,
    ) -> None:
        """Initialize the Multi LLM Unify client.

        Args:
            endpoints: A single endpoint name or a list of endpoint names, with each name
            in OpenAI API format: <model_name>@<provider_name>. Defaults to None.

            system_message: An optional string containing the system message. This
            always appears at the beginning of the list of messages.

            messages: A list of messages comprising the conversation so far. This will
            be appended to the system_message if it is not None, and any user_message
            will be appended if it is not None.

            frequency_penalty: Number between -2.0 and 2.0. Positive values penalize new
            tokens based on their existing frequency in the text so far, decreasing the
            model's likelihood to repeat the same line verbatim.

            logit_bias: Modify the likelihood of specified tokens appearing in the
            completion. Accepts a JSON object that maps tokens (specified by their token
            ID in the tokenizer) to an associated bias value from -100 to 100.
            Mathematically, the bias is added to the logits generated by the model prior
            to sampling. The exact effect will vary per model, but values between -1 and
            1 should decrease or increase likelihood of selection; values like -100 or
            100 should result in a ban or exclusive selection of the relevant token.

            logprobs: Whether to return log probabilities of the output tokens or not.
            If true, returns the log probabilities of each output token returned in the
            content of message.

            top_logprobs: An integer between 0 and 20 specifying the number of most
            likely tokens to return at each token position, each with an associated log
            probability. logprobs must be set to true if this parameter is used.

            max_completion_tokens: The maximum number of tokens that can be generated in
            the chat completion. The total length of input tokens and generated tokens
            is limited by the model's context length. Defaults to the provider's default
            max_completion_tokens when the value is None.

            n: How many chat completion choices to generate for each input message. Note
            that you will be charged based on the number of generated tokens across all
            of the choices. Keep n as 1 to minimize costs.

            presence_penalty: Number between -2.0 and 2.0. Positive values penalize new
            tokens based on whether they appear in the text so far, increasing the
            model's likelihood to talk about new topics.

            response_format: An object specifying the format that the model must output.
            Setting to `{ "type": "json_schema", "json_schema": {...} }` enables
            Structured Outputs which ensures the model will match your supplied JSON
            schema. Learn more in the Structured Outputs guide. Setting to
            `{ "type": "json_object" }` enables JSON mode, which ensures the message the
            model generates is valid JSON.

            seed: If specified, a best effort attempt is made to sample
            deterministically, such that repeated requests with the same seed and
            parameters should return the same result. Determinism is not guaranteed, and
            you should refer to the system_fingerprint response parameter to monitor
            changes in the backend.

            stop: Up to 4 sequences where the API will stop generating further tokens.

            temperature:  What sampling temperature to use, between 0 and 2.
            Higher values like 0.8 will make the output more random,
            while lower values like 0.2 will make it more focused and deterministic.
            It is generally recommended to alter this or top_p, but not both.
            Defaults to the provider's default max_completion_tokens when the value is
            None.

            top_p: An alternative to sampling with temperature, called nucleus sampling,
            where the model considers the results of the tokens with top_p probability
            mass. So 0.1 means only the tokens comprising the top 10% probability mass
            are considered. Generally recommended to alter this or temperature, but not
            both.

            tools: A list of tools the model may call. Currently, only functions are
            supported as a tool. Use this to provide a list of functions the model may
            generate JSON inputs for. A max of 128 functions are supported.

            tool_choice: Controls which (if any) tool is called by the
            model. none means the model will not call any tool and instead generates a
            message. auto means the model can pick between generating a message or
            calling one or more tools. required means the model must call one or more
            tools. Specifying a particular tool via
            `{ "type": "function", "function": {"name": "my_function"} }`
            forces the model to call that tool.
            none is the default when no tools are present. auto is the default if tools
            are present.

            parallel_tool_calls: Whether to enable parallel function calling during tool
            use.

            use_custom_keys:  Whether to use custom API keys or our unified API keys
            with the backend provider.

            tags: Arbitrary number of tags to classify this API query as needed. Helpful
            for generally grouping queries across tasks and users, for logging purposes.

            drop_params: Whether or not to drop unsupported OpenAI params by the
            provider you’re using.

            region: A string used to represent the region where the endpoint is
            accessed. Only relevant for on-prem deployments with certain providers like
            `vertex-ai`, `aws-bedrock` and `azure-ml`, where the endpoint is being
            accessed through a specified region.

            log_query_body: Whether to log the contents of the query json body.

            log_response_body: Whether to log the contents of the response json body.

            stateful:  Whether the conversation history is preserved within the messages
            of this client. If True, then history is preserved. If False, then this acts
            as a stateless client, and message histories must be managed by the user.

            return_full_completion: If False, only return the message content
            chat_completion.choices[0].message.content.strip(" ") from the OpenAI
            return. Otherwise, the full response chat_completion is returned.
            Defaults to False.

            traced: Whether to trace the generate method.

            cache: If True, then the arguments will be stored in a local cache file, and
            any future calls with identical arguments will read from the cache instead
            of running the LLM query. If "write" then the cache will only be written
            to, if "read" then the cache will be read from if a cache is available but
            will not write, and if "read-only" then the argument must be present in the
            cache, else an exception will be raised. Finally, an appending "-closest"
            will read the closest match from the cache, and overwrite it if cache writing
            is enabled. This argument only has any effect when stream=False.

            extra_headers: Additional "passthrough" headers for the request which are
            provider-specific, and are not part of the OpenAI standard. They are handled
            by the provider-specific API.

            extra_query: Additional "passthrough" query parameters for the request which
            are provider-specific, and are not part of the OpenAI standard. They are
            handled by the provider-specific API.

            kwargs: Additional "passthrough" JSON properties for the body of the
            request, which are provider-specific, and are not part of the OpenAI
            standard. They will be handled by the provider-specific API.

        Raises:
            UnifyError: If the API key is missing.
        """
        self._base_constructor_args = dict(
            system_message=system_message,
            messages=messages,
            frequency_penalty=frequency_penalty,
            logit_bias=logit_bias,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            max_completion_tokens=max_completion_tokens,
            n=n,
            presence_penalty=presence_penalty,
            response_format=response_format,
            seed=seed,
            stop=stop,
            stream=False,
            stream_options=None,
            temperature=temperature,
            top_p=top_p,
            tools=tools,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
            # platform arguments
            use_custom_keys=use_custom_keys,
            tags=tags,
            drop_params=drop_params,
            region=region,
            log_query_body=log_query_body,
            log_response_body=log_response_body,
            api_key=api_key,
            # python client arguments
            stateful=stateful,
            return_full_completion=return_full_completion,
            traced=traced,
            cache=cache,
            local_cache=local_cache,
            # passthrough arguments
            extra_headers=extra_headers,
            extra_query=extra_query,
            **kwargs,
        )
        super().__init__(**self._base_constructor_args)
        self._constructor_args = dict(
            endpoints=endpoints,
            **self._base_constructor_args,
        )
        if isinstance(endpoints, str):
            endpoints = [endpoints]
        else:
            endpoints = list(endpoints)
        self._api_key = _validate_api_key(api_key)
        self._endpoints = endpoints
        self._client_class = AsyncUnify
        self._clients = self._create_clients(endpoints)

    def _create_clients(self, endpoints: List[str]) -> Dict[str, AsyncUnify]:
        return {
            endpoint: self._client_class(
                endpoint,
                system_message=self.system_message,
                messages=self.messages,
                frequency_penalty=self.frequency_penalty,
                logit_bias=self.logit_bias,
                logprobs=self.logprobs,
                top_logprobs=self.top_logprobs,
                max_completion_tokens=self.max_completion_tokens,
                n=self.n,
                presence_penalty=self.presence_penalty,
                response_format=self.response_format,
                seed=self.seed,
                stop=self.stop,
                temperature=self.temperature,
                top_p=self.top_p,
                tools=self.tools,
                tool_choice=self.tool_choice,
                parallel_tool_calls=self.parallel_tool_calls,
                # platform arguments
                use_custom_keys=self.use_custom_keys,
                tags=self.tags,
                drop_params=self.drop_params,
                region=self.region,
                log_query_body=self.log_query_body,
                log_response_body=self.log_response_body,
                api_key=self._api_key,
                # python client arguments
                stateful=self.stateful,
                return_full_completion=self.return_full_completion,
                cache=self.cache,
                # passthrough arguments
                extra_headers=self.extra_headers,
                extra_query=self.extra_query,
                **self.extra_body,
            )
            for endpoint in endpoints
        }

    def add_endpoints(
        self,
        endpoints: Union[List[str], str],
        ignore_duplicates: bool = True,
    ) -> Self:
        """
        Add extra endpoints to be queried for each call to generate.

        Args:
            endpoints: The extra endpoints to add.

            ignore_duplicates: Whether or not to ignore duplicate endpoints passed.

        Returns:
            This client, useful for chaining inplace calls.
        """
        if isinstance(endpoints, str):
            endpoints = [endpoints]
        # remove duplicates
        if ignore_duplicates:
            endpoints = [
                endpoint for endpoint in endpoints if endpoint not in self._endpoints
            ]
        elif len(self._endpoints + endpoints) != len(set(self._endpoints + endpoints)):
            raise Exception(
                "at least one of the provided endpoints to add {}"
                "was already set present in the endpoints {}."
                "Set ignore_duplicates to True to ignore errors like this".format(
                    endpoints,
                    self._endpoints,
                ),
            )
        # update endpoints
        self._endpoints = self._endpoints + endpoints
        # create new clients
        self._clients.update(self._create_clients(endpoints))
        return self

    def remove_endpoints(
        self,
        endpoints: Union[List[str], str],
        ignore_missing: bool = True,
    ) -> Self:
        """
        Remove endpoints from the current list, which are queried for each call to
        generate.

        Args:
            endpoints: The extra endpoints to add.

            ignore_missing: Whether or not to ignore endpoints passed which are not
            currently present in the client endpoint list.

        Returns:
            This client, useful for chaining inplace calls.
        """
        if isinstance(endpoints, str):
            endpoints = [endpoints]
        # remove irrelevant
        if ignore_missing:
            endpoints = [
                endpoint for endpoint in endpoints if endpoint in self._endpoints
            ]
        elif len(self._endpoints) != len(set(self._endpoints + endpoints)):
            raise Exception(
                "at least one of the provided endpoints to remove {}"
                "was not present in the current endpoints {}."
                "Set ignore_missing to True to ignore errors like this".format(
                    endpoints,
                    self._endpoints,
                ),
            )
        # update endpoints and clients
        for endpoint in endpoints:
            self._endpoints.remove(endpoint)
            del self._clients[endpoint]
        return self

    def get_credit_balance(self) -> Union[float, None]:
        """
        Get the remaining credits left on your account.

        Returns:
            The remaining credits on the account if successful, otherwise None.
        Raises:
            BadRequestError: If there was an HTTP error.
            ValueError: If there was an error parsing the JSON response.
        """
        url = f"{BASE_URL}/credits"
        headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }
        try:
            response = _requests.get(url, headers=headers, timeout=10)
            if response.status_code != 200:
                raise Exception(response.json())
            return response.json()["credits"]
        except requests.RequestException as e:
            raise Exception("There was an error with the request.") from e
        except (KeyError, ValueError) as e:
            raise ValueError("Error parsing JSON response.") from e

    # Read-only Properties #
    # ---------------------#

    def _get_metrics(self) -> Dict[str, Metrics]:
        return {
            ep: unify.get_endpoint_metrics(ep, api_key=self._api_key)[0]
            for ep in self._endpoints
        }

    @property
    def input_cost(self) -> Dict[str, float]:
        return {ep: metrics.input_cost for ep, metrics in self._get_metrics().items()}

    @property
    def output_cost(self) -> Dict[str, float]:
        return {ep: metrics.output_cost for ep, metrics in self._get_metrics().items()}

    @property
    def ttft(self) -> Dict[str, float]:
        return {ep: metrics.ttft for ep, metrics in self._get_metrics().items()}

    @property
    def itl(self) -> Dict[str, float]:
        return {ep: metrics.itl for ep, metrics in self._get_metrics().items()}

    # Settable Properties #
    # --------------------#

    @property
    def endpoints(self) -> Tuple[str, ...]:
        """
        Get the current tuple of endpoints.

        Returns:
            The tuple of endpoints.
        """
        return tuple(self._endpoints)

    @property
    def clients(self) -> Dict[str, _UniClient]:
        """
        Get the current dictionary of clients, with endpoint names as keys and
        Unify or AsyncUnify instances as values.

        Returns:
            The dictionary of clients.
        """
        return self._clients

    # Representation #
    # ---------------#

    def __repr__(self):
        return "{}(endpoints={})".format(self.__class__.__name__, self._endpoints)

    def __str__(self):
        return "{}(endpoints={})".format(self.__class__.__name__, self._endpoints)

    # Generate #
    # ---------#

    def generate(
        self,
        arg0: Optional[Union[str, List[Union[str, Tuple[Any], Dict[str, Any]]]]] = None,
        /,
        system_message: Optional[str] = None,
        messages: Optional[
            Union[
                List[ChatCompletionMessageParam],
                Dict[str, List[ChatCompletionMessageParam]],
            ]
        ] = None,
        *,
        frequency_penalty: Optional[float] = None,
        logit_bias: Optional[Dict[str, int]] = None,
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None,
        max_completion_tokens: Optional[int] = None,
        n: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        response_format: Optional[Union[Type[BaseModel], Dict[str, str]]] = None,
        seed: Optional[int] = None,
        stop: Union[Optional[str], List[str]] = None,
        stream: Optional[bool] = None,
        stream_options: Optional[ChatCompletionStreamOptionsParam] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        tools: Optional[Iterable[ChatCompletionToolParam]] = None,
        tool_choice: Optional[ChatCompletionToolChoiceOptionParam] = None,
        parallel_tool_calls: Optional[bool] = None,
        # platform arguments
        use_custom_keys: Optional[bool] = None,
        tags: Optional[List[str]] = None,
        drop_params: Optional[bool] = None,
        region: Optional[str] = None,
        log_query_body: Optional[bool] = None,
        log_response_body: Optional[bool] = None,
        # python client arguments
        stateful: Optional[bool] = None,
        return_full_completion: Optional[bool] = None,
        cache: Optional[Union[bool, str]] = None,
        # passthrough arguments
        extra_headers: Optional[Headers] = None,
        extra_query: Optional[Query] = None,
        **kwargs,
    ):
        """Generate a ChatCompletion response for the specified endpoint,
        from the provided query parameters.

        Args:
            arg0: A string containing the user message, or a list containing the inputs
            to send to each of the LLMs, in the format of str (user message), tuple
            (all-positional) or dict (all keyword).

            system_message: An optional string containing the system message. This
            always appears at the beginning of the list of messages.

            messages: A list of messages comprising the conversation so far, or
            optionally a dictionary of such messages, with clients as the keys in the
            case of multi-llm clients. This will be appended to the system_message if it
            is not None, and any user_message will be appended if it is not None.

            frequency_penalty: Number between -2.0 and 2.0. Positive values penalize new
            tokens based on their existing frequency in the text so far, decreasing the
            model's likelihood to repeat the same line verbatim.

            logit_bias: Modify the likelihood of specified tokens appearing in the
            completion. Accepts a JSON object that maps tokens (specified by their token
            ID in the tokenizer) to an associated bias value from -100 to 100.
            Mathematically, the bias is added to the logits generated by the model prior
            to sampling. The exact effect will vary per model, but values between -1 and
            1 should decrease or increase likelihood of selection; values like -100 or
            100 should result in a ban or exclusive selection of the relevant token.

            logprobs: Whether to return log probabilities of the output tokens or not.
            If true, returns the log probabilities of each output token returned in the
            content of message.

            top_logprobs: An integer between 0 and 20 specifying the number of most
            likely tokens to return at each token position, each with an associated log
            probability. logprobs must be set to true if this parameter is used.

            max_completion_tokens: The maximum number of tokens that can be generated in
            the chat completion. The total length of input tokens and generated tokens
            is limited by the model's context length. Defaults value is None. Uses the
            provider's default max_completion_tokens when None is explicitly passed.

            n: How many chat completion choices to generate for each input message. Note
            that you will be charged based on the number of generated tokens across all
            of the choices. Keep n as 1 to minimize costs.

            presence_penalty: Number between -2.0 and 2.0. Positive values penalize new
            tokens based on whether they appear in the text so far, increasing the
            model's likelihood to talk about new topics.

            response_format: An object specifying the format that the model must output.
            Setting to `{ "type": "json_schema", "json_schema": {...} }` enables
            Structured Outputs which ensures the model will match your supplied JSON
            schema. Learn more in the Structured Outputs guide. Setting to
            `{ "type": "json_object" }` enables JSON mode, which ensures the message the
            model generates is valid JSON.

            seed: If specified, a best effort attempt is made to sample
            deterministically, such that repeated requests with the same seed and
            parameters should return the same result. Determinism is not guaranteed, and
            you should refer to the system_fingerprint response parameter to monitor
            changes in the backend.

            stop: Up to 4 sequences where the API will stop generating further tokens.

            stream: If True, generates content as a stream. If False, generates content
            as a single response. Defaults to False.

            stream_options: Options for streaming response. Only set this when you set
            stream: true.

            temperature:  What sampling temperature to use, between 0 and 2.
            Higher values like 0.8 will make the output more random,
            while lower values like 0.2 will make it more focused and deterministic.
            It is generally recommended to alter this or top_p, but not both.
            Default value is 1.0. Defaults to the provider's default temperature when
            None is explicitly passed.

            top_p: An alternative to sampling with temperature, called nucleus sampling,
            where the model considers the results of the tokens with top_p probability
            mass. So 0.1 means only the tokens comprising the top 10% probability mass
            are considered. Generally recommended to alter this or temperature, but not
            both.

            tools: A list of tools the model may call. Currently, only functions are
            supported as a tool. Use this to provide a list of functions the model may
            generate JSON inputs for. A max of 128 functions are supported.

            tool_choice: Controls which (if any) tool is called by the
            model. none means the model will not call any tool and instead generates a
            message. auto means the model can pick between generating a message or
            calling one or more tools. required means the model must call one or more
            tools. Specifying a particular tool via
            `{ "type": "function", "function": {"name": "my_function"} }`
            forces the model to call that tool.
            none is the default when no tools are present. auto is the default if tools
            are present.

            parallel_tool_calls: Whether to enable parallel function calling during tool
            use.

            stateful:  Whether the conversation history is preserved within the messages
            of this client. If True, then history is preserved. If False, then this acts
            as a stateless client, and message histories must be managed by the user.

            use_custom_keys:  Whether to use custom API keys or our unified API keys
            with the backend provider. Defaults to False.

            tags: Arbitrary number of tags to classify this API query as needed. Helpful
            for generally grouping queries across tasks and users, for logging purposes.

            drop_params: Whether or not to drop unsupported OpenAI params by the
            provider you’re using.

            region: A string used to represent the region where the endpoint is
            accessed. Only relevant for on-prem deployments with certain providers like
            `vertex-ai`, `aws-bedrock` and `azure-ml`, where the endpoint is being
            accessed through a specified region.

            log_query_body: Whether to log the contents of the query json body.

            log_response_body: Whether to log the contents of the response json body.

            stateful:  Whether the conversation history is preserved within the messages
            of this client. If True, then history is preserved. If False, then this acts
            as a stateless client, and message histories must be managed by the user.

            return_full_completion: If False, only return the message content
            chat_completion.choices[0].message.content.strip(" ") from the OpenAI
            return. Otherwise, the full response chat_completion is returned.
            Defaults to False.

            cache: If True, then the arguments will be stored in a local cache file, and
            any future calls with identical arguments will read from the cache instead
            of running the LLM query. If "write" then the cache will only be written
            to, if "read" then the cache will be read from if a cache is available but
            will not write, and if "read-only" then the argument must be present in the
            cache, else an exception will be raised. Finally, an appending "-closest"
            will read the closest match from the cache, and overwrite it if cache writing
            is enabled. This argument only has any effect when stream=False.

            extra_headers: Additional "passthrough" headers for the request which are
            provider-specific, and are not part of the OpenAI standard. They are handled
            by the provider-specific API.

            extra_query: Additional "passthrough" query parameters for the request which
            are provider-specific, and are not part of the OpenAI standard. They are
            handled by the provider-specific API.

            kwargs: Additional "passthrough" JSON properties for the body of the
            request, which are provider-specific, and are not part of the OpenAI
            standard. They will be handled by the provider-specific API.

        Returns:
            If stream is True, returns a generator yielding chunks of content.
            If stream is False, returns a single string response.

        Raises:
            UnifyError: If an error occurs during content generation.
        """
        system_message = _default(system_message, self._system_message)
        messages = _default(messages, self._messages)
        stateful = _default(stateful, self._stateful)
        if messages:
            # system message only added once at the beginning
            if isinstance(arg0, str):
                if isinstance(messages, dict):
                    messages = {
                        k: v + [{"role": "user", "content": arg0}]
                        for k, v in messages.items()
                    }
                else:
                    messages += [{"role": "user", "content": arg0}]
        else:
            messages = list()
            if system_message is not None:
                messages += [{"role": "system", "content": system_message}]
            if isinstance(arg0, str):
                messages += [{"role": "user", "content": arg0}]
            self._messages = messages
        return_full_completion = (
            True
            if _default(tools, self._tools)
            else _default(return_full_completion, self._return_full_completion)
        )
        ret = self._generate(
            messages=messages,
            frequency_penalty=_default(frequency_penalty, self._frequency_penalty),
            logit_bias=_default(logit_bias, self._logit_bias),
            logprobs=_default(logprobs, self._logprobs),
            top_logprobs=_default(top_logprobs, self._top_logprobs),
            max_completion_tokens=_default(
                max_completion_tokens,
                self._max_completion_tokens,
            ),
            n=_default(n, self._n),
            presence_penalty=_default(presence_penalty, self._presence_penalty),
            response_format=_default(response_format, self._response_format),
            seed=_default(_default(seed, self._seed), unify.get_seed()),
            stop=_default(stop, self._stop),
            stream=_default(stream, self._stream),
            stream_options=_default(stream_options, self._stream_options),
            temperature=_default(temperature, self._temperature),
            top_p=_default(top_p, self._top_p),
            tools=_default(tools, self._tools),
            tool_choice=_default(tool_choice, self._tool_choice),
            parallel_tool_calls=_default(
                parallel_tool_calls,
                self._parallel_tool_calls,
            ),
            # platform arguments
            use_custom_keys=_default(use_custom_keys, self._use_custom_keys),
            tags=_default(tags, self._tags),
            drop_params=_default(drop_params, self._drop_params),
            region=_default(region, self._region),
            log_query_body=_default(log_query_body, self._log_query_body),
            log_response_body=_default(log_response_body, self._log_response_body),
            # python client arguments
            return_full_completion=return_full_completion,
            cache=_default(cache, self._cache),
            # passthrough arguments
            extra_headers=_default(extra_headers, self._extra_headers),
            extra_query=_default(extra_query, self._extra_query),
            **{**self._extra_body, **kwargs},
        )
        if stateful:
            if return_full_completion:
                msg = [ret.choices[0].message.model_dump()]
            else:
                msg = [{"role": "assistant", "content": ret}]
            if self._messages is None:
                self._messages = []
            self._messages += msg
        return ret


class MultiUnify(_MultiClient):
    async def _async_gen(
        self,
        messages: Optional[
            Union[
                List[ChatCompletionMessageParam],
                Dict[str, List[ChatCompletionMessageParam]],
            ]
        ] = None,
        *,
        frequency_penalty: Optional[float] = None,
        logit_bias: Optional[Dict[str, int]] = None,
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None,
        max_completion_tokens: Optional[int] = None,
        n: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        response_format: Optional[Union[Type[BaseModel], Dict[str, str]]] = None,
        seed: Optional[int] = None,
        stop: Union[Optional[str], List[str]] = None,
        temperature: Optional[float] = 1.0,
        top_p: Optional[float] = None,
        tools: Optional[Iterable[ChatCompletionToolParam]] = None,
        tool_choice: Optional[ChatCompletionToolChoiceOptionParam] = None,
        parallel_tool_calls: Optional[bool] = None,
        # platform arguments
        use_custom_keys: bool = False,
        tags: Optional[List[str]] = None,
        drop_params: Optional[bool] = True,
        region: Optional[str] = None,
        log_query_body: Optional[bool] = True,
        log_response_body: Optional[bool] = True,
        # python client arguments
        return_full_completion: bool = False,
        local_cache: bool = True,
        # passthrough arguments
        extra_headers: Optional[Headers] = None,
        extra_query: Optional[Query] = None,
        **kwargs,
    ) -> Union[Union[str, ChatCompletion], Dict[str, Union[str, ChatCompletion]]]:
        kw = dict(
            messages=messages,
            max_completion_tokens=max_completion_tokens,
            stop=stop,
            temperature=temperature,
            frequency_penalty=frequency_penalty,
            logit_bias=logit_bias,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            n=n,
            presence_penalty=presence_penalty,
            response_format=response_format,
            seed=seed,
            top_p=top_p,
            tools=tools,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
            use_custom_keys=use_custom_keys,
            tags=tags,
            drop_params=drop_params,
            region=region,
            log_query_body=log_query_body,
            log_response_body=log_response_body,
            return_full_completion=return_full_completion,
            extra_headers=extra_headers,
            extra_query=extra_query,
            **kwargs,
        )
        multi_message = isinstance(kw["messages"], dict)
        kw_ = {k: v for k, v in kw.items() if v is not None}
        responses = dict()
        for endpoint, client in self._clients.items():
            these_kw = kw_.copy()
            if multi_message:
                these_kw["messages"] = these_kw["messages"][endpoint]
            responses[endpoint] = await client.generate(**these_kw)
        return responses[self._endpoints[0]] if len(self._endpoints) == 1 else responses

    async def _multi_inp_gen(
        self,
        multi_input: List[Union[str, Tuple[Any], Dict[str, Any]]],
        **kwargs,
    ) -> List[Union[str, ChatCompletion]]:
        assert isinstance(multi_input, list), (
            f"Expected multi_kwargs to be a list, "
            f"but found {multi_input} of type {type(multi_input)}."
        )
        assert all(
            type(multi_input[0]) is type(i) for i in multi_input
        ), "all entries in the list of inputs must be of the same type."
        if isinstance(multi_input[0], str):
            coroutines = [self._async_gen(s, **kwargs) for s in multi_input]
        elif isinstance(multi_input[0], tuple):
            coroutines = [self._async_gen(*a, **kwargs) for a in multi_input]
        elif isinstance(multi_input[0], dict):
            coroutines = [self._async_gen(**{**kwargs, **kw}) for kw in multi_input]
        else:
            raise Exception(
                f"Invalid format for first argument in list, expected either str, "
                f"tuple or dict but found "
                f"{multi_input[0]} of type {type(multi_input[0])}.",
            )
        return await asyncio.gather(*coroutines)

    def _multi_inp_generate(
        self,
        *args,
        **kwargs,
    ) -> List[Union[str, ChatCompletion]]:
        """
        Perform multiple generations to multiple inputs asynchronously, based on the
        list keywords arguments passed in.
        """
        return asyncio.run(self._multi_inp_gen(*args, **kwargs))

    def _generate(  # noqa: WPS234, WPS211
        self,
        *args,
        **kwargs,
    ) -> Union[
        Union[
            Union[str, ChatCompletion],
            List[Union[str, ChatCompletion]],
        ],
        Dict[
            str,
            Union[
                Union[str, ChatCompletion],
                List[Union[str, ChatCompletion]],
            ],
        ],
    ]:
        # refresh the openai client before doing a new event loop
        for key, client in self._clients.items():
            client._client = client._get_client()
        if args and isinstance(args[0], list):
            return self._multi_inp_generate(*args, **kwargs)
        return asyncio.run(
            self._async_gen(
                *args,
                **kwargs,
            ),
        )

    def to_async_client(self):
        """
        Return an asynchronous version of the client (`AsyncMultiUnify` instance), with
        the exact same configuration as this synchronous (`MultiUnify`) client.

        Returns:
            An `AsyncMultiUnify` instance with the same configuration as this `MultiUnify`
            instance.
        """
        return AsyncMultiUnify(**self._constructor_args)


class AsyncMultiUnify(_MultiClient):
    async def _generate(  # noqa: WPS234, WPS211
        self,
        messages: Optional[
            Union[
                List[ChatCompletionMessageParam],
                Dict[str, List[ChatCompletionMessageParam]],
            ]
        ] = None,
        *,
        frequency_penalty: Optional[float] = None,
        logit_bias: Optional[Dict[str, int]] = None,
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None,
        max_completion_tokens: Optional[int] = None,
        n: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        response_format: Optional[Union[Type[BaseModel], Dict[str, str]]] = None,
        seed: Optional[int] = None,
        stop: Union[Optional[str], List[str]] = None,
        temperature: Optional[float] = 1.0,
        top_p: Optional[float] = None,
        tools: Optional[Iterable[ChatCompletionToolParam]] = None,
        tool_choice: Optional[ChatCompletionToolChoiceOptionParam] = None,
        parallel_tool_calls: Optional[bool] = None,
        # platform arguments
        use_custom_keys: bool = False,
        tags: Optional[List[str]] = None,
        drop_params: Optional[bool] = True,
        region: Optional[str] = None,
        log_query_body: Optional[bool] = True,
        log_response_body: Optional[bool] = True,
        # python client arguments
        return_full_completion: bool = False,
        local_cache: bool = True,
        # passthrough arguments
        extra_headers: Optional[Headers] = None,
        extra_query: Optional[Query] = None,
        **kwargs,
    ) -> Dict[str, str]:
        kw = dict(
            messages=messages,
            max_completion_tokens=max_completion_tokens,
            stop=stop,
            temperature=temperature,
            frequency_penalty=frequency_penalty,
            logit_bias=logit_bias,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            n=n,
            presence_penalty=presence_penalty,
            response_format=response_format,
            seed=seed,
            top_p=top_p,
            tools=tools,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
            use_custom_keys=use_custom_keys,
            tags=tags,
            drop_params=drop_params,
            region=region,
            log_query_body=log_query_body,
            log_response_body=log_response_body,
            return_full_completion=return_full_completion,
            local_cache=local_cache,
            extra_headers=extra_headers,
            extra_query=extra_query,
            **kwargs,
        )
        multi_message = isinstance(messages, dict)
        kw = {k: v for k, v in kw.items() if v is not None}
        responses = dict()
        for endpoint, client in self._clients.items():
            these_kw = kw.copy()
            if multi_message:
                these_kw["messages"] = these_kw["messages"][endpoint]
            responses[endpoint] = await client.generate(**these_kw)
        return responses

    def to_sync_client(self):
        """
        Return a synchronous version of the client (`MultiUnify` instance), with the
        exact same configuration as this asynchronous (`AsyncMultiUnify`) client.

        Returns:
            A `MultiUnify` instance with the same configuration as this `AsyncMultiUnify`
            instance.
        """
        return MultiUnify(**self._constructor_args)

```

`/Users/yushaarif/Unify/unify/unify/universal_api/clients/helpers.py`:

```py
import unify

# Helpers


def _is_custom_endpoint(endpoint: str):
    _, provider = endpoint.split("@")
    return "custom" in provider


def _is_local_endpoint(endpoint: str):
    _, provider = endpoint.split("@")
    return provider == "local"


def _is_fallback_provider(provider: str, api_key: str = None):
    public_providers = unify.list_providers(api_key=api_key)
    return all(p in public_providers for p in provider.split("->"))


def _is_fallback_model(model: str, api_key: str = None):
    public_models = unify.list_models(api_key=api_key)
    return all(p in public_models for p in model.split("->"))


def _is_fallback_endpoint(endpoint: str, api_key: str = None):
    public_endpoints = unify.list_endpoints(api_key=api_key)
    return all(e in public_endpoints for e in endpoint.split("->"))


def _is_meta_provider(provider: str, api_key: str = None):
    public_providers = unify.list_providers(api_key=api_key)
    if "skip_providers:" in provider:
        skip_provs = provider.split("skip_providers:")[-1].split("|")[0]
        for prov in skip_provs.split(","):
            if prov.strip() not in public_providers:
                return False
        chnk0, chnk1 = provider.split("skip_providers:")
        chnk2 = "|".join(chnk1.split("|")[1:])
        provider = "".join([chnk0, chnk2])
    if "providers:" in provider:
        provs = provider.split("providers:")[-1].split("|")[0]
        for prov in provs.split(","):
            if prov.strip() not in public_providers:
                return False
        chnk0, chnk1 = provider.split("providers:")
        chnk2 = "|".join(chnk1.split("|")[1:])
        provider = "".join([chnk0, chnk2])
        if provider[-1] == "|":
            provider = provider[:-1]
    public_models = unify.list_models(api_key=api_key)
    if "skip_models:" in provider:
        skip_mods = provider.split("skip_models:")[-1].split("|")[0]
        for md in skip_mods.split(","):
            if md.strip() not in public_models:
                return False
        chnk0, chnk1 = provider.split("skip_models:")
        chnk2 = "|".join(chnk1.split("|")[1:])
        provider = "".join([chnk0, chnk2])
    if "models:" in provider:
        mods = provider.split("models:")[-1].split("|")[0]
        for md in mods.split(","):
            if md.strip() not in public_models:
                return False
        chnk0, chnk1 = provider.split("models:")
        chnk2 = "|".join(chnk1.split("|")[1:])
        provider = "".join([chnk0, chnk2])
    meta_providers = (
        (
            "highest-quality",
            "lowest-time-to-first-token",
            "lowest-inter-token-latency",
            "lowest-input-cost",
            "lowest-output-cost",
            "lowest-cost",
            "lowest-ttft",
            "lowest-itl",
            "lowest-ic",
            "lowest-oc",
            "highest-q",
            "lowest-t",
            "lowest-i",
            "lowest-c",
        )
        + (
            "quality",
            "time-to-first-token",
            "inter-token-latency",
            "input-cost",
            "output-cost",
            "cost",
        )
        + (
            "q",
            "ttft",
            "itl",
            "ic",
            "oc",
            "t",
            "i",
            "c",
        )
    )
    operators = ("<", ">", "=", "|", ".", ":")
    for s in meta_providers + operators:
        provider = provider.replace(s, "")
    return all(c.isnumeric() for c in provider)


# Checks


def _is_valid_endpoint(endpoint: str, api_key: str = None):
    if endpoint == "user-input":
        return True
    if _is_fallback_endpoint(endpoint, api_key):
        return True
    model, provider = endpoint.split("@")
    if _is_valid_provider(provider) and _is_valid_model(model):
        return True
    if endpoint in unify.list_endpoints(api_key=api_key):
        return True
    if _is_custom_endpoint(endpoint) or _is_local_endpoint(endpoint):
        return True
    return False


def _is_valid_provider(provider: str, api_key: str = None):
    if _is_meta_provider(provider):
        return True
    if provider in unify.list_providers(api_key=api_key):
        return True
    if _is_fallback_provider(provider):
        return True
    if provider == "local" or "custom" in provider:
        return True
    return False


def _is_valid_model(model: str, custom_or_local: bool = False, api_key: str = None):
    if custom_or_local:
        return True
    if model in unify.list_models(api_key=api_key):
        return True
    if _is_fallback_model(model):
        return True
    if model == "router":
        return True
    return False


# Assertions


def _assert_is_valid_endpoint(endpoint: str, api_key: str = None):
    assert _is_valid_endpoint(endpoint, api_key), f"{endpoint} is not a valid endpoint"


def _assert_is_valid_provider(provider: str, api_key: str = None):
    assert _is_valid_provider(provider, api_key), f"{provider} is not a valid provider"


def _assert_is_valid_model(
    model: str,
    custom_or_local: bool = False,
    api_key: str = None,
):
    assert _is_valid_model(
        model,
        custom_or_local,
        api_key,
    ), f"{model} is not a valid model"

```

`/Users/yushaarif/Unify/unify/unify/universal_api/clients/base.py`:

```py
# global
import copy
from abc import ABC, abstractmethod
from typing import Dict, Iterable, List, Mapping, Optional, Type, Union

import requests
# noinspection PyProtectedMember
from openai._types import Body, Headers, Query
from openai.types.chat import (ChatCompletionMessageParam,
                               ChatCompletionStreamOptionsParam,
                               ChatCompletionToolChoiceOptionParam,
                               ChatCompletionToolParam)
from pydantic import BaseModel, create_model
from typing_extensions import Self
# local
from unify import BASE_URL
from unify.utils import _requests
# noinspection PyProtectedMember
from unify.utils.helpers import _validate_api_key


class _Client(ABC):
    """Base Abstract class for interacting with the Unify chat completions endpoint."""

    def __init__(
        self,
        *,
        system_message: Optional[str],
        messages: Optional[
            Union[
                List[ChatCompletionMessageParam],
                Dict[str, List[ChatCompletionMessageParam]],
            ]
        ],
        frequency_penalty: Optional[float],
        logit_bias: Optional[Dict[str, int]],
        logprobs: Optional[bool],
        top_logprobs: Optional[int],
        max_completion_tokens: Optional[int],
        n: Optional[int],
        presence_penalty: Optional[float],
        response_format: Optional[Union[Type[BaseModel], Dict[str, str]]],
        seed: Optional[int],
        stop: Union[Optional[str], List[str]],
        stream: Optional[bool],
        stream_options: Optional[ChatCompletionStreamOptionsParam],
        temperature: Optional[float],
        top_p: Optional[float],
        tools: Optional[Iterable[ChatCompletionToolParam]],
        tool_choice: Optional[ChatCompletionToolChoiceOptionParam],
        parallel_tool_calls: Optional[bool],
        # platform arguments
        use_custom_keys: bool,
        tags: Optional[List[str]],
        drop_params: Optional[bool],
        region: Optional[str] = None,
        log_query_body: Optional[bool],
        log_response_body: Optional[bool],
        api_key: Optional[str],
        # python client arguments
        stateful: bool,
        return_full_completion: bool,
        traced: bool,
        cache: Union[bool, str],
        local_cache: bool,
        # passthrough arguments
        extra_headers: Optional[Headers],
        extra_query: Optional[Query],
        **kwargs,
    ) -> None:  # noqa: DAR101, DAR401

        # initial values
        self._api_key = _validate_api_key(api_key)
        self._system_message = None
        self._messages = None
        self._frequency_penalty = None
        self._logit_bias = None
        self._logprobs = None
        self._top_logprobs = None
        self._max_completion_tokens = None
        self._n = None
        self._presence_penalty = None
        self._response_format = None
        self._seed = None
        self._stop = None
        self._stream = None
        self._stream_options = None
        self._temperature = None
        self._top_p = None
        self._tools = None
        self._tool_choice = None
        self._parallel_tool_calls = None
        self._use_custom_keys = None
        self._tags = None
        self._drop_params = None
        self._region = None
        self._log_query_body = None
        self._log_response_body = None
        self._stateful = None
        self._return_full_completion = None
        self._traced = None
        self._cache = None
        self._local_cache = None
        self._extra_headers = None
        self._extra_query = None
        self._extra_body = None

        # set based on arguments
        self.set_system_message(system_message)
        self.set_messages(messages)
        self.set_frequency_penalty(frequency_penalty)
        self.set_logit_bias(logit_bias)
        self.set_logprobs(logprobs)
        self.set_top_logprobs(top_logprobs)
        self.set_max_completion_tokens(max_completion_tokens)
        self.set_n(n)
        self.set_presence_penalty(presence_penalty)
        self.set_response_format(response_format)
        self.set_seed(seed)
        self.set_stop(stop)
        self.set_stream(stream)
        self.set_stream_options(stream_options)
        self.set_temperature(temperature)
        self.set_top_p(top_p)
        self.set_tools(tools)
        self.set_tool_choice(tool_choice)
        self.set_parallel_tool_calls(parallel_tool_calls)
        # platform arguments
        self.set_use_custom_keys(use_custom_keys)
        self.set_tags(tags)
        self.set_drop_params(drop_params)
        self.set_region(region)
        self.set_log_query_body(log_query_body)
        self.set_log_response_body(log_response_body)
        # python client arguments
        self.set_stateful(stateful)
        self.set_return_full_completion(return_full_completion)
        self.set_traced(traced)
        self.set_cache(cache)
        self.set_local_cache(local_cache)
        # passthrough arguments
        self.set_extra_headers(extra_headers)
        self.set_extra_query(extra_query)
        self.set_extra_body(kwargs)

    # Properties #
    # -----------#

    @property
    def system_message(self) -> Optional[str]:
        """
        Get the default system message, if set.

        Returns:
            The default system message.
        """
        return self._system_message

    @property
    def messages(
        self,
    ) -> Optional[
        Union[
            List[ChatCompletionMessageParam],
            Dict[str, List[ChatCompletionMessageParam]],
        ]
    ]:
        """
        Get the default messages, if set.

        Returns:
            The default messages.
        """
        return self._messages

    @property
    def frequency_penalty(self) -> Optional[float]:
        """
        Get the default frequency penalty, if set.

        Returns:
            The default frequency penalty.
        """
        return self._frequency_penalty

    @property
    def logit_bias(self) -> Optional[Dict[str, int]]:
        """
        Get the default logit bias, if set.

        Returns:
            The default logit bias.
        """
        return self._logit_bias

    @property
    def logprobs(self) -> Optional[bool]:
        """
        Get the default logprobs, if set.

        Returns:
            The default logprobs.
        """
        return self._logprobs

    @property
    def top_logprobs(self) -> Optional[int]:
        """
        Get the default top logprobs, if set.

        Returns:
            The default top logprobs.
        """
        return self._top_logprobs

    @property
    def max_completion_tokens(self) -> Optional[int]:
        """
        Get the default max tokens, if set.

        Returns:
            The default max tokens.
        """
        return self._max_completion_tokens

    @property
    def n(self) -> Optional[int]:
        """
        Get the default n, if set.

        Returns:
            The default n value.
        """
        return self._n

    @property
    def presence_penalty(self) -> Optional[float]:
        """
        Get the default presence penalty, if set.

        Returns:
            The default presence penalty.
        """
        return self._presence_penalty

    @property
    def response_format(self) -> Optional[Union[Type[BaseModel], Dict[str, str]]]:
        """
        Get the default response format, if set.

        Returns:
            The default response format.
        """
        return self._response_format

    @property
    def seed(self) -> Optional[int]:
        """
        Get the default seed value, if set.

        Returns:
            The default seed value.
        """
        return self._seed

    @property
    def stop(self) -> Union[Optional[str], List[str]]:
        """
        Get the default stop value, if set.

        Returns:
            The default stop value.
        """
        return self._stop

    @property
    def stream(self) -> Optional[bool]:
        """
        Get the default stream bool, if set.

        Returns:
            The default stream bool.
        """
        return self._stream

    @property
    def stream_options(self) -> Optional[ChatCompletionStreamOptionsParam]:
        """
        Get the default stream options, if set.

        Returns:
            The default stream options.
        """
        return self._stream_options

    @property
    def temperature(self) -> Optional[float]:
        """
        Get the default temperature, if set.

        Returns:
            The default temperature.
        """
        return self._temperature

    @property
    def top_p(self) -> Optional[float]:
        """
        Get the default top p value, if set.

        Returns:
            The default top p value.
        """
        return self._top_p

    @property
    def tools(self) -> Optional[Iterable[ChatCompletionToolParam]]:
        """
        Get the default tools, if set.

        Returns:
            The default tools.
        """
        return self._tools

    @property
    def tool_choice(self) -> Optional[ChatCompletionToolChoiceOptionParam]:
        """
        Get the default tool choice, if set.

        Returns:
            The default tool choice.
        """
        return self._tool_choice

    @property
    def parallel_tool_calls(self) -> Optional[bool]:
        """
        Get the default parallel tool calls bool, if set.

        Returns:
            The default parallel tool calls bool.
        """
        return self._parallel_tool_calls

    @property
    def use_custom_keys(self) -> bool:
        """
        Get the default use custom keys bool, if set.

        Returns:
            The default use custom keys bool.
        """
        return self._use_custom_keys

    @property
    def tags(self) -> Optional[List[str]]:
        """
        Get the default tags, if set.

        Returns:
            The default tags.
        """
        return self._tags

    @property
    def drop_params(self) -> Optional[bool]:
        """
        Get the default drop_params bool, if set.

        Returns:
            The default drop_params bool.
        """
        return self._drop_params

    @property
    def region(self) -> Optional[str]:
        """
        Get the default region, if set.

        Returns:
            The default region.
        """
        return self._region

    @property
    def log_query_body(self) -> Optional[bool]:
        """
        Get the default log query body bool, if set.

        Returns:
            The default log query body bool.
        """
        return self._log_query_body

    @property
    def log_response_body(self) -> Optional[bool]:
        """
        Get the default log response body bool, if set.

        Returns:
            The default log response body bool.
        """
        return self._log_response_body

    @property
    def stateful(self) -> bool:
        """
        Get the default stateful bool, if set.

        Returns:
            The default stateful bool.
        """
        return self._stateful

    @property
    def return_full_completion(self) -> bool:
        """
        Get the default return full completion bool.

        Returns:
            The default return full completion bool.
        """
        return self._return_full_completion

    @property
    def traced(self) -> bool:
        """
        Get the default traced bool.

        Returns:
            The default traced bool.
        """
        return self._traced

    @property
    def cache(self) -> bool:
        """
        Get default the cache bool.

        Returns:
            The default cache bool.
        """
        return self._cache

    @property
    def extra_headers(self) -> Optional[Headers]:
        """
        Get the default extra headers, if set.

        Returns:
            The default extra headers.
        """
        return self._extra_headers

    @property
    def extra_query(self) -> Optional[Query]:
        """
        Get the default extra query, if set.

        Returns:
            The default extra query.
        """
        return self._extra_query

    @property
    def extra_body(self) -> Optional[Mapping[str, str]]:
        """
        Get the default extra body, if set.

        Returns:
            The default extra body.
        """
        return self._extra_body

    # Setters #
    # --------#

    def set_system_message(self, value: str) -> Self:
        """
        Set the default system message.  # noqa: DAR101.

        Args:
            value: The default system message.

        Returns:
            This client, useful for chaining inplace calls.
        """
        self._system_message = value
        if self._messages is None or self._messages == []:
            self._messages = [
                {
                    "role": "system",
                    "content": value,
                },
            ]
        elif self._messages[0]["role"] != "system":
            self._messages = [
                {
                    "role": "system",
                    "content": value,
                },
            ] + self._messages
        else:
            self._messages[0] = {
                "role": "system",
                "content": value,
            }
        return self

    def set_messages(
        self,
        value: Union[
            List[ChatCompletionMessageParam],
            Dict[str, List[ChatCompletionMessageParam]],
        ],
    ) -> Self:
        """
        Set the default messages.  # noqa: DAR101.

        Args:
            value: The default messages.

        Returns:
            This client, useful for chaining inplace calls.
        """
        self._messages = value
        if isinstance(value, list) and value and value[0]["role"] == "system":
            self.set_system_message(value[0]["content"])
        return self

    def append_messages(
        self,
        value: Union[
            List[ChatCompletionMessageParam],
            Dict[str, List[ChatCompletionMessageParam]],
        ],
    ) -> Self:
        """
        Append to the default messages.  # noqa: DAR101.

        Args:
            value: The messages to append to the default.

        Returns:
            This client, useful for chaining inplace calls.
        """
        if self._messages is None:
            self._messages = []
        self._messages += value
        return self

    def set_frequency_penalty(self, value: float) -> Self:
        """
        Set the default frequency penalty.  # noqa: DAR101.

        Args:
            value: The default frequency penalty.

        Returns:
            This client, useful for chaining inplace calls.
        """
        self._frequency_penalty = value
        return self

    def set_logit_bias(self, value: Dict[str, int]) -> Self:
        """
        Set the default logit bias.  # noqa: DAR101.

        Args:
            value: The default logit bias.

        Returns:
            This client, useful for chaining inplace calls.
        """
        self._logit_bias = value
        return self

    def set_logprobs(self, value: bool) -> Self:
        """
        Set the default logprobs.  # noqa: DAR101.

        Args:
            value: The default logprobs.

        Returns:
            This client, useful for chaining inplace calls.
        """
        self._logprobs = value
        return self

    def set_top_logprobs(self, value: int) -> Self:
        """
        Set the default top logprobs.  # noqa: DAR101.

        Args:
            value: The default top logprobs.

        Returns:
            This client, useful for chaining inplace calls.
        """
        self._top_logprobs = value
        return self

    def set_max_completion_tokens(self, value: int) -> Self:
        """
        Set the default max tokens.  # noqa: DAR101.

        Args:
            value: The default max tokens.

        Returns:
            This client, useful for chaining inplace calls.
        """
        self._max_completion_tokens = value
        return self

    def set_n(self, value: int) -> Self:
        """
        Set the default n value.  # noqa: DAR101.

        Args:
            value: The default n value.

        Returns:
            This client, useful for chaining inplace calls.
        """
        self._n = value
        return self

    def set_presence_penalty(self, value: float) -> Self:
        """
        Set the default presence penalty.  # noqa: DAR101.

        Args:
            value: The default presence penalty.

        Returns:
            This client, useful for chaining inplace calls.
        """
        self._presence_penalty = value
        return self

    def set_response_format(
        self,
        value: Optional[Union[Type[BaseModel], Dict[str, str]]],
    ) -> Self:
        """
        Set the default response format.  # noqa: DAR101.

        Args:
            value: The default response format.

        Returns:
            This client, useful for chaining inplace calls.
        """
        self._response_format = value
        return self

    def set_seed(self, value: Optional[int]) -> Self:
        """
        Set the default seed value.  # noqa: DAR101.

        Args:
            value: The default seed value.

        Returns:
            This client, useful for chaining inplace calls.
        """
        self._seed = value
        return self

    def set_stop(self, value: Union[str, List[str]]) -> Self:
        """
        Set the default stop value.  # noqa: DAR101.

        Args:
            value: The default stop value.

        Returns:
            This client, useful for chaining inplace calls.
        """
        self._stop = value
        return self

    def set_stream(self, value: bool) -> Self:
        """
        Set the default stream bool.  # noqa: DAR101.

        Args:
            value: The default stream bool.

        Returns:
            This client, useful for chaining inplace calls.
        """
        self._stream = value
        return self

    def set_stream_options(self, value: ChatCompletionStreamOptionsParam) -> Self:
        """
        Set the default stream options.  # noqa: DAR101.

        Args:
            value: The default stream options.

        Returns:
            This client, useful for chaining inplace calls.
        """
        self._stream_options = value
        return self

    def set_temperature(self, value: float) -> Self:
        """
        Set the default temperature.  # noqa: DAR101.

        Args:
            value: The default temperature.

        Returns:
            This client, useful for chaining inplace calls.
        """
        self._temperature = value
        return self

    def set_top_p(self, value: float) -> Self:
        """
        Set the default top p value.  # noqa: DAR101.

        Args:
            value: The default top p value.

        Returns:
            This client, useful for chaining inplace calls.
        """
        self._top_p = value
        return self

    def set_tools(self, value: Iterable[ChatCompletionToolParam]) -> Self:
        """
        Set the default tools.  # noqa: DAR101.

        Args:
            value: The default tools.

        Returns:
            This client, useful for chaining inplace calls.
        """
        self._tools = value
        return self

    def set_tool_choice(self, value: ChatCompletionToolChoiceOptionParam) -> Self:
        """
        Set the default tool choice.  # noqa: DAR101.

        Args:
            value: The default tool choice.

        Returns:
            This client, useful for chaining inplace calls.
        """
        self._tool_choice = value
        return self

    def set_parallel_tool_calls(self, value: bool) -> Self:
        """
        Set the default parallel tool calls bool.  # noqa: DAR101.

        Args:
            value: The default parallel tool calls bool.

        Returns:
            This client, useful for chaining inplace calls.
        """
        self._parallel_tool_calls = value
        return self

    def set_use_custom_keys(self, value: bool) -> Self:
        """
        Set the default use custom keys bool.  # noqa: DAR101.

        Args:
            value: The default use custom keys bool.

        Returns:
            This client, useful for chaining inplace calls.
        """
        self._use_custom_keys = value
        return self

    def set_tags(self, value: List[str]) -> Self:
        """
        Set the default tags.  # noqa: DAR101.

        Args:
            value: The default tags.

        Returns:
            This client, useful for chaining inplace calls.
        """
        self._tags = value
        return self

    def set_drop_params(self, value: bool) -> Self:
        """
        Set the default drop params bool.  # noqa: DAR101.

        Args:
            value: The default drop params bool.

        Returns:
            This client, useful for chaining inplace calls.
        """
        self._drop_params = value
        return self

    def set_region(self, value: str) -> Self:
        """
        Set the default region.  # noqa: DAR101.

        Args:
            value: The default region.

        Returns:
            This client, useful for chaining inplace calls.
        """
        self._region = value
        return self

    def set_log_query_body(self, value: bool) -> Self:
        """
        Set the default log query body bool.  # noqa: DAR101.

        Args:
            value: The default log query body bool.

        Returns:
            This client, useful for chaining inplace calls.
        """
        self._log_query_body = value
        return self

    def set_log_response_body(self, value: bool) -> Self:
        """
        Set the default log response body bool.  # noqa: DAR101.

        Args:
            value: The default log response body bool.

        Returns:
            This client, useful for chaining inplace calls.
        """
        self._log_response_body = value
        return self

    def set_stateful(self, value: bool) -> Self:
        """
        Set the default stateful bool.  # noqa: DAR101.

        Args:
            value: The default stateful bool.

        Returns:
            This client, useful for chaining inplace calls.
        """
        self._stateful = value
        return self

    def set_return_full_completion(self, value: bool) -> Self:
        """
        Set the default return full completion bool.  # noqa: DAR101.

        Args:
            value: The default return full completion bool.

        Returns:
            This client, useful for chaining inplace calls.
        """
        self._return_full_completion = value
        return self

    # noinspection PyAttributeOutsideInit
    def set_traced(self, value: bool) -> Self:
        """
        Set the default traced bool.  # noqa: DAR101.

        Args:
            value: The default traced bool.

        Returns:
            This client, useful for chaining inplace calls.
        """
        self._traced = value
        return self

    def set_cache(self, value: bool) -> Self:
        """
        Set the default cache bool.  # noqa: DAR101.

        Args:
            value: The default cache bool.

        Returns:
            This client, useful for chaining inplace calls.
        """
        self._cache = value
        return self

    def set_local_cache(self, value: bool) -> Self:
        """
        Set the default local cache bool.  # noqa: DAR101.

        Args:
            value: The default local cache bool.

        Returns:
            This client, useful for chaining inplace calls.
        """
        self._local_cache = value
        return self

    def set_extra_headers(self, value: Headers) -> Self:
        """
        Set the default extra headers.  # noqa: DAR101.

        Args:
            value: The default extra headers.

        Returns:
            This client, useful for chaining inplace calls.
        """
        self._extra_headers = value
        return self

    def set_extra_query(self, value: Query) -> Self:
        """
        Set the default extra query.  # noqa: DAR101.

        Args:
            value: The default extra query.

        Returns:
            This client, useful for chaining inplace calls.
        """
        self._extra_query = value
        return self

    def set_extra_body(self, value: Body) -> Self:
        """
        Set the default extra body.  # noqa: DAR101.

        Args:
            value: The default extra body.

        Returns:
            This client, useful for chaining inplace calls.
        """
        self._extra_body = value
        return self

    # Credits #
    # --------#

    def get_credit_balance(self) -> Union[float, None]:
        # noqa: DAR201, DAR401
        """
        Get the remaining credits left on your account.

        Returns:
            The remaining credits on the account if successful, otherwise None.
        Raises:
            BadRequestError: If there was an HTTP error.
            ValueError: If there was an error parsing the JSON response.
        """
        url = f"{BASE_URL}/credits"
        headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }
        try:
            response = _requests.get(url, headers=headers, timeout=10)
            if response.status_code != 200:
                raise Exception(response.json())
            return response.json()["credits"]
        except requests.RequestException as e:
            raise requests.RequestException(
                "There was an error with the request.",
            ) from e
        except (KeyError, ValueError) as e:
            raise ValueError("Error parsing JSON response.") from e

    # Methods #
    # --------#

    def copy(self):
        # noinspection PyUnresolvedReferences,PyArgumentList
        return type(self)(
            **copy.deepcopy(
                {
                    **self._constructor_args,
                    **dict(
                        system_message=self._system_message,
                        messages=self._messages,
                        frequency_penalty=self._frequency_penalty,
                        logit_bias=self._logit_bias,
                        logprobs=self._logprobs,
                        top_logprobs=self._top_logprobs,
                        max_completion_tokens=self._max_completion_tokens,
                        n=self._n,
                        presence_penalty=self._presence_penalty,
                        response_format=self._response_format,
                        seed=self._seed,
                        stop=self._stop,
                        stream=self._stream,
                        stream_options=self._stream_options,
                        temperature=self._temperature,
                        top_p=self._top_p,
                        tools=self._tools,
                        tool_choice=self._tool_choice,
                        parallel_tool_calls=self._parallel_tool_calls,
                        # platform arguments
                        use_custom_keys=self._use_custom_keys,
                        tags=self._tags,
                        drop_params=self._drop_params,
                        region=self._region,
                        log_query_body=self._log_query_body,
                        log_response_body=self._log_response_body,
                        api_key=self._api_key,
                        # python client arguments
                        stateful=self._stateful,
                        return_full_completion=self._return_full_completion,
                        traced=self._traced,
                        cache=self._cache,
                        # passthrough arguments
                        extra_headers=self._extra_headers,
                        extra_query=self._extra_query,
                        **self._extra_body,
                    ),
                },
            ),
        )

    def json(self):
        model = create_model(type(self).__name__, __config__={"extra": "allow"})
        instance = model(
            **{
                "type": type(self).__name__,
                **self._constructor_args,
                **dict(
                    system_message=self._system_message,
                    messages=self._messages,
                    frequency_penalty=self._frequency_penalty,
                    logit_bias=self._logit_bias,
                    logprobs=self._logprobs,
                    top_logprobs=self._top_logprobs,
                    max_completion_tokens=self._max_completion_tokens,
                    n=self._n,
                    presence_penalty=self._presence_penalty,
                    response_format=self._response_format,
                    seed=self._seed,
                    stop=self._stop,
                    stream=self._stream,
                    stream_options=self._stream_options,
                    temperature=self._temperature,
                    top_p=self._top_p,
                    tools=self._tools,
                    tool_choice=self._tool_choice,
                    parallel_tool_calls=self._parallel_tool_calls,
                    # platform arguments
                    use_custom_keys=self._use_custom_keys,
                    tags=self._tags,
                    drop_params=self._drop_params,
                    region=self._region,
                    log_query_body=self._log_query_body,
                    log_response_body=self._log_response_body,
                    api_key=self._api_key,
                    # python client arguments
                    stateful=self._stateful,
                    return_full_completion=self._return_full_completion,
                    traced=self._traced,
                    cache=self._cache,
                    # passthrough arguments
                    extra_headers=self._extra_headers,
                    extra_query=self._extra_query,
                    extra_body=self._extra_body,
                ),
            },
        )
        return instance.model_dump()

    # Abstract Methods #
    # -----------------#

    @abstractmethod
    def _generate(
        self,
        messages: Optional[
            Union[
                List[ChatCompletionMessageParam],
                Dict[str, List[ChatCompletionMessageParam]],
            ]
        ],
        *,
        frequency_penalty: Optional[float],
        logit_bias: Optional[Dict[str, int]],
        logprobs: Optional[bool],
        top_logprobs: Optional[int],
        max_completion_tokens: Optional[int],
        n: Optional[int],
        presence_penalty: Optional[float],
        response_format: Optional[Union[Type[BaseModel], Dict[str, str]]],
        seed: Optional[int],
        stop: Union[Optional[str], List[str]],
        stream: Optional[bool],
        stream_options: Optional[ChatCompletionStreamOptionsParam],
        temperature: Optional[float],
        top_p: Optional[float],
        tools: Optional[Iterable[ChatCompletionToolParam]],
        tool_choice: Optional[ChatCompletionToolChoiceOptionParam],
        parallel_tool_calls: Optional[bool],
        # platform arguments
        use_custom_keys: bool,
        tags: Optional[List[str]],
        drop_params: Optional[bool],
        region: Optional[str],
        log_query_body: Optional[bool],
        log_response_body: Optional[bool],
        # python client arguments
        return_full_completion: bool,
        cache: Union[bool, str],
        # passthrough arguments
        extra_headers: Optional[Headers],
        extra_query: Optional[Query],
        **kwargs,
    ):
        raise NotImplementedError

```

`/Users/yushaarif/Unify/unify/unify/universal_api/clients/uni_llm.py`:

```py
# global
import abc
import threading
# noinspection PyProtectedMember
import time
import uuid
from typing import (AsyncGenerator, Dict, Generator, Iterable, List, Optional,
                    Type, Union)

import openai
# local
import unify
from openai._types import Headers, Query
from openai.types import CompletionUsage
from openai.types.chat import (ChatCompletion, ChatCompletionMessage,
                               ChatCompletionMessageParam,
                               ChatCompletionStreamOptionsParam,
                               ChatCompletionToolChoiceOptionParam,
                               ChatCompletionToolParam)
from openai.types.chat.chat_completion import Choice
from pydantic import BaseModel
from typing_extensions import Self
from unify import BASE_URL, LOCAL_MODELS
from unify.universal_api.clients.helpers import (_assert_is_valid_endpoint,
                                                 _assert_is_valid_model,
                                                 _assert_is_valid_provider)

from ...utils._caching import _get_cache, _get_caching, _write_to_cache
from ...utils.helpers import _default
from ..clients.base import _Client
from ..types import Prompt
from ..utils.endpoint_metrics import Metrics


class _UniClient(_Client, abc.ABC):
    def __init__(
        self,
        endpoint: Optional[str] = None,
        *,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        system_message: Optional[str] = None,
        messages: Optional[List[ChatCompletionMessageParam]] = None,
        frequency_penalty: Optional[float] = None,
        logit_bias: Optional[Dict[str, int]] = None,
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None,
        max_completion_tokens: Optional[int] = None,
        n: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        response_format: Optional[Union[Type[BaseModel], Dict[str, str]]] = None,
        seed: Optional[int] = None,
        stop: Union[Optional[str], List[str]] = None,
        stream: Optional[bool] = False,
        stream_options: Optional[ChatCompletionStreamOptionsParam] = None,
        temperature: Optional[float] = 1.0,
        top_p: Optional[float] = None,
        tools: Optional[Iterable[ChatCompletionToolParam]] = None,
        tool_choice: Optional[ChatCompletionToolChoiceOptionParam] = None,
        parallel_tool_calls: Optional[bool] = None,
        # platform arguments
        use_custom_keys: bool = False,
        tags: Optional[List[str]] = None,
        drop_params: Optional[bool] = True,
        region: Optional[str] = None,
        log_query_body: Optional[bool] = True,
        log_response_body: Optional[bool] = True,
        api_key: Optional[str] = None,
        # python client arguments
        stateful: bool = False,
        return_full_completion: bool = False,
        traced: bool = False,
        cache: Optional[Union[bool, str]] = None,
        local_cache: bool = True,
        # passthrough arguments
        extra_headers: Optional[Headers] = None,
        extra_query: Optional[Query] = None,
        **kwargs,
    ):
        """Initialize the Uni LLM Unify client.

        Args:
            endpoint: Endpoint name in OpenAI API format:
            <model_name>@<provider_name>
            Defaults to None.

            model: Name of the model. Should only be set if endpoint is not set.

            provider: Name of the provider. Should only be set if endpoint is not set.

            system_message: An optional string containing the system message. This
            always appears at the beginning of the list of messages.

            messages: A list of messages comprising the conversation so far. This will
            be appended to the system_message if it is not None, and any user_message
            will be appended if it is not None.

            frequency_penalty: Number between -2.0 and 2.0. Positive values penalize new
            tokens based on their existing frequency in the text so far, decreasing the
            model's likelihood to repeat the same line verbatim.

            logit_bias: Modify the likelihood of specified tokens appearing in the
            completion. Accepts a JSON object that maps tokens (specified by their token
            ID in the tokenizer) to an associated bias value from -100 to 100.
            Mathematically, the bias is added to the logits generated by the model prior
            to sampling. The exact effect will vary per model, but values between -1 and
            1 should decrease or increase likelihood of selection; values like -100 or
            100 should result in a ban or exclusive selection of the relevant token.

            logprobs: Whether to return log probabilities of the output tokens or not.
            If true, returns the log probabilities of each output token returned in the
            content of message.

            top_logprobs: An integer between 0 and 20 specifying the number of most
            likely tokens to return at each token position, each with an associated log
            probability. logprobs must be set to true if this parameter is used.

            max_completion_tokens: The maximum number of tokens that can be generated in
            the chat completion. The total length of input tokens and generated tokens
            is limited by the model's context length. Defaults to the provider's default
            max_completion_tokens when the value is None.

            n: How many chat completion choices to generate for each input message. Note
            that you will be charged based on the number of generated tokens across all
            of the choices. Keep n as 1 to minimize costs.

            presence_penalty: Number between -2.0 and 2.0. Positive values penalize new
            tokens based on whether they appear in the text so far, increasing the
            model's likelihood to talk about new topics.

            response_format: An object specifying the format that the model must output.
            Setting to `{ "type": "json_schema", "json_schema": {...} }` enables
            Structured Outputs which ensures the model will match your supplied JSON
            schema. Learn more in the Structured Outputs guide. Setting to
            `{ "type": "json_object" }` enables JSON mode, which ensures the message the
            model generates is valid JSON.

            seed: If specified, a best effort attempt is made to sample
            deterministically, such that repeated requests with the same seed and
            parameters should return the same result. Determinism is not guaranteed, and
            you should refer to the system_fingerprint response parameter to monitor
            changes in the backend.

            stop: Up to 4 sequences where the API will stop generating further tokens.

            stream: If True, generates content as a stream. If False, generates content
            as a single response. Defaults to False.

            stream_options: Options for streaming response. Only set this when you set
            stream: true.

            temperature:  What sampling temperature to use, between 0 and 2.
            Higher values like 0.8 will make the output more random,
            while lower values like 0.2 will make it more focused and deterministic.
            It is generally recommended to alter this or top_p, but not both.
            Defaults to the provider's default max_completion_tokens when the value is
            None.

            top_p: An alternative to sampling with temperature, called nucleus sampling,
            where the model considers the results of the tokens with top_p probability
            mass. So 0.1 means only the tokens comprising the top 10% probability mass
            are considered. Generally recommended to alter this or temperature, but not
            both.

            tools: A list of tools the model may call. Currently, only functions are
            supported as a tool. Use this to provide a list of functions the model may
            generate JSON inputs for. A max of 128 functions are supported.

            tool_choice: Controls which (if any) tool is called by the
            model. none means the model will not call any tool and instead generates a
            message. auto means the model can pick between generating a message or
            calling one or more tools. required means the model must call one or more
            tools. Specifying a particular tool via
            `{ "type": "function", "function": {"name": "my_function"} }`
            forces the model to call that tool.
            none is the default when no tools are present. auto is the default if tools
            are present.

            parallel_tool_calls: Whether to enable parallel function calling during tool
            use.

            use_custom_keys:  Whether to use custom API keys or our unified API keys
            with the backend provider.

            tags: Arbitrary number of tags to classify this API query as needed. Helpful
            for generally grouping queries across tasks and users, for logging purposes.

            drop_params: Whether or not to drop unsupported OpenAI params by the
            provider you’re using.

            region: A string used to represent the region where the endpoint is
            accessed. Only relevant for on-prem deployments with certain providers like
            `vertex-ai`, `aws-bedrock` and `azure-ml`, where the endpoint is being
            accessed through a specified region.

            log_query_body: Whether to log the contents of the query json body.

            log_response_body: Whether to log the contents of the response json body.

            stateful:  Whether the conversation history is preserved within the messages
            of this client. If True, then history is preserved. If False, then this acts
            as a stateless client, and message histories must be managed by the user.

            return_full_completion: If False, only return the message content
            chat_completion.choices[0].message.content.strip(" ") from the OpenAI
            return. Otherwise, the full response chat_completion is returned.
            Defaults to False.

            traced: Whether to trace the generate method.

            cache: If True, then the arguments will be stored in a local cache file, and
            any future calls with identical arguments will read from the cache instead
            of running the LLM query. If "write" then the cache will only be written
            to, if "read" then the cache will be read from if a cache is available but
            will not write, and if "read-only" then the argument must be present in the
            cache, else an exception will be raised. Finally, an appending "-closest"
            will read the closest match from the cache, and overwrite it if cache writing
            is enabled. This argument only has any effect when stream=False.

            extra_headers: Additional "passthrough" headers for the request which are
            provider-specific, and are not part of the OpenAI standard. They are handled
            by the provider-specific API.

            extra_query: Additional "passthrough" query parameters for the request which
            are provider-specific, and are not part of the OpenAI standard. They are
            handled by the provider-specific API.

            kwargs: Additional "passthrough" JSON properties for the body of the
            request, which are provider-specific, and are not part of the OpenAI
            standard. They will be handled by the provider-specific API.

        Raises:
            UnifyError: If the API key is missing.
        """
        self._base_constructor_args = dict(
            system_message=system_message,
            messages=messages,
            frequency_penalty=frequency_penalty,
            logit_bias=logit_bias,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            max_completion_tokens=max_completion_tokens,
            n=n,
            presence_penalty=presence_penalty,
            response_format=response_format,
            seed=seed,
            stop=stop,
            stream=stream,
            stream_options=stream_options,
            temperature=temperature,
            top_p=top_p,
            tools=tools,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
            # platform arguments
            use_custom_keys=use_custom_keys,
            tags=tags,
            drop_params=drop_params,
            region=region,
            log_query_body=log_query_body,
            log_response_body=log_response_body,
            api_key=api_key,
            # python client arguments
            stateful=stateful,
            return_full_completion=return_full_completion,
            traced=traced,
            cache=cache,
            local_cache=local_cache,
            # passthrough arguments
            extra_headers=extra_headers,
            extra_query=extra_query,
            **kwargs,
        )
        super().__init__(**self._base_constructor_args)
        self._constructor_args = dict(
            endpoint=endpoint,
            model=model,
            provider=provider,
            **self._base_constructor_args,
        )
        if endpoint and (model or provider):
            raise Exception(
                "if the model or provider are passed, then the endpoint must not be"
                "passed.",
            )
        self._client = self._get_client()
        self._endpoint = None
        self._provider = None
        self._model = None
        if endpoint:
            self.set_endpoint(endpoint)
        if provider:
            self.set_provider(provider)
        if model:
            self.set_model(model)

    # Read-only Properties #
    # ---------------------#

    def _get_metric(self) -> Metrics:
        return unify.get_endpoint_metrics(self._endpoint, api_key=self._api_key)[0]

    @property
    def input_cost(self) -> float:
        return self._get_metric().input_cost

    @property
    def output_cost(self) -> float:
        return self._get_metric().output_cost

    @property
    def ttft(self) -> float:
        return self._get_metric().ttft

    @property
    def itl(self) -> float:
        return self._get_metric().itl

    # Settable Properties #
    # --------------------#

    @property
    def endpoint(self) -> str:
        """
        Get the endpoint name.

        Returns:
            The endpoint name.
        """
        return self._endpoint

    @property
    def model(self) -> str:
        """
        Get the model name.

        Returns:
            The model name.
        """
        return self._model

    @property
    def provider(self) -> str:
        """
        Get the provider name.

        Returns:
            The provider name.
        """
        return self._provider

    # Setters #
    # --------#

    def set_endpoint(self, value: str) -> Self:
        """
        Set the endpoint name.  # noqa: DAR101.

        Args:
            value: The endpoint name.

        Returns:
            This client, useful for chaining inplace calls.
        """
        _assert_is_valid_endpoint(value, api_key=self._api_key)
        self._endpoint = value
        if value == "user-input":
            return self
        lhs = value.split("->")[0]
        if "@" in lhs:
            self._model, self._provider = lhs.split("@")
        else:
            self._model = lhs
            self._provider = value.split("->")[1]
        return self

    def set_model(self, value: str) -> Self:
        """
        Set the model name.  # noqa: DAR101.

        Args:
            value: The model name.

        Returns:
            This client, useful for chaining inplace calls.
        """
        custom_or_local = self._provider == "local" or "custom" in self._provider
        _assert_is_valid_model(
            value,
            custom_or_local=custom_or_local,
            api_key=self._api_key,
        )
        if self._provider:
            self._endpoint = "@".join([value, self._provider])
        return self

    def set_provider(self, value: str) -> Self:
        """
        Set the provider name.  # noqa: DAR101.

        Args:
            value: The provider name.

        Returns:
            This client, useful for chaining inplace calls.
        """
        _assert_is_valid_provider(value, api_key=self._api_key)
        self._provider = value
        if self._model:
            self._endpoint = "@".join([self._model, value])
        return self

    @staticmethod
    def _handle_kw(
        prompt,
        endpoint,
        stream,
        stream_options,
        use_custom_keys,
        tags,
        drop_params,
        region,
        log_query_body,
        log_response_body,
    ):
        prompt_dict = prompt.components
        if "extra_body" in prompt_dict:
            extra_body = prompt_dict["extra_body"]
            del prompt_dict["extra_body"]
        else:
            extra_body = {}
        kw = dict(
            model=endpoint,
            **prompt_dict,
            stream=stream,
            stream_options=stream_options,
            extra_body={  # platform arguments
                "signature": "python",
                "use_custom_keys": use_custom_keys,
                "tags": tags,
                "drop_params": drop_params,
                "region": region,
                "log_query_body": log_query_body,
                "log_response_body": log_response_body,
                # passthrough json arguments
                **extra_body,
            },
        )
        return {k: v for k, v in kw.items() if v is not None}

    # Representation #
    # ---------------#

    def __repr__(self):
        return "{}(endpoint={})".format(self.__class__.__name__, self._endpoint)

    def __str__(self):
        return "{}(endpoint={})".format(self.__class__.__name__, self._endpoint)

    # Abstract #
    # ---------#

    @abc.abstractmethod
    def _get_client(self):
        raise NotImplementedError

    # Generate #
    # ---------#

    def generate(
        self,
        user_message: Optional[str] = None,
        system_message: Optional[str] = None,
        messages: Optional[
            Union[
                List[ChatCompletionMessageParam],
                Dict[str, List[ChatCompletionMessageParam]],
            ]
        ] = None,
        *,
        frequency_penalty: Optional[float] = None,
        logit_bias: Optional[Dict[str, int]] = None,
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None,
        max_completion_tokens: Optional[int] = None,
        n: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        response_format: Optional[Union[Type[BaseModel], Dict[str, str]]] = None,
        seed: Optional[int] = None,
        stop: Union[Optional[str], List[str]] = None,
        stream: Optional[bool] = None,
        stream_options: Optional[ChatCompletionStreamOptionsParam] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        tools: Optional[Iterable[ChatCompletionToolParam]] = None,
        tool_choice: Optional[ChatCompletionToolChoiceOptionParam] = None,
        parallel_tool_calls: Optional[bool] = None,
        # platform arguments
        use_custom_keys: Optional[bool] = None,
        tags: Optional[List[str]] = None,
        drop_params: Optional[bool] = None,
        region: Optional[str] = None,
        log_query_body: Optional[bool] = None,
        log_response_body: Optional[bool] = None,
        # python client arguments
        stateful: Optional[bool] = None,
        return_full_completion: Optional[bool] = None,
        cache: Optional[Union[bool, str]] = None,
        local_cache: Optional[bool] = None,
        # passthrough arguments
        extra_headers: Optional[Headers] = None,
        extra_query: Optional[Query] = None,
        **kwargs,
    ):
        """Generate a ChatCompletion response for the specified endpoint,
        from the provided query parameters.

        Args:
            user_message: A string containing the user message.
            If provided, messages must be None.

            system_message: An optional string containing the system message. This
            always appears at the beginning of the list of messages.

            messages: A list of messages comprising the conversation so far, or
            optionally a dictionary of such messages, with clients as the keys in the
            case of multi-llm clients. This will be appended to the system_message if it
            is not None, and any user_message will be appended if it is not None.

            frequency_penalty: Number between -2.0 and 2.0. Positive values penalize new
            tokens based on their existing frequency in the text so far, decreasing the
            model's likelihood to repeat the same line verbatim.

            logit_bias: Modify the likelihood of specified tokens appearing in the
            completion. Accepts a JSON object that maps tokens (specified by their token
            ID in the tokenizer) to an associated bias value from -100 to 100.
            Mathematically, the bias is added to the logits generated by the model prior
            to sampling. The exact effect will vary per model, but values between -1 and
            1 should decrease or increase likelihood of selection; values like -100 or
            100 should result in a ban or exclusive selection of the relevant token.

            logprobs: Whether to return log probabilities of the output tokens or not.
            If true, returns the log probabilities of each output token returned in the
            content of message.

            top_logprobs: An integer between 0 and 20 specifying the number of most
            likely tokens to return at each token position, each with an associated log
            probability. logprobs must be set to true if this parameter is used.

            max_completion_tokens: The maximum number of tokens that can be generated in
            the chat completion. The total length of input tokens and generated tokens
            is limited by the model's context length. Defaults value is None. Uses the
            provider's default max_completion_tokens when None is explicitly passed.

            n: How many chat completion choices to generate for each input message. Note
            that you will be charged based on the number of generated tokens across all
            of the choices. Keep n as 1 to minimize costs.

            presence_penalty: Number between -2.0 and 2.0. Positive values penalize new
            tokens based on whether they appear in the text so far, increasing the
            model's likelihood to talk about new topics.

            response_format: An object specifying the format that the model must output.
            Setting to `{ "type": "json_schema", "json_schema": {...} }` enables
            Structured Outputs which ensures the model will match your supplied JSON
            schema. Learn more in the Structured Outputs guide. Setting to
            `{ "type": "json_object" }` enables JSON mode, which ensures the message the
            model generates is valid JSON.

            seed: If specified, a best effort attempt is made to sample
            deterministically, such that repeated requests with the same seed and
            parameters should return the same result. Determinism is not guaranteed, and
            you should refer to the system_fingerprint response parameter to monitor
            changes in the backend.

            stop: Up to 4 sequences where the API will stop generating further tokens.

            stream: If True, generates content as a stream. If False, generates content
            as a single response. Defaults to False.

            stream_options: Options for streaming response. Only set this when you set
            stream: true.

            temperature:  What sampling temperature to use, between 0 and 2.
            Higher values like 0.8 will make the output more random,
            while lower values like 0.2 will make it more focused and deterministic.
            It is generally recommended to alter this or top_p, but not both.
            Default value is 1.0. Defaults to the provider's default temperature when
            None is explicitly passed.

            top_p: An alternative to sampling with temperature, called nucleus sampling,
            where the model considers the results of the tokens with top_p probability
            mass. So 0.1 means only the tokens comprising the top 10% probability mass
            are considered. Generally recommended to alter this or temperature, but not
            both.

            tools: A list of tools the model may call. Currently, only functions are
            supported as a tool. Use this to provide a list of functions the model may
            generate JSON inputs for. A max of 128 functions are supported.

            tool_choice: Controls which (if any) tool is called by the
            model. none means the model will not call any tool and instead generates a
            message. auto means the model can pick between generating a message or
            calling one or more tools. required means the model must call one or more
            tools. Specifying a particular tool via
            `{ "type": "function", "function": {"name": "my_function"} }`
            forces the model to call that tool.
            none is the default when no tools are present. auto is the default if tools
            are present.

            parallel_tool_calls: Whether to enable parallel function calling during tool
            use.

            use_custom_keys:  Whether to use custom API keys or our unified API keys
            with the backend provider. Defaults to False.

            tags: Arbitrary number of tags to classify this API query as needed. Helpful
            for generally grouping queries across tasks and users, for logging purposes.

            drop_params: Whether or not to drop unsupported OpenAI params by the
            provider you’re using.

            region: A string used to represent the region where the endpoint is
            accessed. Only relevant for on-prem deployments with certain providers like
            `vertex-ai`, `aws-bedrock` and `azure-ml`, where the endpoint is being
            accessed through a specified region.

            log_query_body: Whether to log the contents of the query json body.

            log_response_body: Whether to log the contents of the response json body.

            stateful:  Whether the conversation history is preserved within the messages
            of this client. If True, then history is preserved. If False, then this acts
            as a stateless client, and message histories must be managed by the user.

            return_full_completion: If False, only return the message content
            chat_completion.choices[0].message.content.strip(" ") from the OpenAI
            return. Otherwise, the full response chat_completion is returned.
            Defaults to False.

            cache: If True, then the arguments will be stored in a local cache file, and
            any future calls with identical arguments will read from the cache instead
            of running the LLM query. If "write" then the cache will only be written
            to, if "read" then the cache will be read from if a cache is available but
            will not write, and if "read-only" then the argument must be present in the
            cache, else an exception will be raised. Finally, an appending "-closest"
            will read the closest match from the cache, and overwrite it if cache writing
            is enabled. This argument only has any effect when stream=False.

            extra_headers: Additional "passthrough" headers for the request which are
            provider-specific, and are not part of the OpenAI standard. They are handled
            by the provider-specific API.

            extra_query: Additional "passthrough" query parameters for the request which
            are provider-specific, and are not part of the OpenAI standard. They are
            handled by the provider-specific API.

            kwargs: Additional "passthrough" JSON properties for the body of the
            request, which are provider-specific, and are not part of the OpenAI
            standard. They will be handled by the provider-specific API.

        Returns:
            If stream is True, returns a generator yielding chunks of content.
            If stream is False, returns a single string response.

        Raises:
            UnifyError: If an error occurs during content generation.
        """
        system_message = _default(system_message, self._system_message)
        messages = _default(messages, self._messages)
        stateful = _default(stateful, self._stateful)
        if messages:
            sys_msg_inside = any(msg["role"] == "system" for msg in messages)
            if not sys_msg_inside and system_message is not None:
                messages = [
                    {"role": "system", "content": system_message},
                ] + messages
            if user_message is not None:
                messages += [{"role": "user", "content": user_message}]
        else:
            messages = list()
            if system_message is not None:
                messages += [{"role": "system", "content": system_message}]
            if user_message is not None:
                messages += [{"role": "user", "content": user_message}]
            self._messages = messages
        return_full_completion = (
            True
            if _default(tools, self._tools)
            else _default(return_full_completion, self._return_full_completion)
        )
        cache = _default(cache, self._cache)
        _cache_modes = ["read", "read-only", "write", "both"]
        assert cache in _cache_modes + [m + "-closest" for m in _cache_modes] + [
            True,
            False,
            None,
        ]
        ret = self._generate(
            messages=messages,
            frequency_penalty=_default(frequency_penalty, self._frequency_penalty),
            logit_bias=_default(logit_bias, self._logit_bias),
            logprobs=_default(logprobs, self._logprobs),
            top_logprobs=_default(top_logprobs, self._top_logprobs),
            max_completion_tokens=_default(
                max_completion_tokens,
                self._max_completion_tokens,
            ),
            n=_default(n, self._n),
            presence_penalty=_default(presence_penalty, self._presence_penalty),
            response_format=_default(response_format, self._response_format),
            seed=_default(_default(seed, self._seed), unify.get_seed()),
            stop=_default(stop, self._stop),
            stream=_default(stream, self._stream),
            stream_options=_default(stream_options, self._stream_options),
            temperature=_default(temperature, self._temperature),
            top_p=_default(top_p, self._top_p),
            tools=_default(tools, self._tools),
            tool_choice=_default(tool_choice, self._tool_choice),
            parallel_tool_calls=_default(
                parallel_tool_calls,
                self._parallel_tool_calls,
            ),
            # platform arguments
            use_custom_keys=_default(use_custom_keys, self._use_custom_keys),
            tags=_default(tags, self._tags),
            drop_params=_default(drop_params, self._drop_params),
            region=_default(region, self._region),
            log_query_body=_default(log_query_body, self._log_query_body),
            log_response_body=_default(log_response_body, self._log_response_body),
            # python client arguments
            return_full_completion=return_full_completion,
            cache=_default(cache, _get_caching()),
            local_cache=_default(local_cache, self._local_cache),
            # passthrough arguments
            extra_headers=_default(extra_headers, self._extra_headers),
            extra_query=_default(extra_query, self._extra_query),
            **{**self._extra_body, **kwargs},
        )
        if stateful:
            if return_full_completion:
                msg = [ret.choices[0].message.model_dump()]
            else:
                msg = [{"role": "assistant", "content": ret}]
            if self._messages is None:
                self._messages = []
            self._messages += msg
        elif self._messages:
            self._messages.clear()
        return ret


class Unify(_UniClient):
    """Class for interacting with the Unify chat completions endpoint in a synchronous
    manner."""

    def _get_client(self):
        try:
            return openai.OpenAI(
                base_url=f"{BASE_URL}",
                api_key=self._api_key,
                timeout=3600.0,  # one hour
            )
        except openai.OpenAIError as e:
            raise Exception(f"Failed to initialize Unify client: {str(e)}")

    def _generate_stream(
        self,
        endpoint: str,
        prompt: Prompt,
        # stream
        stream_options: Optional[ChatCompletionStreamOptionsParam],
        # platform arguments
        use_custom_keys: bool,
        tags: Optional[List[str]],
        drop_params: Optional[bool],
        region: Optional[str],
        log_query_body: Optional[bool],
        log_response_body: Optional[bool],
        # python client arguments
        return_full_completion: bool,
    ) -> Generator[str, None, None]:
        kw = self._handle_kw(
            prompt=prompt,
            endpoint=endpoint,
            stream=True,
            stream_options=stream_options,
            use_custom_keys=use_custom_keys,
            tags=tags,
            drop_params=drop_params,
            region=region,
            log_query_body=log_query_body,
            log_response_body=log_response_body,
        )
        try:
            if endpoint in LOCAL_MODELS:
                kw.pop("extra_body")
                kw.pop("model")
                kw.pop("max_completion_tokens")
                chat_completion = LOCAL_MODELS[endpoint](**kw)
            else:
                if unify.CLIENT_LOGGING:
                    print(f"calling {kw['model']}... (thread {threading.get_ident()})")
                if self.traced:
                    chat_completion = unify.traced(
                        self._client.chat.completions.create,
                        span_type="llm",
                        name=(
                            endpoint
                            if tags is None
                            else endpoint + "[" + ",".join([str(t) for t in tags]) + "]"
                        ),
                    )(**kw)
                else:
                    chat_completion = self._client.chat.completions.create(**kw)
                if unify.CLIENT_LOGGING:
                    print(f"done (thread {threading.get_ident()})")
            for chunk in chat_completion:
                if return_full_completion:
                    content = chunk
                else:
                    content = chunk.choices[0].delta.content  # type: ignore[union-attr]    # noqa: E501
                if content is not None:
                    yield content
        except openai.APIStatusError as e:
            raise Exception(e.message)

    def _generate_non_stream(
        self,
        endpoint: str,
        prompt: Prompt,
        # platform arguments
        use_custom_keys: bool,
        tags: Optional[List[str]],
        drop_params: Optional[bool],
        region: Optional[str],
        log_query_body: Optional[bool],
        log_response_body: Optional[bool],
        # python client arguments
        return_full_completion: bool,
        cache: Union[bool, str],
        local_cache: bool,
    ) -> Union[str, ChatCompletion]:
        kw = self._handle_kw(
            prompt=prompt,
            endpoint=endpoint,
            stream=False,
            stream_options=None,
            use_custom_keys=use_custom_keys,
            tags=tags,
            drop_params=drop_params,
            region=region,
            log_query_body=log_query_body,
            log_response_body=log_response_body,
        )
        if isinstance(cache, str) and cache.endswith("-closest"):
            cache = cache.removesuffix("-closest")
            read_closest = True
        else:
            read_closest = False
        if "response_format" in kw:
            chat_method = self._client.beta.chat.completions.parse
            del kw["stream"]
        elif endpoint == "user-input":
            chat_method = lambda *a, **kw: input("write your agent response:\n")
        else:
            chat_method = self._client.chat.completions.create
        chat_completion = None
        in_cache = False
        if cache in [True, "both", "read", "read-only"]:
            if self._traced:

                def _get_cache_traced(**kw):
                    return _get_cache(
                        fn_name="chat.completions.create",
                        kw=kw,
                        raise_on_empty=cache == "read-only",
                        read_closest=read_closest,
                        delete_closest=read_closest,
                        local=local_cache,
                    )

                chat_completion = unify.traced(
                    _get_cache_traced,
                    span_type="llm-cached",
                    name=(
                        endpoint
                        if tags is None
                        else endpoint + "[" + ",".join([str(t) for t in tags]) + "]"
                    ),
                )(**kw)
            else:
                chat_completion = _get_cache(
                    fn_name="chat.completions.create",
                    kw=kw,
                    raise_on_empty=cache == "read-only",
                    read_closest=read_closest,
                    delete_closest=read_closest,
                    local=local_cache,
                )
                in_cache = True if chat_completion is not None else False
        if chat_completion is None:
            try:
                if endpoint in LOCAL_MODELS:
                    kw.pop("extra_body")
                    kw.pop("model")
                    kw.pop("max_completion_tokens")
                    chat_completion = LOCAL_MODELS[endpoint](**kw)
                else:
                    if unify.CLIENT_LOGGING:
                        print(
                            f"calling {kw['model']}... (thread {threading.get_ident()})",
                        )
                    if self._traced:
                        chat_completion = unify.traced(
                            chat_method,
                            span_type="llm",
                            name=(
                                endpoint
                                if tags is None
                                else endpoint
                                + "["
                                + ",".join([str(t) for t in tags])
                                + "]"
                            ),
                        )(**kw)
                    else:
                        chat_completion = chat_method(**kw)
                    if unify.CLIENT_LOGGING:
                        print(f"done (thread {threading.get_ident()})")
            except openai.APIStatusError as e:
                raise Exception(e.message)
        if (chat_completion is not None or read_closest) and cache in [
            True,
            "both",
            "write",
        ]:
            if not in_cache or cache == "write":
                _write_to_cache(
                    fn_name="chat.completions.create",
                    kw=kw,
                    response=chat_completion,
                    local=local_cache,
                )
        if return_full_completion:
            if endpoint == "user-input":
                input_msg = sum(len(msg) for msg in prompt.components["messages"])
                return ChatCompletion(
                    id=str(uuid.uuid4()),
                    object="chat.completion",
                    created=int(time.time()),
                    model=endpoint,
                    choices=[
                        Choice(
                            index=0,
                            message=ChatCompletionMessage(
                                role="assistant",
                                content=chat_completion,
                            ),
                            finish_reason="stop",
                        ),
                    ],
                    usage=CompletionUsage(
                        prompt_tokens=input_msg,
                        completion_tokens=len(chat_completion),
                        total_tokens=input_msg + len(chat_completion),
                    ),
                )
            return chat_completion
        elif endpoint == "user-input":
            return chat_completion
        content = chat_completion.choices[0].message.content
        if content:
            return content.strip(" ")
        return ""

    def _generate(  # noqa: WPS234, WPS211
        self,
        messages: Optional[List[ChatCompletionMessageParam]],
        *,
        frequency_penalty: Optional[float],
        logit_bias: Optional[Dict[str, int]],
        logprobs: Optional[bool],
        top_logprobs: Optional[int],
        max_completion_tokens: Optional[int],
        n: Optional[int],
        presence_penalty: Optional[float],
        response_format: Optional[Union[Type[BaseModel], Dict[str, str]]],
        seed: Optional[int],
        stop: Union[Optional[str], List[str]],
        stream: Optional[bool],
        stream_options: Optional[ChatCompletionStreamOptionsParam],
        temperature: Optional[float],
        top_p: Optional[float],
        tools: Optional[Iterable[ChatCompletionToolParam]],
        tool_choice: Optional[ChatCompletionToolChoiceOptionParam],
        parallel_tool_calls: Optional[bool],
        # platform arguments
        use_custom_keys: bool,
        tags: Optional[List[str]],
        drop_params: Optional[bool],
        region: Optional[str],
        log_query_body: Optional[bool],
        log_response_body: Optional[bool],
        # python client arguments
        return_full_completion: bool,
        cache: Union[bool, str],
        local_cache: bool,
        # passthrough arguments
        extra_headers: Optional[Headers],
        extra_query: Optional[Query],
        **kwargs,
    ) -> Union[Generator[str, None, None], str]:  # noqa: DAR101, DAR201, DAR401
        prompt = Prompt(
            messages=messages,
            frequency_penalty=frequency_penalty,
            logit_bias=logit_bias,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            max_completion_tokens=max_completion_tokens,
            n=n,
            presence_penalty=presence_penalty,
            response_format=response_format,
            seed=seed,
            stop=stop,
            temperature=temperature,
            top_p=top_p,
            tools=tools,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
            extra_headers=extra_headers,
            extra_query=extra_query,
            extra_body=kwargs,
        )
        if stream:
            return self._generate_stream(
                self._endpoint,
                prompt,
                # stream
                stream_options=stream_options,
                # platform arguments
                use_custom_keys=use_custom_keys,
                tags=tags,
                drop_params=drop_params,
                region=region,
                log_query_body=log_query_body,
                log_response_body=log_response_body,
                # python client arguments
                return_full_completion=return_full_completion,
            )
        return self._generate_non_stream(
            self._endpoint,
            prompt,
            # platform arguments
            use_custom_keys=use_custom_keys,
            tags=tags,
            drop_params=drop_params,
            region=region,
            log_query_body=log_query_body,
            log_response_body=log_response_body,
            # python client arguments
            return_full_completion=return_full_completion,
            cache=cache,
            local_cache=local_cache,
        )

    def to_async_client(self):
        """
        Return an asynchronous version of the client (`AsyncUnify` instance), with the
        exact same configuration as this synchronous (`Unify`) client.

        Returns:
            An `AsyncUnify` instance with the same configuration as this `Unify`
            instance.
        """
        return AsyncUnify(**self._constructor_args)


class AsyncUnify(_UniClient):
    """Class for interacting with the Unify chat completions endpoint in a synchronous
    manner."""

    def _get_client(self):
        try:
            return openai.AsyncOpenAI(
                base_url=f"{BASE_URL}",
                api_key=self._api_key,
                timeout=3600.0,  # one hour
            )
        except openai.APIStatusError as e:
            raise Exception(f"Failed to initialize Unify client: {str(e)}")

    async def _generate_stream(
        self,
        endpoint: str,
        prompt: Prompt,
        # stream
        stream_options: Optional[ChatCompletionStreamOptionsParam],
        # platform arguments
        use_custom_keys: bool,
        tags: Optional[List[str]],
        drop_params: Optional[bool],
        region: Optional[str],
        log_query_body: Optional[bool],
        log_response_body: Optional[bool],
        # python client arguments
        return_full_completion: bool,
    ) -> AsyncGenerator[str, None]:
        kw = self._handle_kw(
            prompt=prompt,
            endpoint=endpoint,
            stream=True,
            stream_options=stream_options,
            use_custom_keys=use_custom_keys,
            tags=tags,
            drop_params=drop_params,
            region=region,
            log_query_body=log_query_body,
            log_response_body=log_response_body,
        )
        try:
            if endpoint in LOCAL_MODELS:
                kw.pop("extra_body")
                kw.pop("model")
                kw.pop("max_completion_tokens")
                async_stream = await LOCAL_MODELS[endpoint](**kw)
            else:
                if unify.CLIENT_LOGGING:
                    print(f"calling {kw['model']}... (thread {threading.get_ident()})")
                if self._traced:
                    # ToDo: test if this works, it probably won't
                    async_stream = await unify.traced(
                        self._client.chat.completions.create,
                        span_type="llm",
                        name=(
                            endpoint
                            if tags is None
                            else endpoint + "[" + ",".join([str(t) for t in tags]) + "]"
                        ),
                    )(**kw)
                else:
                    async_stream = await self._client.chat.completions.create(**kw)
                if unify.CLIENT_LOGGING:
                    print(f"done (thread {threading.get_ident()})")
            async for chunk in async_stream:  # type: ignore[union-attr]
                if return_full_completion:
                    yield chunk
                else:
                    yield chunk.choices[0].delta.content or ""
        except openai.APIStatusError as e:
            raise Exception(e.message)

    async def _generate_non_stream(
        self,
        endpoint: str,
        prompt: Prompt,
        # platform arguments
        use_custom_keys: bool,
        tags: Optional[List[str]],
        drop_params: Optional[bool],
        region: Optional[str],
        log_query_body: Optional[bool],
        log_response_body: Optional[bool],
        # python client arguments
        return_full_completion: bool,
        cache: Union[bool, str],
        local_cache: bool,
    ) -> Union[str, ChatCompletion]:
        kw = self._handle_kw(
            prompt=prompt,
            endpoint=endpoint,
            stream=False,
            stream_options=None,
            use_custom_keys=use_custom_keys,
            tags=tags,
            drop_params=drop_params,
            region=region,
            log_query_body=log_query_body,
            log_response_body=log_response_body,
        )
        # ToDo: add all proper cache support, as is done for synchronous version above
        if cache is True:
            chat_completion = _get_cache(fn_name="chat.completions.create", kw=kw)
        else:
            chat_completion = None
        if chat_completion is None:
            try:
                if endpoint in LOCAL_MODELS:
                    kw.pop("extra_body")
                    kw.pop("model")
                    kw.pop("max_completion_tokens")
                    chat_completion = await LOCAL_MODELS[endpoint](**kw)
                else:
                    if unify.CLIENT_LOGGING:
                        print(
                            f"calling {kw['model']}... (thread {threading.get_ident()})",
                        )
                    if self.traced:
                        # ToDo: test if this works, it probably won't
                        chat_completion = await unify.traced(
                            self._client.chat.completions.create,
                            span_type="llm",
                            name=(
                                endpoint
                                if tags is None
                                else endpoint
                                + "["
                                + ",".join([str(t) for t in tags])
                                + "]"
                            ),
                        )(**kw)
                    else:
                        chat_completion = await self._client.chat.completions.create(
                            **kw,
                        )
                    if unify.CLIENT_LOGGING:
                        print(
                            f"done (thread {threading.get_ident()})",
                        )
            except openai.APIStatusError as e:
                raise Exception(e.message)
            if cache is True:
                _write_to_cache(
                    fn_name="chat.completions.create",
                    kw=kw,
                    response=chat_completion,
                )
        if return_full_completion:
            return chat_completion
        content = chat_completion.choices[0].message.content
        if content:
            return content.strip(" ")
        return ""

    async def _generate(  # noqa: WPS234, WPS211
        self,
        messages: Optional[List[ChatCompletionMessageParam]],
        *,
        frequency_penalty: Optional[float],
        logit_bias: Optional[Dict[str, int]],
        logprobs: Optional[bool],
        top_logprobs: Optional[int],
        max_completion_tokens: Optional[int],
        n: Optional[int],
        presence_penalty: Optional[float],
        response_format: Optional[Union[Type[BaseModel], Dict[str, str]]],
        seed: Optional[int],
        stop: Union[Optional[str], List[str]],
        stream: Optional[bool],
        stream_options: Optional[ChatCompletionStreamOptionsParam],
        temperature: Optional[float],
        top_p: Optional[float],
        tools: Optional[Iterable[ChatCompletionToolParam]],
        tool_choice: Optional[ChatCompletionToolChoiceOptionParam],
        parallel_tool_calls: Optional[bool],
        # platform arguments
        use_custom_keys: bool,
        tags: Optional[List[str]],
        drop_params: Optional[bool],
        region: Optional[str],
        log_query_body: Optional[bool],
        log_response_body: Optional[bool],
        # python client arguments
        return_full_completion: bool,
        cache: Union[bool, str],
        local_cache: bool,
        # passthrough arguments
        extra_headers: Optional[Headers],
        extra_query: Optional[Query],
        **kwargs,
    ) -> Union[AsyncGenerator[str, None], str]:  # noqa: DAR101, DAR201, DAR401
        prompt = Prompt(
            messages=messages,
            frequency_penalty=frequency_penalty,
            logit_bias=logit_bias,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            max_completion_tokens=max_completion_tokens,
            n=n,
            presence_penalty=presence_penalty,
            response_format=response_format,
            seed=seed,
            stop=stop,
            temperature=temperature,
            top_p=top_p,
            tools=tools,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
            extra_headers=extra_headers,
            extra_query=extra_query,
            extra_body=kwargs,
        )
        if stream:
            return self._generate_stream(
                self._endpoint,
                prompt,
                # stream
                stream_options=stream_options,
                # platform arguments
                use_custom_keys=use_custom_keys,
                tags=tags,
                drop_params=drop_params,
                region=region,
                log_query_body=log_query_body,
                log_response_body=log_response_body,
                # python client arguments
                return_full_completion=return_full_completion,
            )
        return await self._generate_non_stream(
            self._endpoint,
            prompt,
            # platform arguments
            use_custom_keys=use_custom_keys,
            tags=tags,
            drop_params=drop_params,
            region=region,
            log_query_body=log_query_body,
            log_response_body=log_response_body,
            # python client arguments
            return_full_completion=return_full_completion,
            cache=cache,
            local_cache=local_cache,
        )

    def to_sync_client(self):
        """
        Return a synchronous version of the client (`Unify` instance), with the
        exact same configuration as this asynchronous (`AsyncUnify`) client.

        Returns:
            A `Unify` instance with the same configuration as this `AsyncUnify`
            instance.
        """
        return Unify(**self._constructor_args)

```

`/Users/yushaarif/Unify/unify/unify/universal_api/types/__init__.py`:

```py
from .prompt import *

```

`/Users/yushaarif/Unify/unify/unify/universal_api/types/prompt.py`:

```py
class Prompt:
    def __init__(
        self,
        **components,
    ):
        """
        Create Prompt instance.

        Args:
            components: All components of the prompt.

        Returns:
            The Prompt instance.
        """
        self.components = components

```

`/Users/yushaarif/Unify/unify/unify/universal_api/utils/custom_api_keys.py`:

```py
from typing import Any, Dict, List, Optional

from unify import BASE_URL
from unify.utils import _requests

from ...utils.helpers import _validate_api_key


def create_custom_api_key(
    name: str,
    value: str,
    *,
    api_key: Optional[str] = None,
) -> Dict[str, str]:
    """
    Create a custom API key.

    Args:
        name: Name of the API key.
        value: Value of the API key.
        api_key: If specified, unify API key to be used. Defaults
        to the value in the `UNIFY_KEY` environment variable.

    Returns:
        A dictionary containing the response information.

    """
    api_key = _validate_api_key(api_key)
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    url = f"{BASE_URL}/custom_api_key"

    params = {"name": name, "value": value}

    response = _requests.post(url, headers=headers, params=params)
    if response.status_code != 200:
        raise Exception(response.json())

    return response.json()


def get_custom_api_key(
    name: str,
    *,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Get the value of a custom API key.

    Args:
        name: Name of the API key to get the value for.
        api_key: If specified, unify API key to be used. Defaults
        to the value in the `UNIFY_KEY` environment variable.

    Returns:
        A dictionary containing the custom API key information.

    Raises:
        requests.HTTPError: If the request fails.
    """
    api_key = _validate_api_key(api_key)
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    url = f"{BASE_URL}/custom_api_key"
    params = {"name": name}

    response = _requests.get(url, headers=headers, params=params)
    if response.status_code != 200:
        raise Exception(response.json())

    return response.json()


def delete_custom_api_key(
    name: str,
    *,
    api_key: Optional[str] = None,
) -> Dict[str, str]:
    """
    Delete a custom API key.

    Args:
        name: Name of the custom API key to delete.
        api_key: If specified, unify API key to be used. Defaults
        to the value in the `UNIFY_KEY` environment variable.

    Returns:
        A dictionary containing the response message if successful.

    Raises:
        requests.HTTPError: If the API request fails.
        KeyError: If the API key is not found.
    """
    api_key = _validate_api_key(api_key)
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    url = f"{BASE_URL}/custom_api_key"

    params = {"name": name}

    response = _requests.delete(url, headers=headers, params=params)

    if response.status_code == 200:
        return response.json()
    elif response.status_code == 404:
        raise KeyError("API key not found.")
    else:
        if response.status_code != 200:
            raise Exception(response.json())


def rename_custom_api_key(
    name: str,
    new_name: str,
    *,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Rename a custom API key.

    Args:
        name: Name of the custom API key to be updated.
        new_name: New name for the custom API key.
        api_key: If specified, unify API key to be used. Defaults
                 to the value in the `UNIFY_KEY` environment variable.

    Returns:
        A dictionary containing the response information.

    Raises:
        requests.HTTPError: If the API request fails.
        KeyError: If the API key is not provided or found in environment variables.
    """
    api_key = _validate_api_key(api_key)
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    url = f"{BASE_URL}/custom_api_key/rename"

    params = {"name": name, "new_name": new_name}

    response = _requests.post(url, headers=headers, params=params)
    if response.status_code != 200:
        raise Exception(response.json())

    return response.json()


def list_custom_api_keys(
    *,
    api_key: Optional[str] = None,
) -> List[Dict[str, str]]:
    """
    Get a list of custom API keys associated with the user's account.

    Args:
        api_key: If specified, unify API key to be used. Defaults
        to the value in the `UNIFY_KEY` environment variable.

    Returns:
        A list of dictionaries containing custom API key information.
        Each dictionary has 'name' and 'value' keys.

    """
    api_key = _validate_api_key(api_key)
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    url = f"{BASE_URL}/custom_api_key/list"

    response = _requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(response.json())

    return response.json()

```

`/Users/yushaarif/Unify/unify/unify/universal_api/utils/endpoint_metrics.py`:

```py
import datetime
from typing import Dict, List, Optional, Union

from pydantic import BaseModel
from unify import BASE_URL
from unify.utils import _requests

from ...utils.helpers import _validate_api_key


class Metrics(BaseModel, extra="allow"):
    ttft: Optional[float]
    itl: Optional[float]
    input_cost: Optional[float]
    output_cost: Optional[float]
    measured_at: Union[datetime.datetime, str, Dict[str, Union[datetime.datetime, str]]]


def get_endpoint_metrics(
    endpoint: str,
    *,
    start_time: Optional[Union[datetime.datetime, str]] = None,
    end_time: Optional[Union[datetime.datetime, str]] = None,
    api_key: Optional[str] = None,
) -> List[Metrics]:
    """
    Retrieve the set of cost and speed metrics for the specified endpoint.

    Args:
        endpoint: The endpoint to retrieve the metrics for, in model@provider format

        start_time: Window start time. Only returns the latest benchmark if unspecified.

        end_time: Window end time. Assumed to be the current time if this is unspecified
        and start_time is specified. Only the latest benchmark is returned if both are
        unspecified.

        api_key: If specified, unify API key to be used. Defaults to the value in the
        `UNIFY_KEY` environment variable.

    Returns:
        The set of metrics for the specified endpoint.
    """
    api_key = _validate_api_key(api_key)
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    params = {
        "model": endpoint.split("@")[0],
        "provider": endpoint.split("@")[1],
        "start_time": start_time,
        "end_time": end_time,
    }
    response = _requests.get(
        BASE_URL + "/endpoint-metrics",
        headers=headers,
        params=params,
    )
    if response.status_code != 200:
        raise Exception(response.json())
    return [
        Metrics(
            ttft=metrics_dct["ttft"],
            itl=metrics_dct["itl"],
            input_cost=metrics_dct["input_cost"],
            output_cost=metrics_dct["output_cost"],
            measured_at=metrics_dct["measured_at"],
        )
        for metrics_dct in response.json()
    ]


def log_endpoint_metric(
    endpoint_name: str,
    *,
    metric_name: str,
    value: float,
    measured_at: Optional[Union[str, datetime.datetime]] = None,
    api_key: Optional[str] = None,
) -> Dict[str, str]:
    """
    Append speed or cost data to the standardized time-series benchmarks for a custom
    endpoint (only custom endpoints are publishable by end users).

    Args:
        endpoint_name: Name of the custom endpoint to append benchmark data for.

        metric_name: Name of the metric to submit. Allowed metrics are: “input_cost”,
        “output_cost”, “ttft”, “itl”.

        value: Value of the metric to submit.

        measured_at: The timestamp to associate with the submission. Defaults to current
        time if unspecified.

        api_key: If specified, unify API key to be used. Defaults to the value in the
        `UNIFY_KEY` environment variable.
    """
    api_key = _validate_api_key(api_key)
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    params = {
        "endpoint_name": endpoint_name,
        "metric_name": metric_name,
        "value": value,
        "measured_at": measured_at,
    }
    response = _requests.post(
        BASE_URL + "/endpoint-metrics",
        headers=headers,
        params=params,
    )
    if response.status_code != 200:
        raise Exception(response.json())
    return response.json()


def delete_endpoint_metrics(
    endpoint_name: str,
    *,
    timestamps: Optional[Union[datetime.datetime, List[datetime.datetime]]] = None,
    api_key: Optional[str] = None,
) -> Dict[str, str]:
    api_key = _validate_api_key(api_key)
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    params = {
        "endpoint_name": endpoint_name,
        "timestamps": timestamps,
    }
    response = _requests.delete(
        BASE_URL + "/endpoint-metrics",
        headers=headers,
        params=params,
    )
    if response.status_code != 200:
        raise Exception(response.json())
    return response.json()

```

`/Users/yushaarif/Unify/unify/unify/universal_api/utils/custom_endpoints.py`:

```py
from typing import Any, Dict, List, Optional

from unify import BASE_URL
from unify.utils import _requests

from ...utils.helpers import _validate_api_key


def create_custom_endpoint(
    *,
    name: str,
    url: str,
    key_name: str,
    model_name: Optional[str] = None,
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create a custom endpoint for API calls.

    Args:
        name: Alias for the custom endpoint. This will be the name used to call the endpoint.
        url: Base URL of the endpoint being called. Must support the OpenAI format.
        key_name: Name of the API key that will be passed as part of the query.
        model_name: Name passed to the custom endpoint as model name. If not specified, it will default to the endpoint alias.
        provider: If the custom endpoint is for a fine-tuned model which is hosted directly via one of the supported providers,
                  then this argument should be specified as the provider used.
        api_key: If specified, unify API key to be used. Defaults to the value in the `UNIFY_KEY` environment variable.

    Returns:
        A dictionary containing the response from the API.

    Raises:
        requests.HTTPError: If the API request fails.
        KeyError: If the UNIFY_KEY is not set and no api_key is provided.
    """
    api_key = _validate_api_key(api_key)
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    params = {
        "name": name,
        "url": url,
        "key_name": key_name,
    }

    if model_name:
        params["model_name"] = model_name
    if provider:
        params["provider"] = provider

    response = _requests.post(
        f"{BASE_URL}/custom_endpoint",
        headers=headers,
        params=params,
    )
    if response.status_code != 200:
        raise Exception(response.json())

    return response.json()


def delete_custom_endpoint(
    name: str,
    *,
    api_key: Optional[str] = None,
) -> Dict[str, str]:
    """
    Delete a custom endpoint.

    Args:
        name: Name of the custom endpoint to delete.
        api_key: If specified, unify API key to be used. Defaults
        to the value in the `UNIFY_KEY` environment variable.

    Returns:
        A dictionary containing the response message.

    Raises:
        requests.HTTPError: If the API request fails.
    """
    api_key = _validate_api_key(api_key)
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    url = f"{BASE_URL}/custom_endpoint"

    params = {"name": name}

    response = _requests.delete(url, headers=headers, params=params)
    if response.status_code != 200:
        raise Exception(response.json())

    return response.json()


def rename_custom_endpoint(
    name: str,
    new_name: str,
    *,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Rename a custom endpoint.

    Args:
        name: Name of the custom endpoint to be updated.
        new_name: New name for the custom endpoint.
        api_key: If specified, unify API key to be used. Defaults
                 to the value in the `UNIFY_KEY` environment variable.

    Returns:
        A dictionary containing the response information.

    Raises:
        requests.HTTPError: If the API request fails.
    """
    api_key = _validate_api_key(api_key)
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    url = f"{BASE_URL}/custom_endpoint/rename"

    params = {"name": name, "new_name": new_name}

    response = _requests.post(url, headers=headers, params=params)
    if response.status_code != 200:
        raise Exception(response.json())

    return response.json()


def list_custom_endpoints(
    *,
    api_key: Optional[str] = None,
) -> List[Dict[str, str]]:
    """
    Get a list of custom endpoints for the authenticated user.

    Args:
        api_key: If specified, unify API key to be used. Defaults
        to the value in the `UNIFY_KEY` environment variable.

    Returns:
        A list of dictionaries containing information about custom endpoints.
        Each dictionary has keys: 'name', 'mdl_name', 'url', and 'key'.

    Raises:
        requests.exceptions.RequestException: If the API request fails.
    """
    api_key = _validate_api_key(api_key)
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    url = f"{BASE_URL}/custom_endpoint/list"

    response = _requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(response.json())

    return response.json()

```

`/Users/yushaarif/Unify/unify/unify/universal_api/utils/supported_endpoints.py`:

```py
from typing import List, Optional

from unify import BASE_URL
from unify.utils import _requests

from ...utils.helpers import _res_to_list, _validate_api_key


def list_providers(
    model: Optional[str] = None,
    *,
    api_key: Optional[str] = None,
) -> List[str]:
    """
    Get a list of available providers, either in total or for a specific model.

    Args:
        model: If specified, returns the list of providers supporting this model.
        api_key: If specified, unify API key to be used. Defaults
        to the value in the `UNIFY_KEY` environment variable.

    Returns:
        A list of provider names associated with the model if successful, otherwise an
        empty list.
    Raises:
        BadRequestError: If there was an HTTP error.
        ValueError: If there was an error parsing the JSON response.
    """
    api_key = _validate_api_key(api_key)
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    url = f"{BASE_URL}/providers"
    if model:
        kw = dict(headers=headers, params={"model": model})
    else:
        kw = dict(headers=headers)
    response = _requests.get(url, **kw)
    if response.status_code != 200:
        raise Exception(response.json())
    return _res_to_list(response)


def list_models(
    provider: Optional[str] = None,
    *,
    api_key: Optional[str] = None,
) -> List[str]:
    """
    Get a list of available models, either in total or for a specific provider.

    Args:
        provider: If specified, returns the list of models supporting this provider.
        api_key: If specified, unify API key to be used. Defaults
        to the value in the `UNIFY_KEY` environment variable.

    Returns:
        A list of available model names if successful, otherwise an empty list.
    Raises:
        BadRequestError: If there was an HTTP error.
        ValueError: If there was an error parsing the JSON response.
    """
    api_key = _validate_api_key(api_key)
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    url = f"{BASE_URL}/models"
    if provider:
        kw = dict(headers=headers, params={"provider": provider})
    else:
        kw = dict(headers=headers)
    response = _requests.get(url, **kw)
    if response.status_code != 200:
        raise Exception(response.json())
    return _res_to_list(response)


def list_endpoints(
    model: Optional[str] = None,
    provider: Optional[str] = None,
    *,
    api_key: Optional[str] = None,
) -> List[str]:
    """
    Get a list of available endpoint, either in total or for a specific model or
    provider.

    Args:
        model: If specified, returns the list of endpoint supporting this model.
        provider: If specified, returns the list of endpoint supporting this provider.

        api_key: If specified, unify API key to be used. Defaults to the value in the
        `UNIFY_KEY` environment variable.

    Returns:
        A list of endpoint names if successful, otherwise an empty list.
    Raises:
        BadRequestError: If there was an HTTP error.
        ValueError: If there was an error parsing the JSON response.
    """
    api_key = _validate_api_key(api_key)
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    url = f"{BASE_URL}/endpoints"
    if model and provider:
        raise ValueError("Please specify either model OR provider, not both.")
    elif model:
        kw = dict(headers=headers, params={"model": model})
        return _res_to_list(
            _requests.get(url, headers=headers, params={"model": model}),
        )
    elif provider:
        kw = dict(headers=headers, params={"provider": provider})
    else:
        kw = dict(headers=headers)
    response = _requests.get(url, **kw)
    if response.status_code != 200:
        raise Exception(response.json())
    return _res_to_list(response)

```

`/Users/yushaarif/Unify/unify/unify/universal_api/utils/queries.py`:

```py
import datetime
from typing import Any, Dict, List, Optional, Union

from unify import BASE_URL
from unify.utils import _requests

from ...utils.helpers import _validate_api_key


def get_query_tags(
    *,
    api_key: Optional[str] = None,
) -> List[str]:
    """
    Get a list of available query tags.

    Args:
        api_key: If specified, unify API key to be used. Defaults
        to the value in the `UNIFY_KEY` environment variable.

    Returns:
        A list of available query tags if successful, otherwise an empty list.
    """
    api_key = _validate_api_key(api_key)
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    url = f"{BASE_URL}/tags"
    response = _requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(response.json())

    return response.json()


def get_queries(
    *,
    tags: Optional[Union[str, List[str]]] = None,
    endpoints: Optional[Union[str, List[str]]] = None,
    start_time: Optional[Union[datetime.datetime, str]] = None,
    end_time: Optional[Union[datetime.datetime, str]] = None,
    page_number: Optional[int] = None,
    failures: Optional[Union[bool, str]] = None,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Get query history based on specified filters.

    Args:
        tags: Tags to filter for queries that are marked with these tags.

        endpoints: Optionally specify an endpoint, or a list of endpoints to filter for.

        start_time: Timestamp of the earliest query to aggregate.
        Format is `YYYY-MM-DD hh:mm:ss`.

        end_time: Timestamp of the latest query to aggregate.
        Format is `YYYY-MM-DD hh:mm:ss`.

        page_number: The query history is returned in pages, with up to 100 prompts per
        page. Increase the page number to see older prompts. Default is 1.

        failures: indicates whether to includes failures in the return
        (when set as True), or whether to return failures exclusively
        (when set as ‘only’). Default is False.

        api_key: If specified, unify API key to be used.
        Defaults to the value in the `UNIFY_KEY` environment variable.

    Returns:
        A dictionary containing the query history data.
    """
    api_key = _validate_api_key(api_key)
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    params = {}
    if tags:
        params["tags"] = tags
    if endpoints:
        params["endpoints"] = endpoints
    if start_time:
        params["start_time"] = start_time
    if end_time:
        params["end_time"] = end_time
    if page_number:
        params["page_number"] = page_number
    if failures:
        params["failures"] = failures

    url = f"{BASE_URL}/queries"
    response = _requests.get(url, headers=headers, params=params)
    if response.status_code != 200:
        raise Exception(response.json())

    return response.json()


def log_query(
    *,
    endpoint: str,
    query_body: Dict,
    response_body: Optional[Dict] = None,
    tags: Optional[List[str]] = None,
    timestamp: Optional[Union[datetime.datetime, str]] = None,
    api_key: Optional[str] = None,
):
    """
    Log a query (and optionally response) for a locally deployed (non-Unify-registered)
    model, with tagging (default None) and timestamp (default datetime.now() also
    optionally writeable.

    Args:
        endpoint: Endpoint to log query for.
        query_body: A dict containing the body of the request.
        response_body: An optional dict containing the response to the request.
        tags: Custom tags for later filtering.
        timestamp: A timestamp (if not set, will be the time of sending).
        api_key: If specified, unify API key to be used. Defaults to the value in the `UNIFY_KEY` environment variable.

    Returns:
        A dictionary containing the response message if successful.

    Raises:
        requests.HTTPError: If the API request fails.
    """
    api_key = _validate_api_key(api_key)
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    data = {
        "endpoint": endpoint,
        "query_body": query_body,
        "response_body": response_body,
        "tags": tags,
        "timestamp": timestamp,
    }

    # Remove None values from params
    data = {k: v for k, v in data.items() if v is not None}

    url = f"{BASE_URL}/queries"

    response = _requests.post(url, headers=headers, json=data)
    if response.status_code != 200:
        raise Exception(response.json())

    return response.json()


def get_query_metrics(
    *,
    start_time: Optional[Union[datetime.datetime, str]] = None,
    end_time: Optional[Union[datetime.datetime, str]] = None,
    models: Optional[str] = None,
    providers: Optional[str] = None,
    interval: int = 300,
    secondary_user_id: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Get query metrics for specified parameters.

    Args:
        start_time: Timestamp of the earliest query to aggregate. Format is `YYYY-MM-DD hh:mm:ss`.
        end_time: Timestamp of the latest query to aggregate. Format is `YYYY-MM-DD hh:mm:ss`.
        models: Models to fetch metrics from. Comma-separated string of model names.
        providers: Providers to fetch metrics from. Comma-separated string of provider names.
        interval: Number of seconds in the aggregation interval. Default is 300.
        secondary_user_id: Secondary user id to match the `user` attribute from `/chat/completions`.
        api_key: If specified, unify API key to be used. Defaults to the value in the `UNIFY_KEY` environment variable.

    Returns:
        A dictionary containing the query metrics.
    """
    api_key = _validate_api_key(api_key)
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    params = {
        "start_time": start_time,
        "end_time": end_time,
        "models": models,
        "providers": providers,
        "interval": interval,
        "secondary_user_id": secondary_user_id,
    }

    # Remove None values from params
    params = {k: v for k, v in params.items() if v is not None}

    url = f"{BASE_URL}/metrics"

    response = _requests.get(url, headers=headers, params=params)
    if response.status_code != 200:
        raise Exception(response.json())

    return response.json()

```

`/Users/yushaarif/Unify/unify/unify/universal_api/utils/credits.py`:

```py
from typing import Optional

from unify import BASE_URL
from unify.utils import _requests

from ...utils.helpers import _res_to_list, _validate_api_key


def get_credits(*, api_key: Optional[str] = None) -> float:
    """
    Returns the credits remaining in the user account, in USD.

    Args:
        api_key: If specified, unify API key to be used. Defaults to the value in the
        `UNIFY_KEY` environment variable.

    Returns:
        The credits remaining in USD.
    Raises:
        ValueError: If there was an HTTP error.
    """
    api_key = _validate_api_key(api_key)
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    # Send GET request to the /get_credits endpoint
    response = _requests.get(BASE_URL + "/credits", headers=headers)
    if response.status_code != 200:
        raise Exception(response.json())
    return _res_to_list(response)["credits"]

```

`/Users/yushaarif/Unify/unify/unify/universal_api/casting.py`:

```py
from typing import List, Type, Union

from openai.types.chat import ChatCompletion
from unify.universal_api.types import Prompt

# Upcasting


def _usr_msg_to_prompt(user_message: str) -> Prompt:
    return Prompt(user_message)


def _bool_to_float(boolean: bool) -> float:
    return float(boolean)


# Downcasting


def _prompt_to_usr_msg(prompt: Prompt) -> str:
    return prompt.messages[-1]["content"]


def _chat_completion_to_assis_msg(chat_completion: ChatCompletion) -> str:
    return chat_completion.choices[0].message.content


def _float_to_bool(float_in: float) -> bool:
    return bool(float_in)


# Cast Dict

_CAST_DICT = {
    str: {Prompt: _usr_msg_to_prompt},
    Prompt: {
        str: _prompt_to_usr_msg,
    },
    ChatCompletion: {str: _chat_completion_to_assis_msg},
    bool: {
        float: _bool_to_float,
    },
    float: {
        bool: _float_to_bool,
    },
}


def _cast_from_selection(
    inp: Union[str, bool, float, Prompt, ChatCompletion],
    targets: List[Union[float, Prompt, ChatCompletion]],
) -> Union[str, bool, float, Prompt, ChatCompletion]:
    """
    Upcasts the input if possible, based on the permitted upcasting targets provided.

    Args:
        inp: The input to cast.

        targets: The set of permitted upcasting targets.

    Returns:
        The input after casting to the new type, if it was possible.
    """
    input_type = type(inp)
    assert input_type in _CAST_DICT, (
        "Cannot upcast input {} of type {}, because this type is not in the "
        "_CAST_DICT, meaning there are no functions for casting this type."
    )
    cast_fns = _CAST_DICT[input_type]
    targets = [target for target in targets if target in cast_fns]
    assert len(targets) == 1, "There must be exactly one valid casting target."
    to_type = targets[0]
    return cast_fns[to_type](inp)


# Public function


def cast(
    inp: Union[str, bool, float, Prompt, ChatCompletion],
    to_type: Union[
        Type[Union[str, bool, float, Prompt, ChatCompletion]],
        List[Type[Union[str, bool, float, Prompt, ChatCompletion]]],
    ],
) -> Union[str, bool, float, Prompt, ChatCompletion]:
    """
    Cast the input to the specified type.

    Args:
        inp: The input to cast.

        to_type: The type to cast the input to.

    Returns:
        The input after casting to the new type.
    """
    if isinstance(to_type, list):
        return _cast_from_selection(inp, to_type)
    input_type = type(inp)
    if input_type is to_type:
        return inp
    return _CAST_DICT[input_type][to_type](inp)


def try_cast(
    inp: Union[str, bool, float, Prompt, ChatCompletion],
    to_type: Union[
        Type[Union[str, bool, float, Prompt, ChatCompletion]],
        List[Type[Union[str, bool, float, Prompt, ChatCompletion]]],
    ],
) -> Union[str, bool, float, Prompt, ChatCompletion]:
    # noinspection PyBroadException
    try:
        return cast(inp, to_type)
    except:
        return inp

```

`/Users/yushaarif/Unify/unify/unify/universal_api/chatbot.py`:

```py
import asyncio
import sys
from typing import Dict, Union

import unify
from unify.universal_api.clients import _Client, _MultiClient, _UniClient


class ChatBot:  # noqa: WPS338
    """Agent class represents an LLM chat agent."""

    def __init__(
        self,
        client: _Client,
    ) -> None:
        """
        Initializes the ChatBot object, wrapped around a client.

        Args:
            client: The Client instance to wrap the chatbot logic around.
        """
        self._paused = False
        assert not client.return_full_completion, (
            "ChatBot currently only supports clients which only generate the message "
            "content in the return"
        )
        self._client = client
        self.clear_chat_history()

    @property
    def client(self) -> _Client:
        """
        Get the client object.  # noqa: DAR201.

        Returns:
            The client.
        """
        return self._client

    def set_client(self, value: client) -> None:
        """
        Set the client.  # noqa: DAR101.

        Args:
            value: The unify client.
        """
        if isinstance(value, _Client):
            self._client = value
        else:
            raise Exception("Invalid client!")

    def _get_credits(self) -> float:
        """
        Retrieves the current credit balance from associated with the UNIFY account.

        Returns:
            Current credit balance.
        """
        return self._client.get_credit_balance()

    def _update_message_history(
        self,
        role: str,
        content: Union[str, Dict[str, str]],
    ) -> None:
        """
        Updates message history with user input.

        Args:
            role: Either "assistant" or "user".
            content: User input message.
        """
        if isinstance(self._client, _UniClient):
            self._client.messages.append(
                {
                    "role": role,
                    "content": content,
                },
            )
        elif isinstance(self._client, _MultiClient):
            if isinstance(content, str):
                content = {endpoint: content for endpoint in self._client.endpoints}
            for endpoint, cont in content.items():
                self._client.messages[endpoint].append(
                    {
                        "role": role,
                        "content": cont,
                    },
                )
        else:
            raise Exception(
                "client must either be a UniClient or MultiClient instance.",
            )

    def clear_chat_history(self) -> None:
        """Clears the chat history."""
        if isinstance(self._client, _UniClient):
            self._client.set_messages([])
        elif isinstance(self._client, _MultiClient):
            self._client.set_messages(
                {endpoint: [] for endpoint in self._client.endpoints},
            )
        else:
            raise Exception(
                "client must either be a UniClient or MultiClient instance.",
            )

    @staticmethod
    def _stream_response(response) -> str:
        words = ""
        for chunk in response:
            words += chunk
            sys.stdout.write(chunk)
            sys.stdout.flush()
        sys.stdout.write("\n")
        return words

    def _handle_uni_llm_response(
        self,
        response: str,
        endpoint: Union[bool, str],
    ) -> str:
        if endpoint:
            endpoint = self._client.endpoint if endpoint is True else endpoint
            sys.stdout.write(endpoint + ":\n")
        if self._client.stream:
            words = self._stream_response(response)
        else:
            words = response
            sys.stdout.write(words)
            sys.stdout.write("\n\n")
        return words

    def _handle_multi_llm_response(self, response: Dict[str, str]) -> Dict[str, str]:
        for endpoint, resp in response.items():
            self._handle_uni_llm_response(resp, endpoint)
        return response

    def _handle_response(
        self,
        response: Union[str, Dict[str, str]],
        show_endpoint: bool,
    ) -> None:
        if isinstance(self._client, _UniClient):
            response = self._handle_uni_llm_response(response, show_endpoint)
        elif isinstance(self._client, _MultiClient):
            response = self._handle_multi_llm_response(response)
        else:
            raise Exception(
                "client must either be a UniClient or MultiClient instance.",
            )
        self._update_message_history(
            role="assistant",
            content=response,
        )

    def run(self, show_credits: bool = False, show_endpoint: bool = False) -> None:
        """
        Starts the chat interaction loop.

        Args:
            show_credits: Whether to show credit consumption. Defaults to False.
            show_endpoint: Whether to show the endpoint used. Defaults to False.
        """
        if not self._paused:
            sys.stdout.write(
                "Let's have a chat. (Enter `pause` to pause and `quit` to exit)\n",
            )
            self.clear_chat_history()
        else:
            sys.stdout.write(
                "Welcome back! (Remember, enter `pause` to pause and `quit` to exit)\n",
            )
        self._paused = False
        while True:
            sys.stdout.write("> ")
            inp = input()
            if inp == "quit":
                self.clear_chat_history()
                break
            elif inp == "pause":
                self._paused = True
                break
            self._update_message_history(role="user", content=inp)
            initial_credit_balance = self._get_credits()
            if isinstance(self._client, unify.AsyncUnify):
                response = asyncio.run(self._client.generate())
            else:
                response = self._client.generate()
            self._handle_response(response, show_endpoint)
            final_credit_balance = self._get_credits()
            if show_credits:
                sys.stdout.write(
                    "\n(spent {:.6f} credits)".format(
                        initial_credit_balance - final_credit_balance,
                    ),
                )

```

`/Users/yushaarif/Unify/unify/unify/universal_api/usage.py`:

```py
import datetime
from typing import List, Optional

import unify

from ..utils.helpers import _validate_api_key


def with_logging(
    model_fn: Optional[callable] = None,
    *,
    endpoint: str,
    tags: Optional[List[str]] = None,
    timestamp: Optional[datetime.datetime] = None,
    log_query_body: bool = True,
    log_response_body: bool = True,
    api_key: Optional[str] = None,
):
    """
    Wrap a local model callable with logging of the queries.

    Args:
        model_fn: The model callable to wrap logging around.
        endpoint: The endpoint name to give to this local callable.
        tags: Tags for later filtering.
        timestamp: A timestamp (if not set, will be the time of sending).
        log_query_body: Whether or not to log the query body.
        log_response_body: Whether or not to log the response body.
        api_key: If specified, unify API key to be used. Defaults to the value in the `UNIFY_KEY` environment variable.

    Returns:
        A new callable, but with logging added every time the function is called.

    Raises:
        requests.HTTPError: If the API request fails.
    """
    _tags = tags
    _timestamp = timestamp
    _log_query_body = log_query_body
    _log_response_body = log_response_body
    api_key = _validate_api_key(api_key)

    # noinspection PyShadowingNames
    def model_fn_w_logging(
        *args,
        tags: Optional[List[str]] = None,
        timestamp: Optional[datetime.datetime] = None,
        log_query_body: bool = True,
        log_response_body: bool = True,
        **kwargs,
    ):
        if len(args) != 0:
            raise Exception(
                "When logging queries for a local model, all arguments to "
                "the model callable must be provided as keyword arguments. "
                "Positional arguments are not supported. This is so the "
                "query body dict can be fully populated with keys for each "
                "entry.",
            )
        query_body = kwargs
        response = model_fn(**query_body)
        if not isinstance(response, dict):
            response = {"response": response}
        kw = dict(
            endpoint=endpoint,
            query_body=query_body,
            response_body=response,
            tags=tags,
            timestamp=timestamp,
            api_key=api_key,
        )
        if log_query_body:
            if not log_response_body:
                del kw["response_body"]
            unify.log_query(**kw)
        return response

    return model_fn_w_logging

```

`/Users/yushaarif/Unify/unify/unify/__init__.py`:

```py
"""Unify python module."""

import os
from typing import Callable, Optional

import dotenv

dotenv.load_dotenv()
if "UNIFY_BASE_URL" in os.environ.keys():
    BASE_URL = os.environ["UNIFY_BASE_URL"]
else:
    BASE_URL = "https://api.unify.ai/v0"


CLIENT_LOGGING = False
LOCAL_MODELS = dict()
SEED = None
UNIFY_DIR = os.path.dirname(__file__)


def set_seed(seed: int) -> None:
    global SEED
    SEED = seed


def get_seed() -> Optional[int]:
    return SEED


def register_local_model(model_name: str, fn: Callable):
    if "@local" not in model_name:
        model_name += "@local"
    LOCAL_MODELS[model_name] = fn


from unify.universal_api.clients.multi_llm import *
from unify.universal_api.clients.uni_llm import *

from .logging import dataset, logs
from .logging.dataset import *
from .logging.logs import *
from .logging.utils import (artifacts, compositions, contexts, datasets, logs,
                            projects)
from .logging.utils.artifacts import *
from .logging.utils.compositions import *
from .logging.utils.contexts import *
from .logging.utils.datasets import *
from .logging.utils.logs import *
from .logging.utils.projects import *
from .universal_api import casting, chatbot, clients, types, usage
from .universal_api.casting import *
from .universal_api.chatbot import *
from .universal_api.clients import multi_llm
from .universal_api.types import *
from .universal_api.usage import *
from .universal_api.utils import (credits, custom_api_keys, custom_endpoints,
                                  endpoint_metrics, queries,
                                  supported_endpoints)
from .universal_api.utils.credits import *
from .universal_api.utils.custom_api_keys import *
from .universal_api.utils.custom_endpoints import *
from .universal_api.utils.endpoint_metrics import *
from .universal_api.utils.queries import *
from .universal_api.utils.supported_endpoints import *
from .utils import _caching, get_map_mode, helpers, map, set_map_mode
from .utils._caching import (cache_file_intersection, cache_file_union, cached,
                             set_caching, set_caching_fname,
                             subtract_cache_files)

# Project #
# --------#

PROJECT: Optional[str] = None


# noinspection PyShadowingNames
def activate(project: str, overwrite: bool = False, api_key: str = None) -> None:
    if project not in list_projects(api_key=api_key):
        create_project(project, api_key=api_key)
    elif overwrite:
        create_project(project, api_key=api_key, overwrite=True)
    global PROJECT
    PROJECT = project


def deactivate() -> None:
    global PROJECT
    PROJECT = None


def active_project() -> str:
    global PROJECT
    if PROJECT is None:
        return os.environ.get("UNIFY_PROJECT")
    return PROJECT


class Project:

    # noinspection PyShadowingNames
    def __init__(
        self,
        project: str,
        overwrite: bool = False,
        api_key: Optional[str] = None,
    ) -> None:
        self._project = project
        self._overwrite = overwrite
        # noinspection PyProtectedMember
        self._api_key = helpers._validate_api_key(api_key)
        self._entered = False

    def create(self) -> None:
        create_project(self._project, overwrite=self._overwrite, api_key=self._api_key)

    def delete(self):
        delete_project(self._project, api_key=self._api_key)

    def rename(self, new_name: str):
        rename_project(self._project, new_name, api_key=self._api_key)
        self._project = new_name
        if self._entered:
            activate(self._project)

    def __enter__(self):
        activate(self._project)
        if self._project not in list_projects(api_key=self._api_key) or self._overwrite:
            self.create()
        self._entered = True

    def __exit__(self, exc_type, exc_val, exc_tb):
        deactivate()
        self._entered = False

```

`/Users/yushaarif/Unify/unify/unify/utils/__init__.py`:

```py
from . import helpers
from .map import *

```

`/Users/yushaarif/Unify/unify/unify/utils/map.py`:

```py
import asyncio
import contextvars
import threading
from typing import Any, List

from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

MAP_MODE = "threading"


def set_map_mode(mode: str):
    global MAP_MODE
    MAP_MODE = mode


def get_map_mode() -> str:
    return MAP_MODE


def _is_iterable(item: Any) -> bool:
    try:
        iter(item)
        return True
    except TypeError:
        return False


# noinspection PyShadowingBuiltins
def map(
    fn: callable,
    *args,
    mode=None,
    name="",
    from_args=False,
    raise_exceptions=True,
    **kwargs,
) -> Any:

    if name:
        name = (
            " ".join(substr[0].upper() + substr[1:] for substr in name.split("_")) + " "
        )

    if mode is None:
        mode = get_map_mode()

    assert mode in (
        "threading",
        "asyncio",
        "loop",
    ), "map mode must be one of threading, asyncio or loop."

    def fn_w_exception_handling(*a, **kw):
        try:
            return fn(*a, **kw)
        except Exception as e:
            if raise_exceptions:
                raise e

    if from_args:
        args = list(args)
        for i, a in enumerate(args):
            if _is_iterable(a):
                args[i] = list(a)

        if args:
            num_calls = len(args[0])
        else:
            for v in kwargs.values():
                if isinstance(v, list):
                    num_calls = len(v)
                    break
            else:
                raise Exception(
                    "At least one of the args or kwargs must be a list, "
                    "which is to be mapped across the threads",
                )
        args_n_kwargs = [
            (
                tuple(a[i] for a in args),
                {
                    k: v[i] if (isinstance(v, list) or isinstance(v, tuple)) else v
                    for k, v in kwargs.items()
                },
            )
            for i in range(num_calls)
        ]
    else:
        args_n_kwargs = args[0]
        if not isinstance(args_n_kwargs[0], tuple):
            if isinstance(args_n_kwargs[0], dict):
                args_n_kwargs = [((), item) for item in args_n_kwargs]
            else:
                args_n_kwargs = [((item,), {}) for item in args_n_kwargs]
        elif (
            not isinstance(args_n_kwargs[0][0], tuple)
            or len(args_n_kwargs[0]) < 2
            or not isinstance(args_n_kwargs[0][1], dict)
        ):
            args_n_kwargs = [(item, {}) for item in args_n_kwargs]
        num_calls = len(args_n_kwargs)

    if mode == "loop":

        pbar = tqdm(total=num_calls)
        pbar.set_description(f"{name}Iterations")

        returns = list()
        for a, kw in args_n_kwargs:
            ret = fn_w_exception_handling(*a, **kw)
            returns.append(ret)
            pbar.update(1)
        pbar.close()
        return returns

    elif mode == "threading":

        pbar = tqdm(total=num_calls)
        pbar.set_description(f"{name}Threads")

        def fn_w_indexing(rets: List[None], thread_idx: int, *a, **kw):
            for var, value in kw["context"].items():
                var.set(value)
            del kw["context"]
            ret = fn_w_exception_handling(*a, **kw)
            pbar.update(1)
            rets[thread_idx] = ret

        threads = list()
        returns = [None] * num_calls
        for i, a_n_kw in enumerate(args_n_kwargs):
            a, kw = a_n_kw
            kw["context"] = contextvars.copy_context()
            thread = threading.Thread(
                target=fn_w_indexing,
                args=(returns, i, *a),
                kwargs=kw,
            )
            thread.start()
            threads.append(thread)
        [thread.join() for thread in threads]
        pbar.close()
        return returns

    def _run_asyncio_in_thread(ret):
        asyncio.set_event_loop(asyncio.new_event_loop())
        MAX_WORKERS = 100
        semaphore = asyncio.Semaphore(MAX_WORKERS)
        fns = []

        async def fn_wrapper(*args, **kwargs):
            async with semaphore:
                return await asyncio.to_thread(fn_w_exception_handling, *args, **kwargs)

        for _, a_n_kw in enumerate(args_n_kwargs):
            a, kw = a_n_kw
            fns.append(fn_wrapper(*a, **kw))

        async def main(fns):
            return await tqdm_asyncio.gather(*fns, desc=f"{name}Coroutines")

        ret += asyncio.run(main(fns))

    ret = []
    thread = threading.Thread(target=_run_asyncio_in_thread, args=(ret,))
    thread.start()
    thread.join()
    return ret

```

`/Users/yushaarif/Unify/unify/unify/utils/_caching.py`:

```py
import difflib
import inspect
import json
import os
import threading
from typing import Any, Dict, List, Optional, Union

from openai.types.chat import ChatCompletion, ParsedChatCompletion
from pydantic import BaseModel

_cache: Optional[Dict] = None
_cache_dir = (
    os.environ["UNIFY_CACHE_DIR"] if "UNIFY_CACHE_DIR" in os.environ else os.getcwd()
)
_cache_fpath: str = os.path.join(_cache_dir, ".cache.json")

CACHE_LOCK = threading.Lock()

CACHING = False
CACHE_FNAME = ".cache.json"
UPSTREAM_CACHE_CONTEXT_NAME = "UNIFY_CACHE"


def set_caching(value: bool) -> None:
    global CACHING, CACHE_FNAME
    CACHING = value


def set_caching_fname(value: Optional[str] = None) -> None:
    global CACHE_FNAME, _cache
    if value is not None:
        CACHE_FNAME = value
    else:
        CACHE_FNAME = ".cache.json"
    _cache = None  # Force a reload of the cache


def _get_caching():
    return CACHING


def _get_caching_fname():
    return CACHE_FNAME


def _get_caching_fpath():
    global _cache_dir, CACHE_FNAME
    return os.path.join(_cache_dir, CACHE_FNAME)


def _create_cache_if_none(filename: str = None, local: bool = True):
    from unify import create_context, get_contexts

    if not local:
        if UPSTREAM_CACHE_CONTEXT_NAME not in get_contexts():
            create_context(UPSTREAM_CACHE_CONTEXT_NAME)
        return

    global _cache, _cache_fpath, _cache_dir
    if filename is None:
        cache_fpath = _get_caching_fpath()
    else:
        cache_fpath = os.path.join(_cache_dir, filename)
    if _cache is None:
        if not os.path.exists(cache_fpath):
            with open(cache_fpath, "w") as outfile:
                json.dump({}, outfile)
        with open(cache_fpath) as outfile:
            _cache = json.load(outfile)


def _minimal_char_diff(a: str, b: str, context: int = 5) -> str:
    matcher = difflib.SequenceMatcher(None, a, b)
    diff_parts = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            segment = a[i1:i2]
            # If the segment is too long, show only a context at the beginning and end.
            if len(segment) > 2 * context:
                diff_parts.append(segment[:context] + "..." + segment[-context:])
            else:
                diff_parts.append(segment)
        elif tag == "replace":
            diff_parts.append(f"[{a[i1:i2]}|{b[j1:j2]}]")
        elif tag == "delete":
            diff_parts.append(f"[-{a[i1:i2]}-]")
        elif tag == "insert":
            diff_parts.append(f"[+{b[j1:j2]}+]")

    return "".join(diff_parts)


def _get_filter_expr(cache_key: str):
    return f"key == {json.dumps(cache_key)}"


def _get_entry_from_cache(cache_key: str, local: bool = True):
    from unify import get_logs

    value = res_types = None

    if local:
        if cache_key in _cache:
            value = json.loads(_cache[cache_key])
        if cache_key + "_res_types" in _cache:
            res_types = _cache[cache_key + "_res_types"]
    else:
        logs = get_logs(
            context=UPSTREAM_CACHE_CONTEXT_NAME,
            filter=_get_filter_expr(cache_key),
        )
        if len(logs) > 0:
            entry = logs[0].entries
            value = json.loads(entry["value"])
            if "res_types" in entry:
                res_types = json.loads(entry["res_types"])

    return value, res_types


def _is_key_in_cache(cache_key: str, local: bool = True):
    from unify import get_logs

    if local:
        return cache_key in _cache
    else:
        logs = get_logs(
            context=UPSTREAM_CACHE_CONTEXT_NAME,
            filter=_get_filter_expr(cache_key),
        )
        return len(logs) > 0


def _delete_from_cache(cache_str: str, local: bool = True):
    from unify.logging.logs import delete_logs, get_logs

    if local:
        del _cache[cache_str]
        del _cache[cache_str + "_res_types"]
    else:
        logs = get_logs(
            context=UPSTREAM_CACHE_CONTEXT_NAME,
            filter=_get_filter_expr(cache_str),
        )
        delete_logs(context=UPSTREAM_CACHE_CONTEXT_NAME, logs=logs)


def _get_cache_keys(local: bool = True):
    from unify import get_logs

    if local:
        return list(_cache.keys())
    else:
        logs = get_logs(context=UPSTREAM_CACHE_CONTEXT_NAME)
        return [log.entries["key"] for log in logs]


# noinspection PyTypeChecker,PyUnboundLocalVariable
def _get_cache(
    fn_name: str,
    kw: Dict[str, Any],
    filename: str = None,
    raise_on_empty: bool = False,
    read_closest: bool = False,
    delete_closest: bool = False,
    local: bool = True,
) -> Optional[Any]:
    global CACHE_LOCK
    # prevents circular import
    from unify.logging.logs import Log

    type_str_to_type = {
        "ChatCompletion": ChatCompletion,
        "Log": Log,
        "ParsedChatCompletion": ParsedChatCompletion,
    }
    CACHE_LOCK.acquire()
    # noinspection PyBroadException
    try:
        _create_cache_if_none(filename, local)
        kw = {k: v for k, v in kw.items() if v is not None}
        kw_str = _dumps(kw)
        cache_str = fn_name + "_" + kw_str
        if not _is_key_in_cache(cache_str, local):
            if raise_on_empty or read_closest:
                keys_to_search = _get_cache_keys(local)
                closest_match = difflib.get_close_matches(
                    cache_str,
                    keys_to_search,
                    n=1,
                    cutoff=0,
                )[0]
                minimal_char_diff = _minimal_char_diff(cache_str, closest_match)
                if read_closest:
                    cache_str = closest_match
                else:
                    CACHE_LOCK.release()
                    raise Exception(
                        f"Failed to get cache for function {fn_name} with kwargs {_dumps(kw, indent=4)} "
                        f"from cache at {filename}. \n\nCorresponding key\n{cache_str}\nwas not found in the cache.\n\n"
                        f"The closest match is:\n{closest_match}\n\n"
                        f"The contracted diff is:\n{minimal_char_diff}\n\n",
                    )
            else:
                CACHE_LOCK.release()
                return
        ret, res_types = _get_entry_from_cache(cache_str, local)
        if res_types is None:
            CACHE_LOCK.release()
            return ret
        for idx_str, type_str in res_types.items():
            type_str = type_str.split("[")[0]
            idx_list = json.loads(idx_str)
            if len(idx_list) == 0:
                if read_closest and delete_closest:
                    _delete_from_cache(cache_str, local)
                CACHE_LOCK.release()
                typ = type_str_to_type[type_str]
                if issubclass(typ, BaseModel):
                    return type_str_to_type[type_str](**ret)
                elif issubclass(typ, Log):
                    return type_str_to_type[type_str].from_json(ret)
                raise Exception(f"Cache indexing found for unsupported type: {typ}")
            item = ret
            for i, idx in enumerate(idx_list):
                if i == len(idx_list) - 1:
                    typ = type_str_to_type[type_str]
                    if issubclass(typ, BaseModel) or issubclass(typ, Log):
                        item[idx] = type_str_to_type[type_str].from_json(item[idx])
                    else:
                        raise Exception(
                            f"Cache indexing found for unsupported type: {typ}",
                        )
                    break
                item = item[idx]
        if read_closest and delete_closest:
            _delete_from_cache(cache_str, local)
        CACHE_LOCK.release()
        return ret
    except:
        if CACHE_LOCK.locked():
            CACHE_LOCK.release()
        raise Exception(
            f"Failed to get cache for function {fn_name} with kwargs {kw} "
            f"from cache at {filename}",
        )


def _dumps(
    obj: Any,
    cached_types: Dict[str, str] = None,
    idx: List[Union[str, int]] = None,
    indent: int = None,
) -> Any:
    # prevents circular import
    from unify.logging.logs import Log

    base = False
    if idx is None:
        base = True
        idx = list()
    if isinstance(obj, BaseModel):
        if cached_types is not None:
            cached_types[json.dumps(idx, indent=indent)] = obj.__class__.__name__
        ret = obj.model_dump()
    elif inspect.isclass(obj) and issubclass(obj, BaseModel):
        ret = obj.schema_json()
    elif isinstance(obj, Log):
        if cached_types is not None:
            cached_types[json.dumps(idx, indent=indent)] = obj.__class__.__name__
        ret = obj.to_json()
    elif isinstance(obj, dict):
        ret = {k: _dumps(v, cached_types, idx + ["k"]) for k, v in obj.items()}
    elif isinstance(obj, list):
        ret = [_dumps(v, cached_types, idx + [i]) for i, v in enumerate(obj)]
    elif isinstance(obj, tuple):
        ret = tuple(_dumps(v, cached_types, idx + [i]) for i, v in enumerate(obj))
    else:
        ret = obj
    return json.dumps(ret, indent=indent) if base else ret


# noinspection PyTypeChecker,PyUnresolvedReferences
def _write_to_cache(
    fn_name: str,
    kw: Dict[str, Any],
    response: Any,
    local: bool = True,
    filename: str = None,
):

    global CACHE_LOCK
    CACHE_LOCK.acquire()
    # noinspection PyBroadException
    try:
        _create_cache_if_none(filename, local)
        kw = {k: v for k, v in kw.items() if v is not None}
        kw_str = _dumps(kw)
        cache_str = fn_name + "_" + kw_str
        _res_types = {}
        response_str = _dumps(response, _res_types)
        if local:
            if _res_types:
                _cache[cache_str + "_res_types"] = _res_types
            _cache[cache_str] = response_str
            if filename is None:
                cache_fpath = _get_caching_fpath()
            else:
                cache_fpath = os.path.join(_cache_dir, filename)
            with open(cache_fpath, "w") as outfile:
                json.dump(_cache, outfile)
        else:
            # prevents circular import
            from unify.logging.logs import delete_logs, get_logs, log

            logs = get_logs(
                context=UPSTREAM_CACHE_CONTEXT_NAME,
                filter=_get_filter_expr(cache_str),
            )
            if len(logs) > 0:
                delete_logs(logs=logs, context=UPSTREAM_CACHE_CONTEXT_NAME)

            entries = {
                "value": response_str,
            }
            if _res_types:
                entries["res_types"] = json.dumps(_res_types)
            log(key=cache_str, context=UPSTREAM_CACHE_CONTEXT_NAME, **entries)
        CACHE_LOCK.release()
    except:
        CACHE_LOCK.release()
        raise Exception(
            f"Failed to write function {fn_name} with kwargs {kw} and "
            f"response {response} to cache at {filename}",
        )


# Decorators #
# -----------#


def cached(
    fn: callable = None,
    *,
    mode: Union[bool, str] = True,
    local: bool = True,
):
    if fn is None:
        return lambda f: cached(
            f,
            mode=mode,
            local=local,
        )

    def wrapped(*args, **kwargs):
        nonlocal mode
        if isinstance(mode, str) and mode.endswith("-closest"):
            mode = mode.removesuffix("-closest")
            read_closest = True
        else:
            read_closest = False
        in_cache = False
        ret = None
        if mode in [True, "both", "read", "read-only"]:
            ret = _get_cache(
                fn_name=fn.__name__,
                kw=kwargs,
                raise_on_empty=mode == "read-only",
                read_closest=read_closest,
                delete_closest=read_closest,
                local=local,
            )
            in_cache = True if ret is not None else False
        if ret is None:
            ret = fn(*args, **kwargs)
        if (ret is not None or read_closest) and mode in [
            True,
            "both",
            "write",
        ]:
            if not in_cache or mode == "write":
                _write_to_cache(
                    fn_name=fn.__name__,
                    kw=kwargs,
                    response=ret,
                    local=local,
                )
        return ret

    return wrapped


# File Manipulation #
# ------------------#


def cache_file_union(
    first_cache_fpath: str,
    second_cache_fpath: str,
    target_cache_fpath: str,
    conflict_mode="raise",
):
    with open(first_cache_fpath, "r") as file:
        first_cache = json.load(file)
    with open(second_cache_fpath, "r") as file:
        second_cache = json.load(file)
    if conflict_mode == "raise":
        for key, value in first_cache.items():
            if key in second_cache:
                assert second_cache[key] == value, (
                    f"key {key} found in both caches, but values conflict:"
                    f"{first_cache_fpath} had value: {value}"
                    f"{second_cache_fpath} had value: {second_cache[key]}"
                )
        union_cache = {**first_cache, **second_cache}
    elif conflict_mode == "first_overrides":
        union_cache = {**second_cache, **first_cache}
    elif conflict_mode == "second_overrides":
        union_cache = {**first_cache, **second_cache}
    else:
        raise Exception(
            "Invalud conflict_mode, must be one of: 'raise', 'first_overrides' or 'second_overrides'",
        )
    with open(target_cache_fpath, "w+") as file:
        json.dump(union_cache, file)


def cache_file_intersection(
    first_cache_fpath: str,
    second_cache_fpath: str,
    target_cache_fpath: str,
    conflict_mode="raise",
):
    with open(first_cache_fpath, "r") as file:
        first_cache = json.load(file)
    with open(second_cache_fpath, "r") as file:
        second_cache = json.load(file)
    if conflict_mode == "raise":
        for key, value in first_cache.items():
            if key in second_cache:
                assert second_cache[key] == value, (
                    f"key {key} found in both caches, but values conflict:"
                    f"{first_cache_fpath} had value: {value}"
                    f"{second_cache_fpath} had value: {second_cache[key]}"
                )
        intersection_cache = {k: v for k, v in first_cache.items() if k in second_cache}
    elif conflict_mode == "first_overrides":
        intersection_cache = {k: v for k, v in first_cache.items() if k in second_cache}
    elif conflict_mode == "second_overrides":
        intersection_cache = {k: v for k, v in second_cache.items() if k in first_cache}
    else:
        raise Exception(
            "Invalud conflict_mode, must be one of: 'raise', 'first_overrides' or 'second_overrides'",
        )
    with open(target_cache_fpath, "w+") as file:
        json.dump(intersection_cache, file)


def subtract_cache_files(
    first_cache_fpath: str,
    second_cache_fpath: str,
    target_cache_fpath: str,
    raise_on_conflict=True,
):
    with open(first_cache_fpath, "r") as file:
        first_cache = json.load(file)
    with open(second_cache_fpath, "r") as file:
        second_cache = json.load(file)
    if raise_on_conflict:
        for key, value in first_cache.items():
            if key in second_cache:
                assert second_cache[key] == value, (
                    f"key {key} found in both caches, but values conflict:"
                    f"{first_cache_fpath} had value: {value}"
                    f"{second_cache_fpath} had value: {second_cache[key]}"
                )
    final_cache = {k: v for k, v in first_cache.items() if k not in second_cache}
    with open(target_cache_fpath, "w+") as file:
        json.dump(final_cache, file)

```

`/Users/yushaarif/Unify/unify/unify/utils/helpers.py`:

```py
import inspect
import json
import os
import threading
from typing import Any, Dict, List, Optional, Tuple, Union

import requests
import unify
from pydantic import BaseModel, ValidationError

PROJECT_LOCK = threading.Lock()


class RequestError(Exception):
    def __init__(self, response: requests.Response):
        req = response.request
        message = (
            f"{req.method} {req.url} failed with status code {response.status_code}. "
            f"Request body: {req.body}, Response: {response.text}"
        )
        super().__init__(message)
        self.response = response


def _check_response(response: requests.Response):
    if not response.ok:
        raise RequestError(response)


def _res_to_list(response: requests.Response) -> Union[List, Dict]:
    return json.loads(response.text)


def _validate_api_key(api_key: Optional[str]) -> str:
    if api_key is None:
        api_key = os.environ.get("UNIFY_KEY")
    if api_key is None:
        raise KeyError(
            "UNIFY_KEY is missing. Please make sure it is set correctly!",
        )
    return api_key


def _default(value: Any, default_value: Any) -> Any:
    return value if value is not None else default_value


def _dict_aligns_with_pydantic(dict_in: Dict, pydantic_cls: type(BaseModel)) -> bool:
    try:
        pydantic_cls.model_validate(dict_in)
        return True
    except ValidationError:
        return False


def _make_json_serializable(
    item: Union[Dict, List, Tuple],
) -> Union[Dict, List, Tuple]:
    if isinstance(item, list):
        return [_make_json_serializable(i) for i in item]
    elif isinstance(item, dict):
        return {k: _make_json_serializable(v) for k, v in item.items()}
    elif isinstance(item, tuple):
        return tuple(_make_json_serializable(i) for i in item)
    elif inspect.isclass(item) and issubclass(item, BaseModel):
        return item.schema()
    elif isinstance(item, BaseModel):
        return item.dict()
    elif hasattr(item, "json") and callable(item.json):
        return _make_json_serializable(item.json())
    else:
        try:
            json.dumps(item)
            return item
        except:
            return str(item)


def _get_and_maybe_create_project(
    project: Optional[str] = None,
    required: bool = True,
    api_key: Optional[str] = None,
    create_if_missing: bool = True,
) -> Optional[str]:
    # noinspection PyUnresolvedReferences
    from unify.logging.utils.logs import ASYNC_LOGGING

    api_key = _validate_api_key(api_key)
    if project is None:
        project = unify.active_project()
        if project is None:
            if required:
                project = "_"
            else:
                return None
    if not create_if_missing:
        return project
    if ASYNC_LOGGING:
        # acquiring the project lock here will block the async logger
        # so we skip the lock if we are in async mode
        return project
    with PROJECT_LOCK:
        if project not in unify.list_projects(api_key=api_key):
            unify.create_project(project, api_key=api_key)
    return project


def _prune_dict(val):
    def keep(v):
        if v in (None, "NOT_GIVEN"):
            return False
        else:
            ret = _prune_dict(v)
            if isinstance(ret, dict) or isinstance(ret, list) or isinstance(ret, tuple):
                return bool(ret)
            return True

    if (
        not isinstance(val, dict)
        and not isinstance(val, list)
        and not isinstance(val, tuple)
    ):
        return val
    elif isinstance(val, dict):
        return {k: _prune_dict(v) for k, v in val.items() if keep(v)}
    elif isinstance(val, list):
        return [_prune_dict(v) for i, v in enumerate(val) if keep(v)]
    else:
        return tuple(_prune_dict(v) for i, v in enumerate(val) if keep(v))


import copy
from typing import Any, Dict, List, Set, Tuple, Union

__all__ = ["flexible_deepcopy"]


# Internal sentinel: return this to signal “skip me”.
class _SkipType:
    pass


_SKIP = _SkipType()

Container = Union[Dict[Any, Any], List[Any], Tuple[Any, ...], Set[Any]]


def flexible_deepcopy(
    obj: Any,
    on_fail: str = "raise",
    _memo: Optional[Dict[int, Any]] = None,
) -> Any:
    """
    Perform a deepcopy that tolerates un‑copyable elements.

    Parameters
    ----------
    obj : Any
        The object you wish to copy.
    on_fail : {'raise', 'skip', 'shallow'}, default 'raise'
        • 'raise'   – re‑raise copy error (standard behaviour).
        • 'skip'    – drop the offending element from the result.
        • 'shallow' – insert the original element unchanged.
    _memo : dict or None (internal)
        Memoisation dict to preserve identity & avoid infinite recursion.

    Returns
    -------
    Any
        A deep‑copied version of *obj*, modified per *on_fail* strategy.

    Raises
    ------
    ValueError
        If *on_fail* is not one of the accepted values.
    Exception
        Re‑raises whatever copy error occurred when *on_fail* == 'raise'.
    """
    if _memo is None:
        _memo = {}

    obj_id = id(obj)
    if obj_id in _memo:  # Handle circular references.
        return _memo[obj_id]

    def _attempt(value: Any) -> Union[Any, _SkipType]:
        """Try to deepcopy *value*; fall back per on_fail."""
        try:
            return flexible_deepcopy(value, on_fail, _memo)
        except Exception:
            if on_fail == "raise":
                raise
            if on_fail == "shallow":
                return value
            if on_fail == "skip":
                return _SKIP
            raise ValueError(f"Invalid on_fail option: {on_fail!r}")

    # --- Handle built‑in containers explicitly ---------------------------
    if isinstance(obj, dict):
        result: Dict[Any, Any] = {}
        _memo[obj_id] = result  # Early memoisation for cycles
        for k, v in obj.items():
            nk = _attempt(k)
            nv = _attempt(v)
            if _SKIP in (nk, nv):  # Skip entry if key or value failed
                continue
            result[nk] = nv
        return result

    if isinstance(obj, list):
        result: List[Any] = []
        _memo[obj_id] = result
        for item in obj:
            nitem = _attempt(item)
            if nitem is not _SKIP:
                result.append(nitem)
        return result

    if isinstance(obj, tuple):
        items = []
        _memo[obj_id] = None  # Placeholder for circular refs
        for item in obj:
            nitem = _attempt(item)
            if nitem is not _SKIP:
                items.append(nitem)
        result = tuple(items)
        _memo[obj_id] = result
        return result

    if isinstance(obj, set):
        result: Set[Any] = set()
        _memo[obj_id] = result
        for item in obj:
            nitem = _attempt(item)
            if nitem is not _SKIP:
                result.add(nitem)
        return result

    # --- Non‑container: fall back to standard deepcopy -------------------
    try:
        result = copy.deepcopy(obj, _memo)
        _memo[obj_id] = result
        return result
    except Exception:
        if on_fail == "raise":
            raise
        if on_fail == "shallow":
            _memo[obj_id] = obj
            return obj
        if on_fail == "skip":
            return _SKIP
        raise ValueError(f"Invalid on_fail option: {on_fail!r}")

```

`/Users/yushaarif/Unify/unify/unify/utils/_requests.py`:

```py
import json
import logging
import os

import requests

_logger = logging.getLogger("unify_requests")
_log_enabled = os.getenv("UNIFY_REQUESTS_DEBUG", "false").lower() in ("true", "1")
_logger.setLevel(logging.DEBUG if _log_enabled else logging.WARNING)


class ResponseDecodeError(Exception):
    def __init__(self, response: requests.Response):
        self.response = response
        super().__init__(f"Request failed to parse response: {response.text}")


def _log(type: str, url: str, mask_key: bool = True, /, **kwargs):
    if not _log_enabled:
        return
    _kwargs_str = ""
    if mask_key and "headers" in kwargs:
        key = kwargs["headers"]["Authorization"]
        kwargs["headers"]["Authorization"] = "***"

    for k, v in kwargs.items():
        if isinstance(v, dict):
            _kwargs_str += f"{k:}:{json.dumps(v, indent=2)},\n"
        else:
            _kwargs_str += f"{k}:{v},\n"

    if mask_key and "headers" in kwargs:
        kwargs["headers"]["Authorization"] = key

    log_msg = f"""
====== {type} =======
url:{url}
{_kwargs_str}
"""
    _logger.debug(log_msg)


def request(method, url, **kwargs):
    _log(f"request:{method}", url, True, **kwargs)
    res = requests.request(method, url, **kwargs)
    try:
        _log(f"request:{method} response:{res.status_code}", url, response=res.json())
    except requests.exceptions.JSONDecodeError as e:
        raise ResponseDecodeError(res)
    return res


def get(url, params=None, **kwargs):
    _log("GET", url, True, params=params, **kwargs)
    res = requests.get(url, params=params, **kwargs)
    try:
        _log(f"GET response:{res.status_code}", url, response=res.json())
    except requests.exceptions.JSONDecodeError as e:
        raise ResponseDecodeError(res)
    return res


def options(url, **kwargs):
    _log("OPTIONS", url, True, **kwargs)
    res = requests.options(url, **kwargs)
    try:
        _log(f"OPTIONS response:{res.status_code}", url, response=res.json())
    except requests.exceptions.JSONDecodeError as e:
        raise ResponseDecodeError(res)
    return res


def head(url, **kwargs):
    _log("HEAD", url, True, **kwargs)
    res = requests.head(url, **kwargs)
    try:
        _log(f"HEAD response:{res.status_code}", url, response=res.json())
    except requests.exceptions.JSONDecodeError as e:
        raise ResponseDecodeError(res)
    return res


def post(url, data=None, json=None, **kwargs):
    _log("POST", url, True, data=data, json=json, **kwargs)
    res = requests.post(url, data=data, json=json, **kwargs)
    try:
        _log(f"POST response:{res.status_code}", url, response=res.json())
    except requests.exceptions.JSONDecodeError as e:
        raise ResponseDecodeError(res)
    return res


def put(url, data=None, **kwargs):
    _log("PUT", url, True, data=data, **kwargs)
    res = requests.put(url, data=data, **kwargs)
    try:
        _log(f"PUT response:{res.status_code}", url, response=res.json())
    except requests.exceptions.JSONDecodeError as e:
        raise ResponseDecodeError(res)
    return res


def patch(url, data=None, **kwargs):
    _log("PATCH", url, True, data=data, **kwargs)
    res = requests.patch(url, data=data, **kwargs)
    try:
        _log(f"PATCH response:{res.status_code}", url, response=res.json())
    except requests.exceptions.JSONDecodeError as e:
        raise ResponseDecodeError(res)
    return res


def delete(url, **kwargs):
    _log("DELETE", url, True, **kwargs)
    res = requests.delete(url, **kwargs)
    try:
        _log(f"DELETE response:{res.status_code}", url, response=res.json())
    except requests.exceptions.JSONDecodeError as e:
        raise ResponseDecodeError(res)
    return res

```

`/Users/yushaarif/Unify/unify/unify/logging/utils/datasets.py`:

```py
from typing import Any, Dict, List, Optional

from ...utils.helpers import _get_and_maybe_create_project, _validate_api_key
from ..logs import Log
from .contexts import *
from .logs import *

# Datasets #
# ---------#


def list_datasets(
    *,
    project: Optional[str] = None,
    prefix: str = "",
    api_key: Optional[str] = None,
) -> Dict[str, str]:
    """
    List all datasets associated with a project and context.

    Args:
        project: Name of the project the datasets belong to.

        prefix: Prefix of the datasets to get.

        api_key: If specified, unify API key to be used. Defaults to the value in the
        `UNIFY_KEY` environment variable.

    Returns:
        A list of datasets.
    """
    api_key = _validate_api_key(api_key)
    contexts = get_contexts(
        prefix=f"Datasets/{prefix}",
        project=project,
        api_key=api_key,
    )
    return {
        "/".join(name.split("/")[1:]): description
        for name, description in contexts.items()
    }


def upload_dataset(
    name: str,
    data: List[Any],
    *,
    overwrite: bool = False,
    allow_duplicates: bool = False,
    project: Optional[str] = None,
    api_key: Optional[str] = None,
) -> List[int]:
    """
    Upload a dataset to the server.

    Args:
        name: Name of the dataset.

        data: Contents of the dataset.

        overwrite: Whether to overwrite the dataset if it already exists.

        allow_duplicates: Whether to allow duplicates in the dataset.

        project: Name of the project the dataset belongs to.

        api_key: If specified, unify API key to be used. Defaults to the value in the
        `UNIFY_KEY` environment variable.
    Returns:
        A list all log ids in the dataset.
    """
    api_key = _validate_api_key(api_key)
    project = _get_and_maybe_create_project(project, api_key=api_key)
    context = f"Datasets/{name}"
    log_instances = [isinstance(item, unify.Log) for item in data]
    are_logs = False
    if not allow_duplicates and not overwrite:
        # ToDo: remove this verbose logic once ignore_duplicates is implemented
        if name in unify.list_datasets():
            upstream_dataset = unify.Dataset(
                unify.download_dataset(name, project=project, api_key=api_key),
            )
        else:
            upstream_dataset = unify.Dataset([])
    if any(log_instances):
        assert all(log_instances), "If any items are logs, all items must be logs"
        are_logs = True
        # ToDo: remove this verbose logic once ignore_duplicates is implemented
        if not allow_duplicates and not overwrite:
            data = [l for l in data if l not in upstream_dataset]
    elif not all(isinstance(item, dict) for item in data):
        # ToDo: remove this verbose logic once ignore_duplicates is implemented
        if not allow_duplicates and not overwrite:
            data = [item for item in data if item not in upstream_dataset]
        data = [{"data": item} for item in data]
    if name in unify.list_datasets():
        upstream_ids = get_logs(
            project=project,
            context=context,
            return_ids_only=True,
        )
    else:
        upstream_ids = []
    if not are_logs:
        return upstream_ids + create_logs(
            project=project,
            context=context,
            entries=data,
            mutable=True,
            batched=True,
            # ToDo: uncomment once ignore_duplicates is implemented
            # ignore_duplicates=not allow_duplicates,
        )
    local_ids = [l.id for l in data]
    matching_ids = [id for id in upstream_ids if id in local_ids]
    matching_data = [l.entries for l in data if l.id in matching_ids]
    assert len(matching_data) == len(
        matching_ids,
    ), "matching data and ids must be the same length"
    if matching_data:
        update_logs(
            logs=matching_ids,
            api_key=api_key,
            entries=matching_data,
            overwrite=True,
        )
    if overwrite:
        upstream_only_ids = [id for id in upstream_ids if id not in local_ids]
        if upstream_only_ids:
            delete_logs(
                logs=upstream_only_ids,
                context=context,
                project=project,
                api_key=api_key,
            )
            upstream_ids = [id for id in upstream_ids if id not in upstream_only_ids]
    ids_not_in_dataset = [
        id for id in local_ids if id not in matching_ids and id is not None
    ]
    if ids_not_in_dataset:
        if context not in unify.get_contexts():
            unify.create_context(
                context,
                project=project,
                api_key=api_key,
            )
        unify.add_logs_to_context(
            log_ids=ids_not_in_dataset,
            context=context,
            project=project,
            api_key=api_key,
        )
    local_only_data = [l.entries for l in data if l.id is None]
    if local_only_data:
        return upstream_ids + create_logs(
            project=project,
            context=context,
            entries=local_only_data,
            mutable=True,
            batched=True,
        )
    return upstream_ids + ids_not_in_dataset


def download_dataset(
    name: str,
    *,
    project: Optional[str] = None,
    api_key: Optional[str] = None,
) -> List[Log]:
    """
    Download a dataset from the server.

    Args:
        name: Name of the dataset.

        project: Name of the project the dataset belongs to.

        api_key: If specified, unify API key to be used. Defaults to the value in the
        `UNIFY_KEY` environment variable.
    """
    api_key = _validate_api_key(api_key)
    project = _get_and_maybe_create_project(project, api_key=api_key)
    logs = get_logs(
        project=project,
        context=f"Datasets/{name}",
    )
    return list(reversed(logs))


def delete_dataset(
    name: str,
    *,
    project: Optional[str] = None,
    api_key: Optional[str] = None,
) -> None:
    """
    Delete a dataset from the server.

    Args:
        name: Name of the dataset.

        project: Name of the project the dataset belongs to.

        api_key: If specified, unify API key to be used. Defaults to the value in the
        `UNIFY_KEY` environment variable.
    """
    api_key = _validate_api_key(api_key)
    project = _get_and_maybe_create_project(project, api_key=api_key)
    delete_context(f"Datasets/{name}", project=project, api_key=api_key)


def add_dataset_entries(
    name: str,
    data: List[Any],
    *,
    project: Optional[str] = None,
    api_key: Optional[str] = None,
) -> List[int]:
    """
    Adds entries to an existing dataset in the server.

    Args:
        name: Name of the dataset.

        contents: Contents to add to the dataset.

        project: Name of the project the dataset belongs to.

        api_key: If specified, unify API key to be used. Defaults to the value in the
        `UNIFY_KEY` environment variable.
    Returns:
        A list of the newly added dataset logs.
    """
    api_key = _validate_api_key(api_key)
    project = _get_and_maybe_create_project(
        project,
        api_key=api_key,
        create_if_missing=False,
    )
    if not all(isinstance(item, dict) for item in data):
        data = [{"data": item} for item in data]
    logs = create_logs(
        project=project,
        context=f"Datasets/{name}",
        entries=data,
        mutable=True,
        batched=True,
    )
    return logs

```

`/Users/yushaarif/Unify/unify/unify/logging/utils/__init__.py`:

```py
import requests


class RequestError(Exception):
    def __init__(self, response: requests.Response):
        req = response.request
        message = (
            f"{req.method} {req.url} failed with status code {response.status_code}. "
            f"Request body: {req.body}, Response: {response.text}"
        )
        super().__init__(message)
        self.response = response


def _check_response(response: requests.Response):
    if not response.ok:
        raise RequestError(response)

```

`/Users/yushaarif/Unify/unify/unify/logging/utils/async_logger.py`:

```py
import asyncio
import logging
import os
import threading
from concurrent.futures import TimeoutError
from typing import List

import aiohttp

# Configure logging based on environment variable
ASYNC_LOGGER_DEBUG = os.getenv("UNIFY_ASYNC_LOGGER_DEBUG", "false").lower() in (
    "true",
    "1",
)
logger = logging.getLogger("async_logger")
logger.setLevel(logging.DEBUG if ASYNC_LOGGER_DEBUG else logging.WARNING)


class AsyncLoggerManager:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str = os.getenv("UNIFY_KEY"),
        num_consumers: int = 256,
    ):

        self.loop = asyncio.new_event_loop()
        self.queue = None
        self.consumers: List[asyncio.Task] = []
        self.num_consumers = num_consumers
        self.start_flag = threading.Event()
        self.shutting_down = False

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "accept": "application/json",
        }
        url = base_url + "/"
        connector = aiohttp.TCPConnector(limit=num_consumers // 2, loop=self.loop)
        self.session = aiohttp.ClientSession(
            url,
            headers=headers,
            loop=self.loop,
            connector=connector,
        )

        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        self.start_flag.wait()
        self.callbacks = []

    def register_callback(self, fn):
        self.callbacks.append(fn)

    def clear_callbacks(self):
        self.callbacks = []

    def _notify_callbacks(self):
        for fn in self.callbacks:
            fn()

    async def _join(self):
        await self.queue.join()

    def join(self):
        try:
            future = asyncio.run_coroutine_threadsafe(self._join(), self.loop)
            while True:
                try:
                    future.result(timeout=0.5)
                    break
                except (asyncio.TimeoutError, TimeoutError):
                    continue
        except Exception as e:
            logger.error(f"Error in join: {e}")
            raise e

    async def _main_loop(self):
        self.start_flag.set()
        await asyncio.gather(*self.consumers, return_exceptions=True)

    def _run_loop(self):
        asyncio.set_event_loop(self.loop)
        self.queue = asyncio.Queue()

        for _ in range(self.num_consumers):
            self.consumers.append(self._log_consumer())

        try:
            self.loop.run_until_complete(self._main_loop())
        except Exception as e:
            logger.error(f"Event loop error: {e}")
            raise e
        finally:
            self.loop.close()

    async def _consume_create(self, body, future, idx):
        async with self.session.post("logs", json=body) as res:
            if res.status != 200:
                txt = await res.text()
                logger.error(f"Error in consume_create {idx}: {txt}")
                return
            res_json = await res.json()
            logger.debug(f"Created {idx} with response {res.status}: {res_json}")
            future.set_result(res_json[0])

    async def _consume_update(self, body, future, idx):
        if not future.done():
            await future
        body["logs"] = [future.result()]
        async with self.session.put("logs", json=body) as res:
            if res.status != 200:
                txt = await res.text()
                logger.error(f"Error in consume_update {idx}: {txt}")
                return
            res_json = await res.json()
            logger.debug(f"Updated {idx} with response {res.status}: {res_json}")

    async def _log_consumer(self):
        while True:
            try:
                event = await self.queue.get()
                idx = self.queue.qsize() + 1
                logger.debug(f"Processing event {event['type']}: {idx}")
                if event["type"] == "create":
                    await self._consume_create(event["_data"], event["future"], idx)
                elif event["type"] == "update":
                    await self._consume_update(event["_data"], event["future"], idx)
                else:
                    raise Exception(f"Unknown event type: {event['type']}")
            except Exception as e:
                event["future"].set_exception(e)
                logger.error(f"Error in consumer: {e}")
                raise e
            finally:
                self.queue.task_done()
                self._notify_callbacks()

    def log_create(
        self,
        project: str,
        context: str,
        params: dict,
        entries: dict,
    ) -> asyncio.Future:
        fut = self.loop.create_future()
        event = {
            "_data": {
                "project": project,
                "context": context,
                "params": params,
                "entries": entries,
            },
            "type": "create",
            "future": fut,
        }
        self.loop.call_soon_threadsafe(self.queue.put_nowait, event)
        return fut

    def log_update(
        self,
        project: str,
        context: str,
        future: asyncio.Future,
        mode: str,
        overwrite: bool,
        data: dict,
    ) -> None:
        event = {
            "_data": {
                mode: data,
                "project": project,
                "context": context,
                "overwrite": overwrite,
            },
            "type": "update",
            "future": future,
        }
        self.loop.call_soon_threadsafe(self.queue.put_nowait, event)

    def stop_sync(self, immediate=False):
        if self.shutting_down:
            return

        self.shutting_down = True
        if immediate:
            logger.debug("Stopping async logger immediately")
            self.loop.stop()
        else:
            self.join()

```

`/Users/yushaarif/Unify/unify/unify/logging/utils/logs.py`:

```py
from __future__ import annotations

import atexit
import inspect
import json
import logging
import queue
import threading

logger = logging.getLogger(__name__)
import os
import sys
from contextvars import ContextVar
from typing import Any, Callable, Dict, List, Optional, Union

import unify
from tqdm import tqdm
from unify import BASE_URL
from unify.utils import _requests
from unify.utils.helpers import flexible_deepcopy

from ...utils._caching import (_get_cache, _get_caching, _get_caching_fname,
                               _write_to_cache)
from ...utils.helpers import (_check_response, _get_and_maybe_create_project,
                              _validate_api_key)
from .async_logger import AsyncLoggerManager

# logging configuration
USR_LOGGING = True
ASYNC_LOGGING = False  # Flag to enable/disable async logging
ASYNC_BATCH_SIZE = 100  # Default batch size for async logging
ASYNC_FLUSH_INTERVAL = 5.0  # Default flush interval in secondss
ASYNC_MAX_QUEUE_SIZE = 10000  # Default maximum queue size

# Async logger instance
_async_logger: Optional[AsyncLoggerManager] = None
_trace_logger: Optional[_AsyncTraceLogger] = None

# log
ACTIVE_LOG = ContextVar("active_log", default=[])
LOGGED = ContextVar("logged", default={})

# context
CONTEXT_READ = ContextVar("context_read", default="")
CONTEXT_WRITE = ContextVar("context_write", default="")
CONTEXT_MODE = ContextVar("context_mode", default="both")

# context function
MODE = None
MODE_TOKEN = None
CONTEXT_READ_TOKEN = None
CONTEXT_WRITE_TOKEN = None

# column context
COLUMN_CONTEXT_READ = ContextVar("column_context_read", default="")
COLUMN_CONTEXT_WRITE = ContextVar("column_context_write", default="")
COLUMN_CONTEXT_MODE = ContextVar("column_context_mode", default="both")

# entries
ACTIVE_ENTRIES_WRITE = ContextVar(
    "active_entries_write",
    default={},
)
ACTIVE_ENTRIES_READ = ContextVar(
    "active_entries_read",
    default={},
)
ACTIVE_ENTRIES_MODE = ContextVar("active_entries_mode", default="both")
ENTRIES_NEST_LEVEL = ContextVar("entries_nest_level", default=0)

# params
ACTIVE_PARAMS_WRITE = ContextVar(
    "active_params_write",
    default={},
)
ACTIVE_PARAMS_READ = ContextVar(
    "active_params_read",
    default={},
)
ACTIVE_PARAMS_MODE = ContextVar("active_params_mode", default="both")
PARAMS_NEST_LEVEL = ContextVar("params_nest_level", default=0)

# span
GLOBAL_SPAN = ContextVar("global_span", default={})
SPAN = ContextVar("span", default={})
RUNNING_TIME = ContextVar("running_time", default=0.0)

# chunking
CHUNK_LIMIT = 5000000


class _AsyncTraceLogger(threading.Thread):
    class _StopEvent:
        pass

    def __init__(self):
        super().__init__(name="TraceLoggerThread", daemon=True)
        self.queue = queue.Queue()
        self.start()
        atexit.register(self.stop)

    def run(self):
        while True:
            item = self.queue.get()
            if isinstance(item, self._StopEvent):
                break

            log_id, trace = item
            unify.add_log_entries(logs=log_id, trace=trace, overwrite=True)

    def update_trace(self, log_id, trace):
        self.queue.put_nowait((log_id, trace))

    def stop(self):
        """Stop normally, waiting for queue to be processed"""
        self.queue.put_nowait(self._StopEvent())
        while self.is_alive():
            try:
                self.join(timeout=0.1)
            except KeyboardInterrupt:
                self.stop_immediate()
                raise

    def stop_immediate(self):
        """Stop immediately, ignoring remaining queue items"""
        self.queue = queue.Queue()
        self.queue.put_nowait(self._StopEvent())
        self.join()


def _removes_unique_trace_values(kw: Dict[str, Any]) -> Dict[str, Any]:
    del kw["id"]
    del kw["exec_time"]
    if "parent_span_id" in kw:
        del kw["parent_span_id"]
    if "child_spans" in kw:
        kw["child_spans"] = [
            _removes_unique_trace_values(cs) for cs in kw["child_spans"]
        ]
    return kw


def initialize_async_logger(
    api_key: Optional[str] = None,
) -> None:
    """
    Initialize the async logger with the specified configuration.

    Args:
        batch_size: Number of logs to batch together before sending
        flush_interval: How often to flush logs in seconds
        max_queue_size: Maximum size of the log queue
        api_key: API key for authentication
    """
    global _async_logger, ASYNC_LOGGING

    if _async_logger is not None:
        return
    api_key = _validate_api_key(api_key)
    _async_logger = AsyncLoggerManager(
        base_url=BASE_URL,
        api_key=api_key,
    )
    ASYNC_LOGGING = True

    # Register shutdown handler
    atexit.register(shutdown_async_logger)


def shutdown_async_logger(immediate=False) -> None:
    """
    Gracefully shutdown the async logger, ensuring all pending logs are flushed.
    """
    global _async_logger, ASYNC_LOGGING

    if _async_logger is not None:
        _async_logger.stop_sync(immediate=immediate)
        _async_logger = None
        ASYNC_LOGGING = False


def _initialize_trace_logger():
    global _trace_logger
    if _trace_logger is None:
        _trace_logger = _AsyncTraceLogger()


def _get_trace_logger():
    return _trace_logger


def _handle_cache(fn: Callable) -> Callable:
    def wrapped(*args, **kwargs):
        if not _get_caching():
            return fn(*args, **kwargs)
        kw_for_key = flexible_deepcopy(kwargs)
        if fn.__name__ == "add_log_entries" and "trace" in kwargs:
            kw_for_key["trace"] = _removes_unique_trace_values(kw_for_key["trace"])
        combined_kw = {**{f"arg{i}": a for i, a in enumerate(args)}, **kw_for_key}
        ret = _get_cache(
            fn_name=fn.__name__,
            kw=combined_kw,
            filename=_get_caching_fname(),
        )
        if ret is not None:
            return ret
        ret = fn(*args, **kwargs)
        _write_to_cache(
            fn_name=fn.__name__,
            kw=combined_kw,
            response=ret,
            filename=_get_caching_fname(),
        )
        return ret

    return wrapped


def _handle_special_types(
    kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    new_kwargs = dict()
    for k, v in kwargs.items():
        if isinstance(v, unify.Dataset):
            v.upload()
            new_kwargs[k] = v.name
        elif callable(v):
            new_kwargs[k] = inspect.getsource(v)
        else:
            new_kwargs[k] = v
    return new_kwargs


def _to_log_ids(
    logs: Optional[Union[int, unify.Log, List[Union[int, unify.Log]]]] = None,
):
    def resolve_log_id(log):
        if isinstance(log, unify.Log):
            if log.id is None and hasattr(log, "_future"):
                try:
                    # Wait (with timeout) for the future to resolve
                    log._id = log._future.result(timeout=5)
                except Exception as e:
                    raise Exception(f"Failed to resolve log id: {e}")
            return log.id
        return log

    if logs is None:
        current_active_logs = ACTIVE_LOG.get()
        if not current_active_logs:
            raise Exception(
                "If logs is unspecified, then current_global_active_log must be.",
            )
        return [resolve_log_id(current_active_logs[-1])]
    elif isinstance(logs, int):
        return [logs]
    elif isinstance(logs, unify.Log):
        return [resolve_log_id(logs)]
    elif isinstance(logs, list):
        if not logs:
            return logs
        elif isinstance(logs[0], int):
            return logs
        elif isinstance(logs[0], unify.Log):
            return [resolve_log_id(lg) for lg in logs]
        else:
            raise Exception(
                f"list must contain int or unify.Log types, but found first entry {logs[0]} of type {type(logs[0])}",
            )
    raise Exception(
        f"logs argument must be of type int, unify.Log, or list, but found {logs} of type {type(logs)}",
    )


def _apply_col_context(**data):
    if COLUMN_CONTEXT_MODE.get() == "both":
        assert COLUMN_CONTEXT_WRITE.get() == COLUMN_CONTEXT_READ.get()
        col_context = COLUMN_CONTEXT_WRITE.get()
    elif COLUMN_CONTEXT_MODE.get() == "write":
        col_context = COLUMN_CONTEXT_WRITE.get()
    elif COLUMN_CONTEXT_MODE.get() == "read":
        col_context = COLUMN_CONTEXT_READ.get()
    return {os.path.join(col_context, k): v for k, v in data.items()}


def _handle_context(context: Optional[Union[str, Dict[str, str]]] = None):
    if context is None:
        return {"name": CONTEXT_WRITE.get()}
    if isinstance(context, str):
        return {"name": context}
    else:
        return context


def _handle_mutability(
    mutable: Optional[Union[bool, Dict[str, bool]]],
    data: Optional[Union[List[Dict[str, Any]], Dict[str, Any]]] = None,
):
    if mutable is None or data is None:
        return data

    if isinstance(data, list):
        single_item = False
        new_data = flexible_deepcopy(data, on_fail="shallow")
    else:
        single_item = True
        new_data = [flexible_deepcopy(data, on_fail="shallow")]
    if isinstance(mutable, dict):
        for field, mut in mutable.items():
            for item in new_data:
                if field in item:
                    item.setdefault("explicit_types", {})[field] = {"mutable": mut}
    elif isinstance(mutable, bool):
        for item in new_data:
            for k in list(item.keys()):
                if k != "explicit_types":
                    item.setdefault("explicit_types", {})[k] = {"mutable": mutable}
    if single_item:
        return new_data[0]
    return new_data


def _json_chunker(big_dict, chunk_size=1024 * 1024):
    json_string = json.dumps(big_dict)
    total_bytes = len(json_string)
    pbar = tqdm(total=total_bytes, unit="B", unit_scale=True, desc="Uploading JSON")
    start = 0
    while start < total_bytes:
        end = min(start + chunk_size, total_bytes)
        chunk = json_string[start:end]
        yield chunk
        pbar.update(len(chunk))
        start = end
    pbar.close()


def log(
    fn: Optional[Callable] = None,
    *,
    project: Optional[str] = None,
    context: Optional[str] = None,
    params: Dict[str, Any] = None,
    new: bool = False,
    overwrite: bool = False,
    mutable: Optional[Union[bool, Dict[str, bool]]] = True,
    api_key: Optional[str] = None,
    **entries,
) -> Union[unify.Log, Callable]:
    """
    Can be used either as a regular function to create logs or as a decorator to log function inputs, intermediates and outputs.

    When used as a regular function:
    Creates one or more logs associated to a project. unify.Logs are LLM-call-level data
    that might depend on other variables.

    When used as a decorator:
    Logs function inputs and intermediate values.

    Args:
        fn: When used as a decorator, this is the function to be wrapped.
        project: Name of the project the stored logs will be associated to.

        context: Context for the logs.

        params: Dictionary containing one or more key:value pairs that will be
        logged into the platform as params.

        new: Whether to create a new log if there is a currently active global log.
        Defaults to False, in which case log will add to the existing log.

        overwrite: If adding to an existing log, dictates whether or not to overwrite
        fields with the same name.

        mutable: Either a boolean to apply uniform mutability for all fields, or a dictionary mapping field names to booleans for per-field control. Defaults to True.

        api_key: If specified, unify API key to be used. Defaults to the value in the
        `UNIFY_KEY` environment variable.

        entries: Dictionary containing one or more key:value pairs that will be logged
        into the platform as entries.

    Returns:
        When used as a regular function: The unique id of newly created log.
        When used as a decorator: The wrapped function.
    """
    # If used as a decorator
    if fn is not None and callable(fn):
        from unify.logging.logs import log_decorator

        if inspect.iscoroutinefunction(fn):

            async def async_wrapper(*args, **kwargs):
                transformed = log_decorator(fn)
                return await transformed(*args, **kwargs)

            return async_wrapper
        transformed = log_decorator(fn)
        return transformed

    # Regular log function logic
    global ASYNC_LOGGING
    api_key = _validate_api_key(api_key)
    context = _handle_context(context)
    if not new and ACTIVE_LOG.get():
        _add_to_log(
            context=context,
            mode="entries",
            overwrite=overwrite,
            mutable=mutable,
            api_key=api_key,
            **entries,
        )
        _add_to_log(
            context=context,
            mode="params",
            overwrite=overwrite,
            mutable=mutable,
            api_key=api_key,
            **(params if params is not None else {}),
        )
        log = ACTIVE_LOG.get()[-1]
        if USR_LOGGING:
            logger.info(f"Updated Log({log.id})")
        return log
    # Process parameters and entries
    params = _apply_col_context(**(params if params else {}))
    params = {**params, **ACTIVE_PARAMS_WRITE.get()}
    params = _handle_special_types(params)
    params = _handle_mutability(mutable, params)
    entries = _apply_col_context(**entries)
    entries = {**entries, **ACTIVE_ENTRIES_WRITE.get()}
    entries = _handle_special_types(entries)
    entries = _handle_mutability(mutable, entries)
    project = _get_and_maybe_create_project(project, api_key=api_key)
    if ASYNC_LOGGING and _async_logger is not None:
        # Use async logging: enqueue a create event and capture the Future.
        log_future = _async_logger.log_create(
            project=project,
            context=context,
            params=params,
            entries=entries,
        )
        created_log = unify.Log(
            id=None,  # Placeholder; will be updated when the Future resolves.
            _future=log_future,
            api_key=api_key,
            **entries,
            params=params,
            context=context,
        )
    else:
        # Use synchronous logging
        created_log = _sync_log(
            project=project,
            context=context,
            params=params,
            entries=entries,
            api_key=api_key,
        )

    if PARAMS_NEST_LEVEL.get() > 0 or ENTRIES_NEST_LEVEL.get() > 0:
        LOGGED.set(
            {
                **LOGGED.get(),
                created_log.id: list(params.keys()) + list(entries.keys()),
            },
        )
    if USR_LOGGING:
        logger.info(f"Created Log({created_log.id})")
    return created_log


def _sync_log(
    project: str,
    context: Optional[str],
    params: Dict[str, Any],
    entries: Dict[str, Any],
    api_key: str,
) -> unify.Log:
    """
    Synchronously create a log entry using direct HTTP request.

    This is a helper function used when async logging is disabled or unavailable.
    """
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    body = {
        "project": project,
        "context": context,
        "params": params,
        "entries": entries,
    }
    response = _requests.post(BASE_URL + "/logs", headers=headers, json=body)
    _check_response(response)
    return unify.Log(
        id=response.json()["log_event_ids"][0],
        api_key=api_key,
        **entries,
        params=params,
        context=context,
    )


def _create_log(dct, params, context, api_key, context_entries=None):
    if context_entries is None:
        context_entries = {}
    return unify.Log(
        id=dct["id"],
        ts=dct["ts"],
        **dct["entries"],
        **dct["derived_entries"],
        **context_entries,
        params={
            param_name: (param_ver, params[param_name][param_ver])
            for param_name, param_ver in dct["params"].items()
        },
        context=context,
        api_key=api_key,
    )


def _create_log_groups_nested(
    params,
    context,
    api_key,
    node,
    context_entries,
    prev_key=None,
):
    if isinstance(node, dict) and "group" not in node:
        ret = unify.LogGroup(list(node.keys())[0])
        ret.value = _create_log_groups_nested(
            params,
            context,
            api_key,
            node[ret.field],
            context_entries,
            ret.field,
        )
        return ret
    else:
        if isinstance(node["group"][0]["value"], list):
            ret = {}
            for n in node["group"]:
                context_entries[prev_key] = n["key"]
                ret[n["key"]] = [
                    _create_log(
                        item,
                        item["params"],
                        context,
                        api_key,
                        context_entries,
                    )
                    for item in n["value"]
                ]
            return ret
        else:
            ret = {}
            for n in node["group"]:
                context_entries[prev_key] = n["key"]
                ret[n["key"]] = _create_log_groups_nested(
                    params,
                    context,
                    api_key,
                    n["value"],
                    context_entries,
                    n["key"],
                )
            return ret


def _create_log_groups_not_nested(logs, groups, params, context, api_key):
    logs_mapping = {}
    for dct in logs:
        logs_mapping[dct["id"]] = _create_log(dct, params, context, api_key)

    ret = []
    for group_key, group_value in groups.items():
        if isinstance(group_value, dict):
            val = {}
            for k, v in group_value.items():
                if isinstance(v, list):
                    val[k] = [logs_mapping[log_id] for log_id in v]
            ret.append(unify.LogGroup(group_key, val))
    return ret


def create_logs(
    *,
    project: Optional[str] = None,
    context: Optional[str] = None,
    params: Optional[Union[List[Dict[str, Any]], Dict[str, Any]]] = None,
    entries: Optional[Union[List[Dict[str, Any]], Dict[str, Any]]] = None,
    mutable: Optional[Union[bool, Dict[str, bool]]] = True,
    batched: Optional[bool] = None,
    api_key: Optional[str] = None,
) -> List[int]:
    """
    Creates one or more logs associated to a project.

    Args:
        project: Name of the project the stored logs will be associated to.

        context: Context for the logs.

        entries: List of dictionaries with the entries to be logged.

        params: List of dictionaries with the params to be logged.

        mutable: Either a boolean to apply uniform mutability for all fields, or a dictionary mapping field names to booleans for per-field control. Defaults to True.

        api_key: If specified, unify API key to be used. Defaults to the value in the
        `UNIFY_KEY` environment variable.

    Returns:
        A list of the created logs.
    """
    api_key = _validate_api_key(api_key)
    project = _get_and_maybe_create_project(project, api_key=api_key)
    context = _handle_context(context)
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    # ToDo: add support for all of the context variables, as is done for `unify.log` above
    params = _handle_mutability(mutable, params)
    entries = _handle_mutability(mutable, entries)
    # ToDo remove the params/entries logic above once this [https://app.clickup.com/t/86c25g263] is done
    params = [{}] * len(entries) if params in [None, []] else params
    entries = [{}] * len(params) if entries in [None, []] else entries
    # end ToDo
    body = {
        "project": project,
        "context": context,
        "params": params,
        "entries": entries,
    }
    body_size = sys.getsizeof(json.dumps(body))
    if batched is None:
        batched = body_size < CHUNK_LIMIT
    if batched:
        if body_size < CHUNK_LIMIT:
            response = _requests.post(BASE_URL + "/logs", headers=headers, json=body)
        else:
            response = _requests.post(
                BASE_URL + "/logs",
                headers=headers,
                data=_json_chunker(body),
            )
        _check_response(response)
        return [
            unify.Log(
                project=project,
                context=context,
                **{k: v for k, v in e.items() if k != "explicit_types"},
                **p,
                id=i,
            )
            for e, p, i in zip(entries, params, response.json()["log_event_ids"])
        ]

    pbar = tqdm(total=len(params), unit="logs", desc="Creating Logs")
    try:
        unify.initialize_async_logger()
        _async_logger.register_callback(lambda: pbar.update(1))
        ret = []

        for p, e in zip(params, entries):
            ret.append(
                log(
                    project=project,
                    context=context,
                    params=p,
                    new=True,
                    mutable=mutable,
                    api_key=api_key,
                    **e,
                ),
            )
    finally:
        unify.shutdown_async_logger()
        pbar.close()
    return ret


def _add_to_log(
    *,
    context: Optional[str] = None,
    logs: Optional[Union[int, unify.Log, List[Union[int, unify.Log]]]] = None,
    mode: str = None,
    overwrite: bool = False,
    mutable: Optional[Union[bool, Dict[str, bool]]] = True,
    api_key: Optional[str] = None,
    **data,
) -> Dict[str, str]:
    assert mode in ("params", "entries"), "mode must be one of 'params', 'entries'"
    data = _apply_col_context(**data)
    nest_level = {"params": PARAMS_NEST_LEVEL, "entries": ENTRIES_NEST_LEVEL}[mode]
    active = {"params": ACTIVE_PARAMS_WRITE, "entries": ACTIVE_ENTRIES_WRITE}[mode]
    api_key = _validate_api_key(api_key)
    context = _handle_context(context)
    data = _handle_special_types(data)
    data = _handle_mutability(mutable, data)
    if ASYNC_LOGGING and _async_logger is not None:
        # For simplicity, assume logs is a single unify.Log.
        if logs is None:
            log_obj = ACTIVE_LOG.get()[-1]
        elif isinstance(logs, unify.Log):
            log_obj = logs
        elif isinstance(logs, list) and logs and isinstance(logs[0], unify.Log):
            log_obj = logs[0]
        else:
            # If not a Log, resolve synchronously.
            log_id = _to_log_ids(logs)[0]
            lf = _async_logger._loop.create_future()
            lf.set_result(log_id)
            log_obj = unify.Log(id=log_id, _future=lf, api_key=api_key)
        # Prepare the future to pass (if the log is still pending, use its _future)
        if hasattr(log_obj, "_future") and log_obj._future is not None:
            lf = log_obj._future
        else:
            lf = _async_logger._loop.create_future()
            lf.set_result(log_obj.id)
        _async_logger.log_update(
            project=_get_and_maybe_create_project(None, api_key=api_key),
            context=context,
            future=lf,
            mode=mode,
            overwrite=overwrite,
            data=data,
        )
        return {"detail": "Update queued asynchronously"}
    else:
        # Fallback to synchronous update if async logging isn’t enabled.
        log_ids = _to_log_ids(logs)
        headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        all_kwargs = []
        if nest_level.get() > 0:
            for log_id in log_ids:
                combined_kwargs = {
                    **data,
                    **{
                        k: v
                        for k, v in active.get().items()
                        if k not in LOGGED.get().get(log_id, {})
                    },
                }
                all_kwargs.append(combined_kwargs)
            assert all(
                kw == all_kwargs[0] for kw in all_kwargs
            ), "All logs must share the same context if they're all being updated at the same time."
            data = all_kwargs[0]
        body = {"logs": log_ids, mode: data, "overwrite": overwrite, "context": context}
        response = _requests.put(BASE_URL + "/logs", headers=headers, json=body)
        _check_response(response)
        if nest_level.get() > 0:
            logged = LOGGED.get()
            new_logged = {}
            for log_id in log_ids:
                if log_id in logged:
                    new_logged[log_id] = logged[log_id] + list(data.keys())
                else:
                    new_logged[log_id] = list(data.keys())
            LOGGED.set({**logged, **new_logged})
        return response.json()


def add_log_params(
    *,
    logs: Optional[Union[int, unify.Log, List[Union[int, unify.Log]]]] = None,
    mutable: Optional[Union[bool, Dict[str, bool]]] = True,
    api_key: Optional[str] = None,
    **params,
) -> Dict[str, str]:
    """
    Add extra params into an existing log.

    Args:
        logs: The log(s) to update with extra params. Looks for the current active log if
        no id is provided.

        mutable: Either a boolean to apply uniform mutability for all parameters, or a dictionary mapping parameter names to booleans for per-field control.
        Defaults to True.
        api_key: If specified, unify API key to be used. Defaults to the value in the
        `UNIFY_KEY` environment variable.

        params: Dictionary containing one or more key:value pairs that will be
        logged into the platform as params.

    Returns:
        A message indicating whether the logs were successfully updated.
    """
    ret = _add_to_log(
        logs=logs,
        mode="params",
        mutable=mutable,
        api_key=api_key,
        **params,
    )
    if USR_LOGGING:
        logger.info(
            f"Added Params {', '.join(list(params.keys()))} "
            f"to [Logs({', '.join([str(i) for i in _to_log_ids(logs)])})]",
        )
    return ret


def add_log_entries(
    *,
    logs: Optional[Union[int, unify.Log, List[Union[int, unify.Log]]]] = None,
    overwrite: bool = False,
    mutable: Optional[Union[bool, Dict[str, bool]]] = True,
    api_key: Optional[str] = None,
    **entries,
) -> Dict[str, str]:
    """
    Add extra entries into an existing log.

    Args:
        logs: The log(s) to update with extra entries. Looks for the current active log if
        no id is provided.

        overwrite: Whether or not to overwrite an entry pre-existing with the same name.

        mutable: Either a boolean to apply uniform mutability for all entries, or a dictionary mapping entry names to booleans for per-field control.
        Defaults to True.

        api_key: If specified, unify API key to be used. Defaults to the value in the
        `UNIFY_KEY` environment variable.

        entries: Dictionary containing one or more key:value pairs that will be logged
        into the platform as entries.

    Returns:
        A message indicating whether the logs were successfully updated.
    """
    ret = _add_to_log(
        logs=logs,
        mode="entries",
        overwrite=overwrite,
        mutable=mutable,
        api_key=api_key,
        **entries,
    )
    if USR_LOGGING:
        logger.info(
            f"Added Entries {', '.join(list(entries.keys()))} "
            f"to Logs({', '.join([str(i) for i in _to_log_ids(logs)])})",
        )
    return ret


def update_logs(
    *,
    logs: Optional[Union[int, unify.Log, List[Union[int, unify.Log]]]] = None,
    context: Optional[Union[str, List[str]]] = None,
    params: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
    entries: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
    overwrite: bool = False,
    api_key: Optional[str] = None,
) -> Dict[str, str]:
    """
    Updates existing logs.
    """
    if not logs and not params and not entries:
        return {"detail": "No logs to update."}
    api_key = _validate_api_key(api_key)
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    body = {
        "logs": _to_log_ids(logs),
        "context": context,
        # ToDo: remove once this [https://app.clickup.com/t/86c25g263] is done
        "params": [{}] * len(entries) if params is None else params,
        "entries": [{}] * len(params) if entries is None else entries,
        # end ToDo
        "overwrite": overwrite,
    }
    response = _requests.put(BASE_URL + "/logs", headers=headers, json=body)
    _check_response(response)
    return response.json()


def delete_logs(
    *,
    logs: Optional[Union[int, unify.Log, List[Union[int, unify.Log]]]] = None,
    project: Optional[str] = None,
    context: Optional[str] = None,
    delete_empty_logs: bool = False,
    source_type: str = "all",
    api_key: Optional[str] = None,
) -> Dict[str, str]:
    """
    Deletes logs from a project.

    Args:
        logs: log(s) to delete from a project.

        project: Name of the project to delete logs from.

        context: Context of the logs to delete. Logs will be removed from that context instead of being entirely deleted,
        unless it is the last context associated with the log.

        delete_empty_logs: Whether to delete logs that become empty after deleting the specified fields.

        source_type: Type of logs to delete. Can be "all", "derived", or "base".

        api_key: If specified, unify API key to be used. Defaults to the value in the
        `UNIFY_KEY` environment variable.

    Returns:
        A message indicating whether the logs were successfully deleted.
    """
    if logs is None:
        logs = get_logs(project=project, context=context, api_key=api_key)
        if not logs:
            return {"message": "No logs to delete"}
    project = _get_and_maybe_create_project(project, api_key=api_key)
    context = context if context else CONTEXT_READ.get()
    log_ids = _to_log_ids(logs)
    api_key = _validate_api_key(api_key)
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    body = {
        "project": project,
        "context": context,
        "ids_and_fields": [(log_ids, None)],
        "source_type": source_type,
    }
    params = {"delete_empty_logs": delete_empty_logs}
    response = _requests.delete(
        BASE_URL + "/logs",
        headers=headers,
        params=params,
        json=body,
    )
    _check_response(response)
    if USR_LOGGING:
        logger.info(f"Deleted Logs({', '.join([str(i) for i in log_ids])})")
    return response.json()


def delete_log_fields(
    *,
    field: str,
    logs: Optional[Union[int, unify.Log, List[Union[int, unify.Log]]]] = None,
    project: Optional[str] = None,
    context: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Dict[str, str]:
    """
    Deletes an entry from a log.

    Args:
        field: Name of the field to delete from the given logs.

        logs: log(s) to delete entries from.

        project: Name of the project to delete logs from.

        context: Context of the logs to delete entries from.

        api_key: If specified, unify API key to be used. Defaults to the value in the
        `UNIFY_KEY` environment variable.

    Returns:
        A message indicating whether the log entries were successfully deleted.
    """
    log_ids = _to_log_ids(logs)
    api_key = _validate_api_key(api_key)
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    project = _get_and_maybe_create_project(project, api_key=api_key)
    context = context if context else CONTEXT_READ.get()
    body = {
        "project": project,
        "context": context,
        "ids_and_fields": [(log_ids, field)],
    }
    response = _requests.delete(
        BASE_URL + f"/logs",
        headers=headers,
        json=body,
    )
    _check_response(response)
    if USR_LOGGING:
        logger.info(
            f"Deleted Field `{field}` from Logs({', '.join([str(i) for i in log_ids])})",
        )
    return response.json()


# noinspection PyShadowingBuiltins
def get_logs(
    *,
    project: Optional[str] = None,
    context: Optional[str] = None,
    column_context: Optional[str] = None,
    filter: Optional[str] = None,
    limit: Optional[int] = None,
    offset: int = 0,
    return_versions: Optional[bool] = None,
    group_threshold: Optional[int] = None,
    value_limit: Optional[int] = None,
    sorting: Optional[Dict[str, Any]] = None,
    group_sorting: Optional[Dict[str, Any]] = None,
    from_ids: Optional[List[int]] = None,
    exclude_ids: Optional[List[int]] = None,
    from_fields: Optional[List[str]] = None,
    exclude_fields: Optional[List[str]] = None,
    group_by: Optional[List[str]] = None,
    group_limit: Optional[int] = None,
    group_offset: Optional[int] = 0,
    group_depth: Optional[int] = None,
    nested_groups: Optional[bool] = True,
    groups_only: Optional[bool] = None,
    return_timestamps: Optional[bool] = None,
    return_ids_only: bool = False,
    api_key: Optional[str] = None,
) -> Union[List[unify.Log], Dict[str, Any]]:
    """
    Returns a list of filtered logs from a project.

    Args:
        project: Name of the project to get logs from.

        context: Context of the logs to get.

        column_context: Column context of the logs to get.

        filter: Boolean string to filter logs, for example:
        "(temperature > 0.5 and (len(system_msg) < 100 or 'no' in usr_response))"

        limit: The maximum number of logs to return. Default is None (unlimited).

        offset: The starting index of the logs to return. Default is 0.

        return_versions: Whether to return all versions of logs.

        group_threshold: Entries that appear in at least this many logs will be grouped together.

        value_limit: Maximum number of characters to return for string values.

        sorting: A dictionary specifying the sorting order for the logs by field names.

        group_sorting: A dictionary specifying the sorting order for the groups relative to each other based on aggregated metrics.

        from_ids: A list of log IDs to include in the results.

        exclude_ids: A list of log IDs to exclude from the results.

        from_fields: A list of field names to include in the results.

        exclude_fields: A list of field names to exclude from the results.

        group_by: A list of field names to group the logs by.

        group_limit: The maximum number of groups to return at each level.

        group_offset: Number of groups to skip at each level.

        group_depth: Maximum depth of nested groups to return.

        nested_groups: Whether to return nested groups.

        groups_only: Whether to return only the groups.

        return_timestamps: Whether to return the timestamps of the logs.

        return_ids_only: Whether to return only the log ids.

        api_key: If specified, unify API key to be used. Defaults to the value in the
        `UNIFY_KEY` environment variable.

    Returns:
        The list of logs for the project, after optionally applying filtering.
    """
    # ToDo: add support for all context handlers
    api_key = _validate_api_key(api_key)
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    project = _get_and_maybe_create_project(project, api_key=api_key)
    context = context if context else CONTEXT_READ.get()
    column_context = column_context if column_context else COLUMN_CONTEXT_READ.get()
    merged_filters = ACTIVE_PARAMS_READ.get() | ACTIVE_ENTRIES_READ.get()
    if merged_filters:
        _filter = " and ".join(f"{k}=={repr(v)}" for k, v in merged_filters.items())
        if filter:
            filter = f"({filter}) and ({_filter})"
        else:
            filter = _filter
    params = {
        "project": project,
        "context": context,
        "filter_expr": filter,
        "limit": limit,
        "offset": offset,
        "return_ids_only": return_ids_only,
        "column_context": column_context,
        "return_versions": return_versions,
        "group_threshold": group_threshold,
        "value_limit": value_limit,
        "sorting": json.dumps(sorting) if sorting is not None else None,
        "group_sorting": (
            json.dumps(group_sorting) if group_sorting is not None else None
        ),
        "from_ids": "&".join(map(str, from_ids)) if from_ids else None,
        "exclude_ids": "&".join(map(str, exclude_ids)) if exclude_ids else None,
        "from_fields": "&".join(from_fields) if from_fields else None,
        "exclude_fields": "&".join(exclude_fields) if exclude_fields else None,
        "group_by": group_by,
        "group_limit": group_limit,
        "group_offset": group_offset,
        "group_depth": group_depth,
        "nested_groups": nested_groups,
        "groups_only": groups_only,
        "return_timestamps": return_timestamps,
    }

    response = _requests.get(BASE_URL + "/logs", headers=headers, params=params)
    _check_response(response)

    if not group_by:
        if return_ids_only:
            return response.json()
        params, logs, _ = response.json().values()
        return [_create_log(dct, params, context, api_key) for dct in logs]

    if nested_groups:
        params, logs, _ = response.json().values()
        return _create_log_groups_nested(params, context, api_key, logs, {})
    else:
        params, groups, logs, _ = response.json().values()
        return _create_log_groups_not_nested(logs, groups, params, context, api_key)


# noinspection PyShadowingBuiltins
def get_log_by_id(
    id: int,
    project: Optional[str] = None,
    *,
    api_key: Optional[str] = None,
) -> unify.Log:
    """
    Returns the log associated with a given id.

    Args:
        id: IDs of the logs to fetch.

        project: Name of the project to get logs from.

        api_key: If specified, unify API key to be used. Defaults to the value in the
        `UNIFY_KEY` environment variable.

    Returns:
        The full set of log data.
    """
    api_key = _validate_api_key(api_key)
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    project = _get_and_maybe_create_project(project, api_key=api_key)
    response = _requests.get(
        BASE_URL + "/logs",
        params={"project": project, "from_ids": [id]},
        headers=headers,
    )
    _check_response(response)
    params, lgs, count = response.json().values()
    if len(lgs) == 0:
        raise Exception(f"Log with id {id} does not exist")
    lg = lgs[0]
    return unify.Log(
        id=lg["id"],
        ts=lg["ts"],
        **lg["entries"],
        **lg["derived_entries"],
        params={k: (v, params[k][v]) for k, v in lg["params"].items()},
        api_key=api_key,
    )


# noinspection PyShadowingBuiltins
def get_logs_metric(
    *,
    metric: str,
    key: str,
    filter: Optional[str] = None,
    project: Optional[str] = None,
    context: Optional[str] = None,
    from_ids: Optional[List[int]] = None,
    exclude_ids: Optional[List[int]] = None,
    api_key: Optional[str] = None,
) -> Union[float, int, bool]:
    """
    Retrieve a set of log metrics across a project, after applying the filtering.

    Args:
        metric: The reduction metric to compute for the specified key. Supported are:
        sum, mean, var, std, min, max, median, mode.

        key: The key to compute the reduction statistic for.

        filter: The filtering to apply to the various log values, expressed as a string,
        for example:
        "(temperature > 0.5 and (len(system_msg) < 100 or 'no' in usr_response))"

        project: The id of the project to retrieve the logs for.

        context: The context of the logs to retrieve the metrics for.

        from_ids: A list of log IDs to include in the results.

        exclude_ids: A list of log IDs to exclude from the results.

        api_key: If specified, unify API key to be used. Defaults to the value in the
        `UNIFY_KEY` environment variable.

    Returns:
        The full set of reduced log metrics for the project, after optionally applying
        the optional filtering.
    """
    api_key = _validate_api_key(api_key)
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    project = _get_and_maybe_create_project(project, api_key=api_key)
    params = {
        "project": project,
        "filter_expr": filter,
        "key": key,
        "from_ids": "&".join(map(str, from_ids)) if from_ids else None,
        "exclude_ids": "&".join(map(str, exclude_ids)) if exclude_ids else None,
        "context": context if context else CONTEXT_READ.get(),
    }
    response = _requests.get(
        BASE_URL + f"/logs/metric/{metric}",
        headers=headers,
        params=params,
    )
    _check_response(response)
    return response.json()


def get_groups(
    *,
    key: str,
    project: Optional[str] = None,
    filter: Optional[Dict[str, Any]] = None,
    from_ids: Optional[List[int]] = None,
    exclude_ids: Optional[List[int]] = None,
    api_key: Optional[str] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Returns a list of the different version/values of one entry within a given project
    based on its key.

    Args:
        key: Name of the log entry to do equality matching for.

        project: Name of the project to get logs from.

        filter: Boolean string to filter logs, for example:
        "(temperature > 0.5 and (len(system_msg) < 100 or 'no' in usr_response))"

        from_ids: A list of log IDs to include in the results.

        exclude_ids: A list of log IDs to exclude from the results.

        api_key: If specified, unify API key to be used. Defaults to the value in the
        `UNIFY_KEY` environment variable.

    Returns:
        A dict containing the grouped logs, with each key of the dict representing the
        version of the log key with equal values, and the value being the equal value.
    """
    api_key = _validate_api_key(api_key)
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    project = _get_and_maybe_create_project(project, api_key=api_key)
    params = {
        "project": project,
        "key": key,
        "filter_expr": filter,
        "from_ids": from_ids,
        "exclude_ids": exclude_ids,
    }
    response = _requests.get(BASE_URL + "/logs/groups", headers=headers, params=params)
    _check_response(response)
    return response.json()


def get_logs_latest_timestamp(
    *,
    project: Optional[str] = None,
    context: Optional[str] = None,
    column_context: Optional[str] = None,
    filter: Optional[str] = None,
    sort_by: Optional[str] = None,
    from_ids: Optional[List[int]] = None,
    exclude_ids: Optional[List[int]] = None,
    limit: Optional[int] = None,
    offset: Optional[int] = None,
    api_key: Optional[str] = None,
) -> int:
    """
    Returns the update timestamp of the most recently updated log within the specified page and filter bounds.
    """
    api_key = _validate_api_key(api_key)
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    project = _get_and_maybe_create_project(project, api_key=api_key)
    context = context if context else CONTEXT_READ.get()
    column_context = column_context if column_context else COLUMN_CONTEXT_READ.get()
    params = {
        "project": project,
        "context": context,
        "column_context": column_context,
        "filter_expr": filter,
        "sort_by": sort_by,
        "from_ids": "&".join(map(str, from_ids)) if from_ids else None,
        "exclude_ids": "&".join(map(str, exclude_ids)) if exclude_ids else None,
        "limit": limit,
        "offset": offset,
    }
    response = _requests.get(
        BASE_URL + "/logs/latest_timestamp",
        headers=headers,
        params=params,
    )
    _check_response(response)
    return response.json()


def update_derived_log(
    *,
    target: Union[List[int], int],
    key: Optional[str] = None,
    equation: Optional[str] = None,
    referenced_logs: Optional[List[int]] = None,
    project: Optional[str] = None,
    context: Optional[str] = None,
    api_key: Optional[str] = None,
) -> None:
    """
    Update the derived entries for a log.
    """
    api_key = _validate_api_key(api_key)
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    project = _get_and_maybe_create_project(project, api_key=api_key)
    context = context if context else CONTEXT_READ.get()
    params = {
        "project": project,
        "context": context,
        "target": target,
        "key": key,
        "equation": equation,
        "referenced_logs": referenced_logs,
    }
    response = _requests.put(BASE_URL + "/logs/derived", headers=headers, params=params)
    _check_response(response)
    return response.json()


def join_logs(
    *,
    pair_of_args: List[Dict[str, Any]],
    join_expr: str,
    mode: str,
    new_context: str,
    columns: Optional[List[str]] = None,
    project: Optional[str] = None,
    api_key: Optional[str] = None,
):
    """
    Join two sets of logs based on specified criteria and creates new logs with the joined data.
    """
    api_key = _validate_api_key(api_key)
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    project = _get_and_maybe_create_project(project, api_key=api_key)
    params = {
        "project": project,
        "pair_of_args": pair_of_args,
        "join_expr": join_expr,
        "mode": mode,
        "new_context": new_context,
        "columns": columns,
    }
    response = _requests.post(BASE_URL + "/logs/join", headers=headers, params=params)
    _check_response(response)
    return response.json()


def create_fields(
    fields: Union[Dict[str, Any], List[str]],
    *,
    project: Optional[str] = None,
    context: Optional[str] = None,
    api_key: Optional[str] = None,
):
    """
    Creates one or more fields in a project.

    Args:
        fields: Dictionary mapping field names to their types (or None if no explicit type).

        project: Name of the project to create fields in.

        context: The context to create fields in.

        api_key: If specified, unify API key to be used. Defaults to the value in the
        `UNIFY_KEY` environment variable.
    """
    api_key = _validate_api_key(api_key)
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    project = _get_and_maybe_create_project(project, api_key=api_key)
    context = context if context else CONTEXT_WRITE.get()
    if isinstance(fields, list):
        fields = {field: None for field in fields}
    body = {
        "project": project,
        "context": context,
        "fields": fields,
    }
    response = _requests.post(BASE_URL + "/logs/fields", headers=headers, json=body)
    _check_response(response)
    return response.json()


def rename_field(
    name: str,
    new_name: str,
    *,
    project: Optional[str] = None,
    context: Optional[str] = None,
    api_key: Optional[str] = None,
):
    """
    Rename a field in a project.

    Args:
        name: The name of the field to rename.

        new_name: The new name for the field.

        project: Name of the project to rename the field in.

        context: The context to rename the field in.

        api_key: If specified, unify API key to be used. Defaults to the value in the
        `UNIFY_KEY` environment variable.
    """
    api_key = _validate_api_key(api_key)
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    project = _get_and_maybe_create_project(project, api_key=api_key)
    context = context if context else CONTEXT_WRITE.get()
    body = {
        "project": project,
        "context": context,
        "old_field_name": name,
        "new_field_name": new_name,
    }
    response = _requests.post(
        BASE_URL + "/logs/rename_field",
        headers=headers,
        json=body,
    )
    _check_response(response)
    return response.json()


def get_fields(
    *,
    project: Optional[str] = None,
    context: Optional[str] = None,
    api_key: Optional[str] = None,
):
    """
    Get a dictionary of fields names and their types

    Args:
        project: Name of the project to get fields from.

        context: The context to get fields from.

        api_key: If specified, unify API key to be used. Defaults to the value in the
        `UNIFY_KEY` environment variable.

    Returns:
        A dictionary of fields names and their types
    """
    api_key = _validate_api_key(api_key)
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    project = _get_and_maybe_create_project(project, api_key=api_key)
    context = context if context else CONTEXT_READ.get()
    params = {
        "project": project,
        "context": context,
    }
    response = _requests.get(BASE_URL + "/logs/fields", headers=headers, params=params)
    _check_response(response)
    return response.json()


def delete_fields(
    fields: List[str],
    *,
    project: Optional[str] = None,
    context: Optional[str] = None,
    api_key: Optional[str] = None,
):
    """
    Delete one or more fields from a project.

    Args:
        fields: List of field names to delete.

        project: Name of the project to delete fields from.

        context: The context to delete fields from.

        api_key: If specified, unify API key to be used. Defaults to the value in the
        `UNIFY_KEY` environment variable.
    """
    api_key = _validate_api_key(api_key)
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    project = _get_and_maybe_create_project(project, api_key=api_key)
    context = context if context else CONTEXT_WRITE.get()
    body = {
        "project": project,
        "context": context,
        "fields": fields,
    }
    response = _requests.delete(
        BASE_URL + "/logs/fields",
        headers=headers,
        json=body,
    )
    _check_response(response)
    return response.json()


# User Logging #
# -------------#


def set_user_logging(value: bool):
    global USR_LOGGING
    USR_LOGGING = value

```

`/Users/yushaarif/Unify/unify/unify/logging/utils/artifacts.py`:

```py
from typing import Any, Dict, Optional

from unify import BASE_URL
from unify.utils import _requests

from ...utils.helpers import (_check_response, _get_and_maybe_create_project,
                              _validate_api_key)

# Artifacts #
# ----------#


def add_project_artifacts(
    *,
    project: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs,
) -> Dict[str, str]:
    """
    Creates one or more artifacts associated to a project. Artifacts are project-level
    metadata that don’t depend on other variables.

    Args:
        project: Name of the project the artifacts belong to.

        api_key: If specified, unify API key to be used. Defaults to the value in the
        `UNIFY_KEY` environment variable.

        kwargs: Dictionary containing one or more key:value pairs that will be stored
        as artifacts.

    Returns:
        A message indicating whether the artifacts were successfully added.
    """
    api_key = _validate_api_key(api_key)
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    body = {"artifacts": kwargs}
    project = _get_and_maybe_create_project(project, api_key=api_key)
    response = _requests.post(
        BASE_URL + f"/project/{project}/artifacts",
        headers=headers,
        json=body,
    )
    _check_response(response)
    return response.json()


def delete_project_artifact(
    key: str,
    *,
    project: Optional[str] = None,
    api_key: Optional[str] = None,
) -> str:
    """
    Deletes an artifact from a project.

    Args:
        project: Name of the project to delete an artifact from.

        key: Key of the artifact to delete.

        api_key: If specified, unify API key to be used. Defaults to the value in the
        `UNIFY_KEY` environment variable.

    Returns:
        Whether the artifact was successfully deleted.
    """
    api_key = _validate_api_key(api_key)
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    project = _get_and_maybe_create_project(project, api_key=api_key)
    response = _requests.delete(
        BASE_URL + f"/project/{project}/artifacts/{key}",
        headers=headers,
    )
    _check_response(response)
    return response.json()


def get_project_artifacts(
    *,
    project: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Returns the key-value pairs for all artifacts in a project.

    Args:
        project: Name of the project to delete an artifact from.

        api_key: If specified, unify API key to be used. Defaults to the value in the
        `UNIFY_KEY` environment variable.

    Returns:
        A dictionary of all artifacts associated with the project, with keys for
        artifact names and values for the artifacts themselves.
    """
    api_key = _validate_api_key(api_key)
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    project = _get_and_maybe_create_project(project, api_key=api_key)
    response = _requests.get(
        BASE_URL + f"/project/{project}/artifacts",
        headers=headers,
    )
    _check_response(response)
    return response.json()

```

`/Users/yushaarif/Unify/unify/unify/logging/utils/contexts.py`:

```py
from typing import Dict, List, Optional

from unify import BASE_URL
from unify.utils import _requests

from ...utils.helpers import (_check_response, _get_and_maybe_create_project,
                              _validate_api_key)
from .logs import CONTEXT_WRITE

# Contexts #
# ---------#


def create_context(
    name: str,
    description: str = None,
    is_versioned: bool = False,
    allow_duplicates: bool = True,
    *,
    project: Optional[str] = None,
    api_key: Optional[str] = None,
) -> None:
    """
    Create a context.

    Args:
        name: Name of the context to create.

        description: Description of the context to create.

        is_versioned: Whether the context is versioned.

        allow_duplicates: Whether to allow duplicates in the context.

        project: Name of the project the context belongs to.

        api_key: If specified, unify API key to be used. Defaults to the value in the
        `UNIFY_KEY` environment variable.

    Returns:
        A message indicating whether the context was successfully created.
    """
    api_key = _validate_api_key(api_key)
    project = _get_and_maybe_create_project(
        project,
        api_key=api_key,
        create_if_missing=False,
    )
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    body = {
        "name": name,
        "description": description,
        "is_versioned": is_versioned,
        "allow_duplicates": allow_duplicates,
    }
    response = _requests.post(
        BASE_URL + f"/project/{project}/contexts",
        headers=headers,
        json=body,
    )
    _check_response(response)
    return response.json()


def rename_context(
    name: str,
    new_name: str,
    *,
    project: Optional[str] = None,
    api_key: Optional[str] = None,
) -> None:
    """
    Rename a context.

    Args:
        name: Name of the context to rename.

        new_name: New name of the context.

        project: Name of the project the context belongs to.

        api_key: If specified, unify API key to be used. Defaults to the value in the
        `UNIFY_KEY` environment variable.
    """
    api_key = _validate_api_key(api_key)
    project = _get_and_maybe_create_project(
        project,
        api_key=api_key,
        create_if_missing=False,
    )
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    response = _requests.patch(
        BASE_URL + f"/project/{project}/contexts/{name}/rename",
        headers=headers,
        json={"name": new_name},
    )
    _check_response(response)
    return response.json()


def get_context(
    name: str,
    *,
    project: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Dict[str, str]:
    """
    Get information about a specific context including its versioning status and current version.

    Args:
        name: Name of the context to get.

        project: Name of the project the context belongs to.

        api_key: If specified, unify API key to be used. Defaults to the value in the
        `UNIFY_KEY` environment variable.
    """
    api_key = _validate_api_key(api_key)
    project = _get_and_maybe_create_project(
        project,
        api_key=api_key,
        create_if_missing=False,
    )
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    response = _requests.get(
        BASE_URL + f"/project/{project}/contexts/{name}",
        headers=headers,
    )
    _check_response(response)
    return response.json()


def get_contexts(
    project: Optional[str] = None,
    *,
    prefix: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Dict[str, str]:
    """
    Gets all contexts associated with a project, with the corresponding prefix.

    Args:
        prefix: Prefix of the contexts to get.

        project: Name of the project the artifacts belong to.

        api_key: If specified, unify API key to be used. Defaults to the value in the
        `UNIFY_KEY` environment variable.

        kwargs: Dictionary containing one or more key:value pairs that will be stored
        as artifacts.

    Returns:
        A message indicating whether the artifacts were successfully added.
    """
    api_key = _validate_api_key(api_key)
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    project = _get_and_maybe_create_project(
        project,
        api_key=api_key,
        create_if_missing=False,
    )
    response = _requests.get(
        BASE_URL + f"/project/{project}/contexts",
        headers=headers,
    )
    _check_response(response)
    contexts = response.json()
    contexts = {context["name"]: context["description"] for context in contexts}
    if prefix:
        contexts = {
            context: description
            for context, description in contexts.items()
            if context.startswith(prefix)
        }
    return contexts


def delete_context(
    name: str,
    *,
    project: Optional[str] = None,
    api_key: Optional[str] = None,
) -> None:
    """
    Delete a context from the server.

    Args:
        name: Name of the context to delete.

        project: Name of the project the context belongs to.

        api_key: If specified, unify API key to be used. Defaults to the value in the
        `UNIFY_KEY` environment variable.
    """
    api_key = _validate_api_key(api_key)
    project = _get_and_maybe_create_project(
        project,
        api_key=api_key,
        create_if_missing=False,
    )
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    response = _requests.delete(
        BASE_URL + f"/project/{project}/contexts/{name}",
        headers=headers,
    )
    _check_response(response)
    return response.json()


def add_logs_to_context(
    log_ids: List[int],
    *,
    context: Optional[str] = None,
    project: Optional[str] = None,
    api_key: Optional[str] = None,
) -> None:
    """
    Add logs to a context.

    Args:
        log_ids: List of log ids to add to the context.

        context: Name of the context to add the logs to.

        project: Name of the project the logs belong to.

        api_key: If specified, unify API key to be used. Defaults to the value in the
        `UNIFY_KEY` environment variable.

    Returns:
        A message indicating whether the logs were successfully added to the context.
    """
    api_key = _validate_api_key(api_key)
    context = context if context else CONTEXT_WRITE.get()
    project = _get_and_maybe_create_project(
        project,
        api_key=api_key,
        create_if_missing=False,
    )
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    body = {
        "context_name": context,
        "log_ids": log_ids,
    }
    response = _requests.post(
        BASE_URL + f"/project/{project}/contexts/add_logs",
        headers=headers,
        json=body,
    )
    _check_response(response)
    return response.json()

```

`/Users/yushaarif/Unify/unify/unify/logging/utils/projects.py`:

```py
from typing import Dict, List, Optional

from unify import BASE_URL
from unify.utils import _requests

from ...utils.helpers import _check_response, _validate_api_key

# Projects #
# ---------#


def create_project(
    name: str,
    *,
    overwrite: bool = False,
    api_key: Optional[str] = None,
) -> Dict[str, str]:
    """
    Creates a logging project and adds this to your account. This project will have
    a set of logs associated with it.

    Args:
        name: A unique, user-defined name used when referencing the project.

        overwrite: Whether to overwrite an existing project if is already exists.

        api_key: If specified, unify API key to be used. Defaults to the value in the
        `UNIFY_KEY` environment variable.

    Returns:
        A message indicating whether the project was created successfully.
    """
    api_key = _validate_api_key(api_key)
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    body = {"name": name}
    if overwrite:
        if name in list_projects(api_key=api_key):
            delete_project(name=name, api_key=api_key)
    response = _requests.post(BASE_URL + "/project", headers=headers, json=body)
    _check_response(response)
    return response.json()


def rename_project(
    name: str,
    new_name: str,
    *,
    api_key: Optional[str] = None,
) -> Dict[str, str]:
    """
    Renames a project from `name` to `new_name` in your account.

    Args:
        name: Name of the project to rename.

        new_name: A unique, user-defined name used when referencing the project.

        api_key: If specified, unify API key to be used. Defaults to the value in the
        `UNIFY_KEY` environment variable.

    Returns:
        A message indicating whether the project was successfully renamed.
    """
    api_key = _validate_api_key(api_key)
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    body = {"name": new_name}
    response = _requests.patch(
        BASE_URL + f"/project/{name}",
        headers=headers,
        json=body,
    )
    _check_response(response)
    return response.json()


def delete_project(
    name: str,
    *,
    api_key: Optional[str] = None,
) -> str:
    """
    Deletes a project from your account.

    Args:
        name: Name of the project to delete.

        api_key: If specified, unify API key to be used. Defaults to the value in the
        `UNIFY_KEY` environment variable.

    Returns:
        Whether the project was successfully deleted.
    """
    api_key = _validate_api_key(api_key)
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    response = _requests.delete(BASE_URL + f"/project/{name}", headers=headers)
    _check_response(response)
    return response.json()


def delete_project_logs(
    name: str,
    *,
    api_key: Optional[str] = None,
) -> None:
    """
    Deletes all logs from a project.

    Args:
        name: Name of the project to delete logs from.

        api_key: If specified, unify API key to be used. Defaults to the value in the
        `UNIFY_KEY` environment variable.
    """
    api_key = _validate_api_key(api_key)
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    response = _requests.delete(BASE_URL + f"/project/{name}/logs", headers=headers)
    _check_response(response)
    return response.json()


def delete_project_contexts(
    name: str,
    *,
    api_key: Optional[str] = None,
) -> None:
    """
    Deletes all contexts and their associated logs from a project

    Args:
        name: Name of the project to delete contexts from.

        api_key: If specified, unify API key to be used. Defaults to the value in the
        `UNIFY_KEY` environment variable.
    """
    api_key = _validate_api_key(api_key)
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    response = _requests.delete(BASE_URL + f"/project/{name}/contexts", headers=headers)
    _check_response(response)
    return response.json()


def list_projects(
    *,
    api_key: Optional[str] = None,
) -> List[str]:
    """
    Returns the names of all projects stored in your account.

    Args:
        api_key: If specified, unify API key to be used. Defaults to the value in the
        `UNIFY_KEY` environment variable.

    Returns:
        List of all project names.
    """
    api_key = _validate_api_key(api_key)
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    response = _requests.get(BASE_URL + "/projects", headers=headers)
    _check_response(response)
    return response.json()

```

`/Users/yushaarif/Unify/unify/unify/logging/utils/compositions.py`:

```py
from __future__ import annotations

import json

from ...utils.helpers import _validate_api_key
from .logs import *

# Parameters #
# -----------#


def get_param_by_version(
    field: str,
    version: Union[str, int],
    api_key: Optional[str] = None,
) -> Any:
    """
    Gets the parameter by version.

    Args:
        field: The field of the parameter to get.

        version: The version of the parameter to get.

        api_key: If specified, unify API key to be used. Defaults to the value in the
        `UNIFY_KEY` environment variable.

    Returns:
        The parameter by version.
    """
    api_key = _validate_api_key(api_key)
    version = str(version)
    filter_exp = f"version({field}) == {version}"
    return get_logs(filter=filter_exp, limit=1, api_key=api_key)[0].params[field][1]


def get_param_by_value(
    field: str,
    value: Any,
    api_key: Optional[str] = None,
) -> Any:
    """
    Gets the parameter by value.

    Args:
        field: The field of the parameter to get.

        value: The value of the parameter to get.

        api_key: If specified, unify API key to be used. Defaults to the value in the
        `UNIFY_KEY` environment variable.

    Returns:
        The parameter by version.
    """
    api_key = _validate_api_key(api_key)
    filter_exp = f"{field} == {json.dumps(value)}"
    return get_logs(filter=filter_exp, limit=1, api_key=api_key)[0].params[field][0]


def get_source() -> str:
    """
    Extracts the source code for the file from where this function was called.

    Returns:
        The source code for the file, as a string.
    """
    frame = inspect.getouterframes(inspect.currentframe())[1]
    with open(frame.filename, "r") as file:
        source = file.read()
    return f"```python\n{source}\n```"


# Experiments #
# ------------#


def get_experiment_name(version: int, api_key: Optional[str] = None) -> str:
    """
    Gets the experiment name (by version).

    Args:
        version: The version of the experiment to get.

        api_key: If specified, unify API key to be used. Defaults to the value in the
        `UNIFY_KEY` environment variable.

    Returns:
        The experiment name with said version.
    """
    experiments = get_groups(key="experiment", api_key=api_key)
    if not experiments:
        return None
    elif version < 0:
        version = len(experiments) + version
    if str(version) not in experiments:
        return None
    return experiments[str(version)]


def get_experiment_version(name: str, api_key: Optional[str] = None) -> int:
    """
    Gets the experiment version (by name).

    Args:
        name: The name of the experiment to get.

        api_key: If specified, unify API key to be used. Defaults to the value in the
        `UNIFY_KEY` environment variable.

    Returns:
        The experiment version with said name.
    """
    experiments = get_groups(key="experiment", api_key=api_key)
    if not experiments:
        return None
    experiments = {v: k for k, v in experiments.items()}
    if name not in experiments:
        return None
    return int(experiments[name])

```

`/Users/yushaarif/Unify/unify/unify/logging/dataset.py`:

```py
from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any, Dict, List, Optional, Union

import unify
from typing_extensions import Self

from ..universal_api.types import Prompt
# noinspection PyProtectedMember
from ..utils.helpers import _validate_api_key


def _to_raw_data(x: Dict[str, Any]):
    return x["data"] if "data" in x and len(x) == 1 else x


class Dataset(Sequence):
    def __init__(
        self,
        data: Optional[Any] = None,
        *,
        name: str = None,
        allow_duplicates: bool = False,
        api_key: Optional[str] = None,
    ) -> None:
        """
        Initialize a local dataset.

        Args:
            data: The data for populating the dataset.
            This needs to be a list of JSON serializable objects.

            name: The name of the dataset. To create a dataset for a specific project
            with name {project_name}, then prefix the name with {project_name}/{name}.

            allow_duplicates: Whether to allow duplicates in the dataset.

            api_key: API key for accessing the Unify API. If None, it attempts to
            retrieve the API key from the environment variable UNIFY_KEY. Defaults to
            None.

        Raises:
            UnifyError: If the API key is missing.
        """
        self._name = name
        if isinstance(data, tuple):
            data = list(data)
        elif not isinstance(data, list):
            data = [data]
        self._allow_duplicates = allow_duplicates
        self._api_key = _validate_api_key(api_key)
        self._logs = [
            (
                entry
                if isinstance(entry, unify.Log)
                else unify.Log(
                    **(entry if isinstance(entry, dict) else {"data": entry}),
                )
            )
            for entry in data
        ]
        super().__init__()

    @property
    def name(self) -> str:
        """
        Name of the dataset.
        """
        return self._name

    @property
    def allow_duplicates(self) -> bool:
        """
        Whether to allow duplicates in the dataset.
        """
        return self._allow_duplicates

    @property
    def data(self):
        """
        Dataset entries.
        """
        return [_to_raw_data(l.entries) for l in self._logs]

    def _set_data(self, data):
        self._logs = [
            unify.Log(**(entry if isinstance(entry, dict) else {"data": entry}))
            for entry in data
        ]

    def set_name(self, name: str) -> Self:
        """
        Set the name of the dataset.

        Args:
            name: The name to set the dataset to.

        Returns:
            This dataset, useful for chaining methods.
        """
        self._name = name
        return self

    def set_allow_duplicates(self, allow_duplicates: bool) -> Self:
        """
        Set whether to allow duplicates in the dataset.

        Args:
            allow_duplicates: Whether to allow duplicates in the dataset.

        Returns:
            This dataset, useful for chaining methods.
        """
        self._allow_duplicates = allow_duplicates
        return self

    @staticmethod
    def from_upstream(
        name: str,
        api_key: Optional[str] = None,
    ) -> Dataset:
        """
        Initialize a local dataset from the upstream dataset.

        Args:
            name: The name of the dataset.

            api_key: API key for accessing the Unify API. If None, it attempts to
            retrieve the API key from the environment variable UNIFY_KEY. Defaults to
            None.

        Returns:
            The dataset, with contents downloaded from upstream.

        Raises:
            UnifyError: If the API key is missing.
        """
        data = unify.download_dataset(name=name, api_key=api_key)
        return Dataset(
            data,
            name=name,
            api_key=api_key,
        )

    def _assert_name_exists(self) -> None:
        assert self._name is not None, (
            "Dataset name must be specified in order to upload, download, sync or "
            "compare to a corresponding dataset in your upstream account. "
            "You can simply use .set_name() and set it to the same name as your "
            "upstream dataset, or create a new name if it doesn't yet exist upstream."
        )

    def upload(self, overwrite: bool = False) -> Self:
        """
        Uploads all unique local data in the dataset to the user account upstream.
        This function will not download any uniques from upstream.
        Use `sync` to synchronize and superset the datasets in both directions.
        Set `overwrite=True` to disregard any pre-existing upstream data.

        Args:
            overwrite: Whether to overwrite the upstream dataset if it already exists.

        Returns:
            This dataset, useful for chaining methods.
        """
        self._assert_name_exists()
        dataset_ids = unify.upload_dataset(
            name=self._name,
            data=self._logs,
            overwrite=overwrite,
            allow_duplicates=self._allow_duplicates,
        )
        assert len(dataset_ids) >= len(
            self._logs,
        ), "Number of upstream items must be greater than or equal to items"
        return self

    def download(self, overwrite: bool = False) -> Self:
        """
        Downloads all unique upstream data from the user account to the local dataset.
        This function will not upload any unique values stored locally.
        Use `sync` to synchronize and superset the datasets in both directions.
        Set `overwrite=True` to disregard any pre-existing data stored in this class.

        Args:
            overwrite: Whether to overwrite the local data, if any already exists

        Returns:
            This dataset after the in-place download, useful for chaining methods.
        """
        self._assert_name_exists()

        if f"Datasets/{self._name}" not in unify.get_contexts():
            upstream_dataset = list()
        else:
            upstream_dataset = unify.download_dataset(
                name=self._name,
                api_key=self._api_key,
            )
        if overwrite:
            self._logs = upstream_dataset
            return self
        if self._allow_duplicates:
            local_ids = set([l.id for l in self._logs if l.id is not None])
            new_data = [l for l in upstream_dataset if l.id not in local_ids]
        else:
            local_vals_to_logs = {json.dumps(l.entries): l for l in self._logs}
            local_values = set([json.dumps(l.entries) for l in self._logs])
            upstream_values = set()
            new_data = list()
            for l in upstream_dataset:
                uid = l.id
                value = json.dumps(l.entries)
                if value not in local_values.union(upstream_values):
                    new_data.append(l)
                    upstream_values.add(value)
                elif value in local_vals_to_logs:
                    local_log = local_vals_to_logs[value]
                    if local_log.id != uid:
                        local_log.set_id(uid)

        self._logs += new_data
        return self

    def sync(self) -> Self:
        """
        Synchronize the dataset in both directions, downloading any values missing
        locally, and uploading any values missing from upstream in the account.

        Returns:
            This dataset after the in-place sync, useful for chaining methods.
        """
        self.upload()
        self.download(overwrite=True)
        return self

    def upstream_diff(self) -> Self:
        """
        Prints the difference between the local dataset and the upstream dataset.

        Returns:
            This dataset after printing the diff, useful for chaining methods.
        """
        self._assert_name_exists()
        upstream_dataset = unify.download_dataset(
            name=self._name,
            api_key=self._api_key,
        )
        unique_upstream = [
            item["entry"] for item in upstream_dataset if item["entry"] not in self.data
        ]
        print(
            "The following {} entries are stored upstream but not locally\n: "
            "{}".format(len(unique_upstream), unique_upstream),
        )
        unique_local = [item for item in self.data if item not in upstream_dataset]
        print(
            "The following {} entries are stored upstream but not locally\n: "
            "{}".format(len(unique_local), unique_local),
        )
        return self

    def add(
        self,
        other: Union[
            Dataset,
            str,
            Dict,
            Prompt,
            int,
            List[Union[str, Dict, Prompt]],
        ],
    ) -> Self:
        """
        Adds another dataset to this one, return a new Dataset instance, with this
        new dataset receiving all unique queries from the other added dataset.

        Args:
            other: The other dataset being added to this one.

        Returns:
            The new dataset following the addition.
        """
        if other == 0:
            return self
        other = other if isinstance(other, Dataset) else Dataset(other)
        data = self.data + [d for d in other.data if d not in self.data]
        return Dataset(data=data, api_key=self._api_key)

    def sub(
        self,
        other: Union[Dataset, str, Dict, Prompt, List[Union[str, Dict, Prompt]]],
    ) -> Self:
        """
        Subtracts another dataset from this one, return a new Dataset instance, with
        this new dataset losing all queries from the other subtracted dataset.

        Args:
            other: The other dataset being added to this one.

        Returns:
            The new dataset following the subtraction.
        """
        other = other if isinstance(other, Dataset) else Dataset(other)
        assert other in self, (
            "cannot subtract dataset B from dataset A unless all queries of dataset "
            "B are also present in dataset A"
        )
        data = [item for item in self.data if item not in other]
        return Dataset(data=data, api_key=self._api_key)

    def inplace_add(
        self,
        other: Union[
            Dataset,
            str,
            Dict,
            Prompt,
            int,
            List[Union[str, Dict, Prompt]],
        ],
    ) -> Self:
        """
        Adds another dataset to this one, with this dataset receiving all unique queries
        from the other added dataset.

        Args:
            other: The other dataset being added to this one.

        Returns:
            This dataset following the in-place addition.
        """
        if other == 0:
            return self
        other = other if isinstance(other, Dataset) else Dataset(other)
        self._logs = self._logs + [d for d in other._logs if d not in self._logs]
        return self

    def inplace_sub(
        self,
        other: Union[Dataset, str, Dict, Prompt, List[Union[str, Dict, Prompt]]],
    ) -> Self:
        """
        Subtracts another dataset from this one, with this dataset losing all queries
        from the other subtracted dataset.

        Args:
            other: The other dataset being added to this one.

        Returns:
            This dataset following the in-place subtraction.
        """
        other = other if isinstance(other, Dataset) else Dataset(other)
        assert other in self, (
            "cannot subtract dataset B from dataset A unless all queries of dataset "
            "B are also present in dataset A"
        )
        self._logs = [item for item in self._logs if item not in other]
        return self

    def __add__(
        self,
        other: Union[Dataset, str, Dict, Prompt, List[Union[str, Dict, Prompt]]],
    ) -> Self:
        """
        Adds another dataset to this one via the + operator, return a new Dataset
        instance, with this new dataset receiving all unique queries from the other
        added dataset.

        Args:
            other: The other dataset being added to this one.

        Returns:
            The new dataset following the addition.
        """
        return self.add(other)

    def __radd__(
        self,
        other: Union[
            Dataset,
            str,
            Dict,
            Prompt,
            int,
            List[Union[str, Dict, Prompt]],
        ],
    ) -> Self:
        """
        Adds another dataset to this one via the + operator, this is used if the
        other item does not have a valid __add__ method for these two types. Return a
        new Dataset instance, with this new dataset receiving all unique queries from
        the other added dataset.

        Args:
            other: The other dataset being added to this one.

        Returns:
            The new dataset following the addition.
        """
        if other == 0:
            return self
        return Dataset(other).add(self)

    def __iadd__(
        self,
        other: Union[Dataset, str, Dict, Prompt, List[Union[str, Dict, Prompt]]],
    ) -> Self:
        """
        Adds another dataset to this one, with this dataset receiving all unique queries
        from the other added dataset.

        Args:
            other: The other dataset being added to this one.

        Returns:
            This dataset following the in-place addition.
        """
        return self.inplace_add(other)

    def __sub__(
        self,
        other: Union[Dataset, str, Dict, Prompt, List[Union[str, Dict, Prompt]]],
    ) -> Self:
        """
        Subtracts another dataset from this one via the - operator, return a new Dataset
        instance, with this new dataset losing all queries from the other subtracted
        dataset.

        Args:
            other: The other dataset being subtracted from this one.

        Returns:
            The new dataset following the subtraction.
        """
        return self.sub(other)

    def __rsub__(
        self,
        other: Union[Dataset, str, Dict, Prompt, List[Union[str, Dict, Prompt]]],
    ) -> Self:
        """
        Subtracts another dataset from this one via the - operator, this is used if the
        other item does not have a valid __sub__ method for these two types. Return a
        new Dataset instance, with this new dataset losing all queries from the other
        subtracted dataset.

        Args:
            other: The other dataset being subtracted from this one.

        Returns:
            The new dataset following the subtraction.
        """
        return Dataset(other).sub(self)

    def __isub__(
        self,
        other: Union[Dataset, str, Dict, Prompt, List[Union[str, Dict, Prompt]]],
    ) -> Self:
        """
        Subtracts another dataset from this one, with this dataset losing all queries
        from the other subtracted dataset.

        Args:
            other: The other dataset being added to this one.

        Returns:
            This dataset following the in-place subtraction.
        """
        return self.inplace_sub(other)

    def __iter__(self) -> Any:
        """
        Iterates through the dataset, return one instance at a time.

        Returns:
            The next instance in the dataset.
        """
        for l in self._logs:
            yield l

    def __contains__(
        self,
        item: Union[Dataset, str, Dict, Prompt, List[Union[str, Dict, Prompt]]],
    ) -> bool:
        """
        Determine whether the item is contained within the dataset. The item is cast to
        a Dataset instance, and can therefore take on many different types. Only returns
        True if *all* entries in the passed dataset are contained within this dataset.

        Args:
            item: The item to cast to a Dataset before checking if it's a subset of this
            one.

        Returns:
            Boolean, whether the passed Dataset is a subset of this one.
        """
        item = item if isinstance(item, Dataset) else Dataset(item)
        this_serialized = [
            json.dumps(
                {
                    k: v
                    for k, v in l.to_json().items()
                    if k not in ("id", "ts") and v not in ({}, None)
                },
            )
            for l in self._logs
        ]
        item_serialized = [
            json.dumps(
                {
                    k: v
                    for k, v in l.to_json().items()
                    if k not in ("id", "ts") and v not in ({}, None)
                },
            )
            for l in item._logs
        ]
        this_set = set(this_serialized)
        combined_set = set(this_serialized + item_serialized)
        return len(this_set) == len(combined_set)

    def __len__(self) -> int:
        """
        Returns the number of entries contained within the dataset.

        Returns:
            The number of entries in the dataset.
        """
        return len(self._logs)

    def __getitem__(self, item: Union[int, slice]) -> Union[Any, Dataset]:
        """
        Gets an item from the dataset, either via an int or slice. In the case of an
        int, then a data instance is returned, and for a slice a Dataset instance is
        returned.

        Args:
            item: integer or slice for extraction.

        Returns:
            An individual item or Dataset slice, for int and slice queries respectively.
        """
        if isinstance(item, int):
            return self._logs[item]
        elif isinstance(item, slice):
            return Dataset(self._logs[item.start : item.stop : item.step])
        raise TypeError(
            "expected item to be of type int or slice,"
            "but found {} of type {}".format(item, type(item)),
        )

    def __setitem__(self, item: Union[int, slice], value: Union[Any, Dataset]):
        if isinstance(item, int):
            if isinstance(value, unify.Log):
                self._logs[item] = value
            else:
                self._logs[item] = unify.Log(
                    **(value if isinstance(value, dict) else {"data": value}),
                )
        elif isinstance(item, slice):
            self._logs[item.start : item.stop : item.step] = [
                (
                    unify.Log(**(v if isinstance(v, dict) else {"data": v}))
                    if not isinstance(v, unify.Log)
                    else v
                )
                for v in value
            ]
        else:
            raise TypeError(
                "expected item to be of type int or slice,",
            )

    def __repr__(self):
        return f"unify.Dataset({self.data}, name='{self._name}')"

```

`/Users/yushaarif/Unify/unify/unify/logging/logs.py`:

```py
from __future__ import annotations

import ast
import copy
import functools
import inspect
import textwrap
import time
import traceback
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

from ..utils.helpers import (_make_json_serializable, _prune_dict,
                             _validate_api_key)
from .utils.compositions import *
from .utils.logs import (_get_trace_logger, _handle_special_types,
                         _initialize_trace_logger)
from .utils.logs import log as unify_log

# Context Handlers #
# -----------------#


def _validate_mode(mode: str) -> None:
    assert mode in (
        "both",
        "read",
        "write",
    ), f"mode must be one of 'read', 'write', or 'both', but found {mode}"


def _validate_mode_nesting(parent_mode: str, child_mode: str) -> None:
    if not (parent_mode in ("both", child_mode)):
        raise Exception(
            f"Cannot nest context with mode '{child_mode}' under parent with mode '{parent_mode}'",
        )


# noinspection PyShadowingBuiltins
class Log:
    def __init__(
        self,
        *,
        id: int = None,
        _future=None,
        ts: Optional[datetime] = None,
        project: Optional[str] = None,
        context: Optional[str] = None,
        api_key: Optional[str] = None,
        params: Dict[str, Tuple[str, Any]] = None,
        **entries,
    ):
        self._id = id
        self._future = _future
        self._ts = ts
        self._project = project
        self._context = context
        self._entries = entries
        self._params = params
        self._api_key = _validate_api_key(api_key)

    # Setters

    def set_id(self, id: int) -> None:
        self._id = id

    # Properties

    @property
    def context(self) -> Optional[str]:
        return self._context

    @property
    def id(self) -> int:
        if self._id is None and self._future is not None and self._future.done():
            self._id = self._future.result()
        return self._id

    @property
    def ts(self) -> Optional[datetime]:
        return self._ts

    @property
    def entries(self) -> Dict[str, Any]:
        return self._entries

    @property
    def params(self) -> Dict[str, Tuple[str, Any]]:
        return self._params

    # Dunders

    def __eq__(self, other: Union[dict, Log]) -> bool:
        if isinstance(other, dict):
            other = Log(id=other["id"], **other["entries"])
        if self._id is not None and other._id is not None:
            return self._id == other._id
        return self.to_json() == other.to_json()

    def __len__(self):
        return len(self._entries) + len(self._params)

    def __repr__(self) -> str:
        return f"Log(id={self._id})"

    # Public

    def download(self):
        # If id is not yet resolved, wait for the future
        if self._id is None and self._future is not None:
            self._id = self._future.result(timeout=5)
        log = get_log_by_id(id=self._id, api_key=self._api_key)
        self._params = log._params
        self._entries = log._entries

    def add_entries(self, **entries) -> None:
        add_log_entries(logs=self._id, api_key=self._api_key, **entries)
        self._entries = {**self._entries, **entries}

    def update_entries(self, **entries) -> None:
        update_logs(
            logs=self._id,
            api_key=self._api_key,
            context=self._context,
            entries=entries,
            overwrite=True,
        )
        self._entries = {**self._entries, **entries}

    def delete_entries(
        self,
        keys_to_delete: List[str],
    ) -> None:
        for key in keys_to_delete:
            delete_log_fields(field=key, logs=self._id, api_key=self._api_key)
            del self._entries[key]

    def delete(self) -> None:
        delete_logs(logs=self._id, api_key=self._api_key)

    def to_json(self):
        return {
            "id": self._id,
            "ts": self._ts,
            "entries": self._entries,
            "params": self._params,
            "api_key": self._api_key,
        }

    @staticmethod
    def from_json(state):
        entries = state["entries"]
        del state["entries"]
        state = {**state, **entries}
        return Log(**state)

    # Context #

    def __enter__(self):
        lg = unify.log(
            project=self._project,
            new=True,
            api_key=self._api_key,
            **self._entries,
        )
        self._log_token = ACTIVE_LOG.set(ACTIVE_LOG.get() + [lg])
        self._active_log_set = False
        self._id = lg.id
        self._ts = lg.ts

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._id is None and self._future is not None:
            self.download()
        ACTIVE_LOG.reset(self._log_token)


class LogGroup:
    def __init__(self, field, value: Union[List[unify.Log], "LogGroup"] = None):
        self.field = field
        self.value = value

    def __repr__(self):
        return f"LogGroup(field={self.field}, value={self.value})"


def _join_path(base_path: str, context: str) -> str:
    return os.path.join(
        base_path,
        os.path.normpath(context),
    ).replace("\\", "/")


def set_context(context: str, mode: str = "both", overwrite: bool = False):
    global MODE, MODE_TOKEN, CONTEXT_READ_TOKEN, CONTEXT_WRITE_TOKEN
    MODE = mode
    _validate_mode_nesting(CONTEXT_MODE.get(), mode)
    MODE_TOKEN = CONTEXT_MODE.set(mode)

    if overwrite and context in unify.get_contexts():
        if mode == "read":
            raise Exception(f"Cannot overwrite logs in read mode.")
        unify.delete_context(context)
    if context not in unify.get_contexts():
        unify.create_context(context)

    if mode in ("both", "write"):
        CONTEXT_WRITE_TOKEN = CONTEXT_WRITE.set(
            _join_path(CONTEXT_WRITE.get(), context),
        )
    if mode in ("both", "read"):
        CONTEXT_READ_TOKEN = CONTEXT_READ.set(
            _join_path(CONTEXT_READ.get(), context),
        )


def unset_context():
    global MODE, MODE_TOKEN, CONTEXT_READ_TOKEN, CONTEXT_WRITE_TOKEN
    if MODE in ("both", "write"):
        CONTEXT_WRITE.reset(CONTEXT_WRITE_TOKEN)
    if MODE in ("both", "read"):
        CONTEXT_READ.reset(CONTEXT_READ_TOKEN)

    CONTEXT_MODE.reset(MODE_TOKEN)


class Context:
    def __init__(self, context: str, mode: str = "both", overwrite: bool = False):
        self._context = context
        _validate_mode(mode)
        self._mode = mode
        self._overwrite = overwrite

    def __enter__(self):
        _validate_mode_nesting(CONTEXT_MODE.get(), self._mode)
        self._mode_token = CONTEXT_MODE.set(self._mode)

        if self._mode in ("both", "write"):
            self._context_write_token = CONTEXT_WRITE.set(
                _join_path(CONTEXT_WRITE.get(), self._context),
            )
        if self._mode in ("both", "read"):
            self._context_read_token = CONTEXT_READ.set(
                _join_path(CONTEXT_READ.get(), self._context),
            )

        if self._overwrite and self._context in unify.get_contexts():
            if self._mode == "read":
                raise Exception(f"Cannot overwrite logs in read mode.")

            unify.delete_context(self._context)
            unify.create_context(self._context)

    def __exit__(self, *args, **kwargs):
        if self._mode in ("both", "write"):
            CONTEXT_WRITE.reset(self._context_write_token)
        if self._mode in ("both", "read"):
            CONTEXT_READ.reset(self._context_read_token)

        CONTEXT_MODE.reset(self._mode_token)


class ColumnContext:
    def _join_path(self, base_path: str, context: str) -> str:
        return os.path.join(
            base_path,
            os.path.normpath(context),
            "",
        ).replace("\\", "/")

    def __init__(self, context: str, mode: str = "both", overwrite: bool = False):
        self._col_context = context
        _validate_mode(mode)
        self._mode = mode
        self._overwrite = overwrite

    def __enter__(self):
        _validate_mode_nesting(COLUMN_CONTEXT_MODE.get(), self._mode)
        self._mode_token = COLUMN_CONTEXT_MODE.set(self._mode)

        if self._mode in ("both", "write"):
            self._context_write_token = COLUMN_CONTEXT_WRITE.set(
                self._join_path(COLUMN_CONTEXT_WRITE.get(), self._col_context),
            )
        if self._mode in ("both", "read"):
            self._context_read_token = COLUMN_CONTEXT_READ.set(
                self._join_path(COLUMN_CONTEXT_READ.get(), self._col_context),
            )

        if self._overwrite:
            if self._mode == "read":
                raise Exception(f"Cannot overwrite logs in read mode.")

            logs = unify.get_logs(return_ids_only=True)
            if len(logs) > 0:
                unify.delete_logs(logs=logs)

    def __exit__(self, *args, **kwargs):
        if self._mode in ("both", "write"):
            COLUMN_CONTEXT_WRITE.reset(self._context_write_token)
        if self._mode in ("both", "read"):
            COLUMN_CONTEXT_READ.reset(self._context_read_token)
        COLUMN_CONTEXT_MODE.reset(self._mode_token)


class Entries:
    def __init__(self, mode: str = "both", overwrite: bool = False, **entries):
        self._entries = _handle_special_types(entries)
        _validate_mode(mode)
        self._mode = mode
        self._overwrite = overwrite

    def __enter__(self):
        _validate_mode_nesting(ACTIVE_ENTRIES_MODE.get(), self._mode)
        self._mode_token = ACTIVE_ENTRIES_MODE.set(self._mode)
        if self._mode in ("both", "write"):
            self._entries_token = ACTIVE_ENTRIES_WRITE.set(
                {**ACTIVE_ENTRIES_WRITE.get(), **self._entries},
            )
            self._nest_token = ENTRIES_NEST_LEVEL.set(
                ENTRIES_NEST_LEVEL.get() + 1,
            )

        if self._mode in ("both", "read"):
            self._entries_read_token = ACTIVE_ENTRIES_READ.set(
                {**ACTIVE_ENTRIES_READ.get(), **self._entries},
            )

        if self._overwrite:
            if self._mode == "read":
                raise Exception(f"Cannot overwrite logs in read mode.")

            logs = unify.get_logs(return_ids_only=True)
            if len(logs) > 0:
                unify.delete_logs(logs=logs)

    def __exit__(self, *args, **kwargs):
        if self._mode in ("both", "write"):
            ACTIVE_ENTRIES_WRITE.reset(self._entries_token)
            ENTRIES_NEST_LEVEL.reset(self._nest_token)
            if ENTRIES_NEST_LEVEL.get() == 0:
                LOGGED.set({})

        if self._mode in ("both", "read"):
            ACTIVE_ENTRIES_READ.reset(self._entries_read_token)

        ACTIVE_ENTRIES_MODE.reset(self._mode_token)


class Params:
    def __init__(self, mode: str = "both", overwrite: bool = False, **params):
        self._params = _handle_special_types(params)
        _validate_mode(mode)
        self._mode = mode
        self._overwrite = overwrite

    def __enter__(self):
        _validate_mode_nesting(ACTIVE_PARAMS_MODE.get(), self._mode)
        self._mode_token = ACTIVE_PARAMS_MODE.set(self._mode)
        if self._mode in ("both", "write"):
            self._params_token = ACTIVE_PARAMS_WRITE.set(
                {**ACTIVE_PARAMS_WRITE.get(), **self._params},
            )
            self._nest_token = PARAMS_NEST_LEVEL.set(
                PARAMS_NEST_LEVEL.get() + 1,
            )

        if self._mode in ("both", "read"):
            self._params_read_token = ACTIVE_PARAMS_READ.set(
                {**ACTIVE_PARAMS_READ.get(), **self._params},
            )

        if self._overwrite:
            if self._mode == "read":
                raise Exception(f"Cannot overwrite logs in read mode.")

            logs = unify.get_logs(return_ids_only=True)
            if len(logs) > 0:
                unify.delete_logs(logs=logs)

    def __exit__(self, *args, **kwargs):
        ACTIVE_PARAMS_MODE.reset(self._mode_token)

        if self._mode in ("both", "write"):
            ACTIVE_PARAMS_WRITE.reset(self._params_token)
            PARAMS_NEST_LEVEL.reset(self._nest_token)
            if PARAMS_NEST_LEVEL.get() == 0:
                LOGGED.set({})

        if self._mode in ("both", "read"):
            ACTIVE_PARAMS_READ.reset(self._params_read_token)


class Experiment:
    def __init__(
        self,
        name: Optional[Union[str, int]] = None,
        overwrite: bool = False,
        mode: str = "both",
    ):
        _validate_mode(mode)
        self._mode = mode

        latest_exp_name = get_experiment_name(-1)
        if latest_exp_name is None:
            self._name = name if name is not None else "exp0"
            self._overwrite = overwrite
            return
        if isinstance(name, int):
            self._name = f"exp{get_experiment_name(name)}"
        elif name is None:
            self._name = f"exp{int(get_experiment_version(latest_exp_name)) + 1}"
        else:
            self._name = str(name)
        self._overwrite = overwrite

    def __enter__(self):
        _validate_mode_nesting(ACTIVE_PARAMS_MODE.get(), self._mode)
        self._mode_token = ACTIVE_PARAMS_MODE.set(self._mode)

        if self._mode in ("both", "write"):
            self._params_token_write = ACTIVE_PARAMS_WRITE.set(
                {**ACTIVE_PARAMS_WRITE.get(), **{"experiment": self._name}},
            )
            self._nest_token = PARAMS_NEST_LEVEL.set(
                PARAMS_NEST_LEVEL.get() + 1,
            )
        if self._mode in ("both", "read"):
            self._params_read_token = ACTIVE_PARAMS_READ.set(
                {**ACTIVE_PARAMS_READ.get(), **{"experiment": self._name}},
            )

        if self._overwrite:
            if self._mode == "read":
                raise Exception(f"Cannot overwrite logs in read mode.")

            logs = unify.get_logs(return_ids_only=True)
            if len(logs) > 0:
                unify.delete_logs(logs=logs)

    def __exit__(self, *args, **kwargs):
        ACTIVE_PARAMS_MODE.reset(self._mode_token)
        if self._mode in ("both", "write"):
            ACTIVE_PARAMS_WRITE.reset(self._params_token_write)
            PARAMS_NEST_LEVEL.reset(self._nest_token)
            if PARAMS_NEST_LEVEL.get() == 0:
                LOGGED.set({})
        if self._mode in ("both", "read"):
            ACTIVE_PARAMS_READ.reset(self._params_read_token)


# Tracing #
# --------#


class TraceTransformer(ast.NodeTransformer):
    def __init__(self, trace_dirs: list[str]):
        self.trace_dirs = trace_dirs
        self.trace_dir_ast = ast.List(
            elts=[ast.Constant(value=dir) for dir in trace_dirs],
            ctx=ast.Load(),
        )

    def visit_Call(self, node):
        self.generic_visit(node)
        return ast.Call(
            func=ast.Name(id="check_path_at_runtime", ctx=ast.Load()),
            args=[node.func, self.trace_dir_ast, *node.args],
            keywords=node.keywords,
        )


def _nested_add(a, b):
    if a is None and isinstance(b, dict):
        a = {k: None if isinstance(v, dict) else 0 for k, v in b.items()}
    elif b is None and isinstance(a, dict):
        b = {k: None if isinstance(v, dict) else 0 for k, v in a.items()}
    if isinstance(a, dict) and isinstance(b, dict):
        return {k: _nested_add(a[k], b[k]) for k in a if k in b}
    elif a is None and b is None:
        return None
    return a + b


def _create_span(fn, args, kwargs, span_type, name):
    exec_start_time = time.perf_counter()
    ts = datetime.now(timezone.utc).isoformat()
    if not SPAN.get():
        RUNNING_TIME.set(exec_start_time)
    signature = inspect.signature(fn)
    bound_args = signature.bind(*args, **kwargs)
    bound_args.apply_defaults()
    inputs = bound_args.arguments
    inputs = inputs["kw"] if span_type == "llm-cached" else inputs
    inputs = _make_json_serializable(inputs)
    try:
        lines, start_line = inspect.getsourcelines(fn)
        code = textwrap.dedent("".join(lines))
    except:
        lines, start_line = None, None
        try:
            code = textwrap.dedent(inspect.getsource(fn))
        except:
            code = None
    name_w_sub = name
    if name_w_sub is not None:
        for k, v in inputs.items():
            substr = "{" + k + "}"
            if substr in name_w_sub:
                name_w_sub = name_w_sub.replace(substr, str(v))
    new_span = {
        "id": str(uuid.uuid4()),
        "type": span_type,
        "parent_span_id": (None if not SPAN.get() else SPAN.get()["id"]),
        "span_name": fn.__name__ if name_w_sub is None else name_w_sub,
        "exec_time": None,
        "timestamp": ts,
        "offset": round(
            0.0 if not SPAN.get() else exec_start_time - RUNNING_TIME.get(),
            2,
        ),
        "llm_usage": None,
        "llm_usage_inc_cache": None,
        "code": f"```python\n{code}\n```",
        "code_fpath": inspect.getsourcefile(fn),
        "code_start_line": start_line,
        "inputs": inputs,
        "outputs": None,
        "errors": None,
        "child_spans": [],
        "completed": False,
    }
    if inspect.ismethod(fn) and hasattr(fn.__self__, "endpoint"):
        new_span["endpoint"] = fn.__self__.endpoint
    if not GLOBAL_SPAN.get():
        global_token = GLOBAL_SPAN.set(new_span)
        local_token = SPAN.set(GLOBAL_SPAN.get())
    else:
        global_token = None
        SPAN.get()["child_spans"].append(new_span)
        local_token = SPAN.set(new_span)
    _get_trace_logger().update_trace(ACTIVE_LOG.get(), copy.deepcopy(GLOBAL_SPAN.get()))
    return new_span, exec_start_time, local_token, global_token


def _finalize_span(
    new_span,
    local_token,
    outputs,
    exec_time,
    prune_empty,
    global_token,
):
    SPAN.get()["exec_time"] = exec_time
    SPAN.get()["outputs"] = outputs
    SPAN.get()["completed"] = True
    if SPAN.get()["type"] == "llm" and outputs is not None:
        SPAN.get()["llm_usage"] = outputs["usage"]
    if SPAN.get()["type"] in ("llm", "llm-cached") and outputs is not None:
        SPAN.get()["llm_usage_inc_cache"] = outputs["usage"]
    trace = SPAN.get()
    if prune_empty:
        trace = _prune_dict(trace)
        SPAN.set(trace)
        if global_token:
            GLOBAL_SPAN.set(trace)
    SPAN.reset(local_token)
    if local_token.old_value is not local_token.MISSING:
        SPAN.get()["llm_usage"] = _nested_add(
            SPAN.get()["llm_usage"],
            new_span["llm_usage"],
        )
        SPAN.get()["llm_usage_inc_cache"] = _nested_add(
            SPAN.get()["llm_usage_inc_cache"],
            new_span["llm_usage_inc_cache"],
        )
    _get_trace_logger().update_trace(ACTIVE_LOG.get(), copy.deepcopy(GLOBAL_SPAN.get()))
    if global_token:
        GLOBAL_SPAN.reset(global_token)


def _trace_class(cls, prune_empty, span_type, name, filter):
    for member_name, value in inspect.getmembers(cls, predicate=inspect.isfunction):
        if member_name.startswith("__") and member_name.endswith("__"):
            continue
        if filter is not None and not filter(value):
            continue
        _name = f"{name if name is not None else cls.__name__}.{member_name}"
        setattr(
            cls,
            member_name,
            traced(value, prune_empty=prune_empty, span_type=span_type, name=_name),
        )
    return cls


def _trace_module(module, prune_empty, span_type, name, filter):
    _obj_filter = lambda obj: inspect.isfunction(obj) or inspect.isclass(obj)
    for member_name, value in inspect.getmembers(module, predicate=_obj_filter):
        if member_name.startswith("__") and member_name.endswith("__"):
            continue
        if filter is not None and not filter(value):
            continue
        _name = f"{name if name is not None else module.__name__}.{member_name}"
        setattr(
            module,
            member_name,
            traced(value, prune_empty=prune_empty, span_type=span_type, name=_name),
        )
    return module


def _transform_function(fn, prune_empty, span_type, trace_dirs):
    def check_path_at_runtime(fn, target_dirs, *args, **kwargs):
        if (
            inspect.isbuiltin(fn)
            or not os.path.dirname(inspect.getsourcefile(fn)) in target_dirs
        ):
            return fn(*args, **kwargs)

        try:
            return traced(
                prune_empty=prune_empty,
                span_type=span_type,
                trace_dirs=target_dirs,
            )(fn)(*args, **kwargs)
        except Exception as e:
            raise e

    for i, dir_path in enumerate(trace_dirs):
        if not os.path.isabs(dir_path):
            dir_path = os.path.normpath(
                os.path.join(os.path.dirname(inspect.getsourcefile(fn)), dir_path),
            )
            trace_dirs[i] = dir_path

    source = textwrap.dedent(inspect.getsource(fn))
    source_lines = source.split("\n")
    if source_lines[0].strip().startswith("@"):
        source = "\n".join(source_lines[1:])

    tree = ast.parse(source)
    transformer = TraceTransformer(trace_dirs)
    tree = transformer.visit(tree)
    ast.fix_missing_locations(tree)
    code = compile(tree, filename=inspect.getsourcefile(fn), mode="exec")
    module = inspect.getmodule(fn)

    func_globals = module.__dict__.copy() if module else globals().copy()
    func_globals["check_path_at_runtime"] = check_path_at_runtime

    exec(code, func_globals)
    old_fn = fn
    fn = func_globals[fn.__name__]
    functools.update_wrapper(fn, old_fn)
    return fn


def _trace_function(
    fn,
    prune_empty,
    span_type,
    name,
    trace_contexts,
    trace_dirs,
    filter,
):
    if trace_dirs is not None:
        fn = _transform_function(fn, prune_empty, span_type, trace_dirs)

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        log_token = None if ACTIVE_LOG.get() else ACTIVE_LOG.set([unify.log()])
        new_span, exec_start_time, local_token, global_token = _create_span(
            fn,
            args,
            kwargs,
            span_type,
            name,
        )
        result = None
        try:
            result = fn(*args, **kwargs)
            return result
        except Exception as e:
            new_span["errors"] = traceback.format_exc()
            raise e
        finally:
            outputs = _make_json_serializable(result) if result is not None else None
            exec_time = time.perf_counter() - exec_start_time
            _finalize_span(
                new_span,
                local_token,
                outputs,
                exec_time,
                prune_empty,
                global_token,
            )
            if log_token:
                ACTIVE_LOG.set([])

    async def async_wrapped(*args, **kwargs):
        log_token = None if ACTIVE_LOG.get() else ACTIVE_LOG.set([unify.log()])
        new_span, exec_start_time, local_token, global_token = _create_span(
            fn,
            args,
            kwargs,
            span_type,
            name,
        )
        result = None
        try:
            result = await fn(*args, **kwargs)
            return result
        except Exception as e:
            new_span["errors"] = traceback.format_exc()
            raise e
        finally:
            outputs = _make_json_serializable(result) if result is not None else None
            exec_time = time.perf_counter() - exec_start_time
            _finalize_span(
                new_span,
                local_token,
                outputs,
                exec_time,
                prune_empty,
                global_token,
            )
            if log_token:
                ACTIVE_LOG.set([])

    return wrapped if not inspect.iscoroutinefunction(fn) else async_wrapped


def traced(
    fn: callable = None,
    *,
    prune_empty: bool = True,
    span_type: str = "function",
    name: Optional[str] = None,
    trace_contexts: Optional[List[str]] = None,
    trace_dirs: Optional[List[str]] = None,
    filter: Optional[Callable[[callable], bool]] = None,
):
    _initialize_trace_logger()

    if fn is None:
        return lambda f: traced(
            f,
            prune_empty=prune_empty,
            span_type=span_type,
            name=name,
            trace_contexts=trace_contexts,
            trace_dirs=trace_dirs,
        )

    if inspect.isclass(fn):
        return _trace_class(fn, prune_empty, span_type, name, filter)

    if inspect.ismodule(fn):
        return _trace_module(fn, prune_empty, span_type, name, filter)

    return _trace_function(
        fn,
        prune_empty,
        span_type,
        name,
        trace_contexts,
        trace_dirs,
        filter,
    )


class LogTransformer(ast.NodeTransformer):
    def __init__(self):
        super().__init__()
        self.param_names = []
        self.assigned_names = set()
        self._in_function = False

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self._in_function = True
        # Collect non-underscore params
        self.param_names = [
            arg.arg for arg in node.args.args if not arg.arg.startswith("_")
        ]
        # Collect non-underscore kwonlyargs
        self.param_names += [
            arg.arg for arg in node.args.kwonlyargs if not arg.arg.startswith("_")
        ]

        # Add **kwargs parameter if not already present
        if not node.args.kwarg:
            node.args.kwarg = ast.arg(arg="kwargs")

        # TODO: this is a hack to ensure that the function always returns something
        if not isinstance(node.body[-1], ast.Return):
            node.body.append(ast.Return(value=ast.Constant(value=None)))

        node = self.generic_visit(node)
        self._in_function = False
        self.param_names = []
        self.assigned_names = set()
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self._in_function = True
        self.param_names = [
            arg.arg for arg in node.args.args if not arg.arg.startswith("_")
        ]
        node = self.generic_visit(node)
        self._in_function = False
        self.param_names = []
        self.assigned_names = set()
        return node

    def visit_Assign(self, node: ast.Assign):
        if self._in_function:
            for target in node.targets:
                if isinstance(target, ast.Name) and not target.id.startswith("_"):
                    # Remove from param_names if it's a reassigned parameter
                    if target.id in self.param_names:
                        self.param_names.remove(target.id)
                    self.assigned_names.add(target.id)
        return node

    def visit_Return(self, node: ast.Return):
        if not self._in_function:
            return node

        log_keywords = []
        # Add regular parameters (that weren't reassigned)
        for p in self.param_names:
            log_keywords.append(
                ast.keyword(arg=p, value=ast.Name(id=p, ctx=ast.Load())),
            )

        # Add assigned variables (including reassigned parameters)
        for var_name in sorted(self.assigned_names):
            log_keywords.append(
                ast.keyword(arg=var_name, value=ast.Name(id=var_name, ctx=ast.Load())),
            )

        # Add filtered kwargs (non-underscore keys)
        kwargs_dict = ast.DictComp(
            key=ast.Name(id="k", ctx=ast.Load()),
            value=ast.Name(id="v", ctx=ast.Load()),
            generators=[
                ast.comprehension(
                    target=ast.Tuple(
                        elts=[
                            ast.Name(id="k", ctx=ast.Store()),
                            ast.Name(id="v", ctx=ast.Store()),
                        ],
                        ctx=ast.Store(),
                    ),
                    iter=ast.Call(
                        func=ast.Attribute(
                            value=ast.Name(id="kwargs", ctx=ast.Load()),
                            attr="items",
                            ctx=ast.Load(),
                        ),
                        args=[],
                        keywords=[],
                    ),
                    ifs=[
                        ast.UnaryOp(
                            op=ast.Not(),
                            operand=ast.Call(
                                func=ast.Attribute(
                                    value=ast.Name(id="k", ctx=ast.Load()),
                                    attr="startswith",
                                    ctx=ast.Load(),
                                ),
                                args=[ast.Constant(value="_")],
                                keywords=[],
                            ),
                        ),
                    ],
                    is_async=0,
                ),
            ],
        )
        log_keywords.append(ast.keyword(arg=None, value=kwargs_dict))

        return_value = (
            node.value if node.value is not None else ast.Constant(value=None)
        )

        log_call = ast.Expr(
            value=ast.Call(
                func=ast.Name(id="unify_log", ctx=ast.Load()),
                args=[],
                keywords=log_keywords,
            ),
        )

        return [log_call, ast.Return(value=return_value)]


def log_decorator(func):
    """
    Decorator that rewrites the function's AST so that it logs non-underscore
    parameters, and assigned variables.
    """
    # 1) Parse the source to an AST
    source = textwrap.dedent(inspect.getsource(func))

    # Remove the decorator line if present
    source_lines = source.split("\n")
    if source_lines[0].strip().startswith("@"):
        source = "\n".join(source_lines[1:])

    mod = ast.parse(source)

    # 2) Transform the AST
    transformer = LogTransformer()
    mod = transformer.visit(mod)
    ast.fix_missing_locations(mod)

    # 3) Compile the new AST
    code = compile(mod, filename="<ast>", mode="exec")

    # 4) Get the current module's globals
    module = inspect.getmodule(func)
    func_globals = module.__dict__.copy() if module else globals().copy()
    func_globals["unify_log"] = unify_log

    # 5) Execute the compiled module code in that namespace
    exec(code, func_globals)
    trans = func_globals[func.__name__]

    # 6 ) Add logging context
    def transformed_func(*args, **kwargs):
        with unify.Log():
            return trans(*args, **kwargs)

    # Copy necessary attributes
    transformed_func.__name__ = func.__name__
    transformed_func.__doc__ = func.__doc__
    transformed_func.__module__ = func.__module__
    transformed_func.__annotations__ = func.__annotations__

    # Copy closure and cell variables if they exist
    if func.__closure__:
        transformed_func.__closure__ = func.__closure__

    return transformed_func

```
