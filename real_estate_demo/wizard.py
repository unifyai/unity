import re
from typing import Callable, Literal, Union, Optional
from textwrap import dedent

from pydantic import BaseModel, Field, create_model

class GoNext(BaseModel):
    "advance to the next node"
    next: Literal[True] = True

class GoBack(BaseModel):
    "go back to the previous node"
    back: Literal[True] = True

class EndSession(BaseModel):
    "conclude the session and end the call, can only be used in a terminal node, using EndSession will end the call and you won't be able to listen to the user anymore."
    end_session: Literal[True] = True
    closing_message: str

class PromptUser(BaseModel):
    "prompt user (ask a question, provide clarification)"
    prompt: str

# class UpdateUser(BaseModel):
#     "update user one what you are doing at the moment, check the <conversation_history> and give a quick update if you have not responded for while"
#     update: str = Field(..., description="short update")

class BaseGoToNode(BaseModel):
    pass

class BaseDataFieldAction(BaseModel):
    pass

class InputField:
    def __init__(self, id: str, label: str=None, value=None, required: bool = True):
        self.id = id
        self.label = label if label is not None else self.id
        self.original_value = value.copy() if value is not None else None
        self.value = value
        self.required = required
    
    def set_value(self, value):
        self.value = value
    
    def render(self):
        return dedent(f"""
{self.label} (Input Field)
{'[...]' if self.value is None else '['+self.value+']'}""").strip()


class RadioField:
    def __init__(self, id: str, label: str, options: list[str], value=None, required: bool=True):
        self.id = id
        self.label = label
        self.options = options
        self.original_value = value.copy() if value is not None else None
        self.value = value
        self.required = required
    
    def set_value(self, value):
        self.value = value
    
    def render(self):
        str_options = "\n".join([f"( ) {o}" if self.value != o else f"(x) {o} <- currently selected option" for o in self.options])
        return dedent(f"""
{self.label} (Radio Field)
{str_options}""").strip()
    
class CheckBoxField:
    def __init__(self, id: str, label: str, options: list[str], value=None, required: bool=True):
        self.id = id
        self.label = label
        self.options = options
        self.original_value = value.copy() if value is not None else None
        self.value = value
        self.required = required
    
    def set_value(self, value):
        self.value = value
    
    def render(self):
        str_options = "\n".join([f"[ ] {o}" if o not in self.value else f"[x] {o} <- checked" for o in self.options])
        return dedent(f"""
{self.label} (CheckBox Field)
{str_options}""").strip()

class Node:
    def __init__(self, id: str, title: str, instructions: str, fields: list, next: str|Callable=None):
        self.id = id
        self.title = title
        self.instructions = instructions
        self.fields = fields

        # if self.fields:
        #     self.instructions += "\nConsult the <event_stream> if any of the fields can be filled by already provided information."


        # bind input fields to data values
        self.data = {}
        for field in self.fields:
            self.data[field.id] = field.value
        
        # action model dict
        self.action_to_field = {}
        self.set_up_action_model()
        
        self.is_terminal = False
        if isinstance(next, str):
            self.next = lambda ctx: next
        elif isinstance(next, dict):
            # assume there is one field in the screen
            self.next = lambda ctx: next[ctx[self.fields[0].id]]
        elif next is None:
            self.next = None
            self.is_terminal = True
        else:
            self.next = next
        
        self.can_advance = all([self.data.get(f.id) is not None for f in self.fields if f.required])

    def set_up_action_model(self):
        # dynamically create the pydantic model representing 
        # the available actions in the current screen

        fields_action_models = []
        for field in self.fields:
            if isinstance(field, InputField):
                pascal_action_name = f'Fill{"".join([w.title() for w in field.label.split(" ")])}'
                snake_case_action_name = f'fill_{"_".join([w.lower() for w in field.label.split(" ")])}'
                pascal_action_name = re.sub(r"[\?\!\/\\]", "", pascal_action_name)
                snake_case_action_name = re.sub(r"[\?\!\/\\]", "", snake_case_action_name)
                docs = f"Fills input field '{field.label}' and reflects it to the agent script current node." + (f" Current value is `{self.data.get(field.id)}`" if self.data.get(field.id) is not None else "")
                field_action_model = create_model(
                    pascal_action_name,
                    field_label=(Literal[field.label], field.label),
                    value=(str, Field(..., description="value to input")),
                    __doc__=docs
                )
                
            elif isinstance(field, RadioField):
                pascal_action_name = f'Select{"".join([w.title() for w in field.label.split(" ")])}'
                snake_case_action_name = f'select_{"_".join([w.lower() for w in field.label.split(" ")])}'
                pascal_action_name = re.sub(r"[\?\!\/\\]", "", pascal_action_name)
                snake_case_action_name = re.sub(r"[\?\!\/\\]", "", snake_case_action_name)
                docs = f"Selects an option for radio field '{field.label}' and reflects it to the agent script current node." + (f" Current value is `{self.data.get(field.id)}`" if self.data.get(field.id) is not None else "")
                field_action_model = create_model(
                    pascal_action_name,
                    field_label=(Literal[field.label], field.label),
                    value=(Literal[*field.options], Field(..., description="option to select")),
                    __doc__= docs
                )
            
            elif isinstance(field, CheckBoxField):
                pascal_action_name = f'Check{"".join([w.title() for w in field.label.split(" ")])}'
                snake_case_action_name = f'check_{"_".join([w.lower() for w in field.label.split(" ")])}'
                pascal_action_name = re.sub(r"[\?\!\/\\]", "", pascal_action_name)
                snake_case_action_name = re.sub(r"[\?\!\/\\]", "", snake_case_action_name)
                docs = f"Checks options for checkbox field '{field.label}' and reflects it to the agent script current node." +  (f" Current value is `{self.data.get(field.id)}`" if self.data.get(field.id) is not None else "")
                field_action_model = create_model(
                    pascal_action_name,
                    field_label=(Literal[field.label], field.label),
                    value=(list[Literal[*field.options]], Field(..., description="options to check")),
                    __doc__=docs
                )

            self.action_to_field[field_action_model] = field.id
            fields_action_models.append(field_action_model)
        
        # self.action_model = create_model(
        #     "ActionModel",
        #     **{k:(Optional[v], Field(default=None, description=d)) for k, d, v in fields_action_models}
        # )

        self.action_model = Union[*fields_action_models]
        
    
    def play_actions(self, action: BaseModel):
        action_cls = action.__class__
        field_id = self.action_to_field[action_cls]
        self.data[field_id] = action.value
        self.can_advance = all([self.data.get(f.id) is not None for f in self.fields if f.required])
        
        # refresh action model after each action taken
        self.set_up_action_model()
        print("can advance", self.can_advance)
        


    def render(self, ctx={}):
        body = ""
        for field in self.fields:
            field.set_value(self.data[field.id])
            body += field.render()
            body += '\n'
        return dedent(f"""
Node: {self.title}
Is Terminal Node?: {self.is_terminal}
Instructions: {self.instructions.format(**ctx)}
---

{body}""").strip()
    
    def reset(self):
        self.data = {}
        for field in self.fields:
            field.set_value(field.original_value)
            self.data[field.id] = field.value
        self.can_advance = all([self.data.get(f.id) is not None for f in self.fields if f.required])
        # action model dict
        self.action_to_field = {}
        self.set_up_action_model()


class Flow:
    def __init__(self, screens: list[Node], start: str=None):
        self.screens = screens
        for s in screens:
            s.reset()
        
        self.current_node: Node = self.screens[0] if start is None else list(filter(lambda s: s.id==start, self.screens))[0]
        self.root_node = self.current_node
        
        # this will hold all the data collected so far
        self.ctx = self.current_node.data
        self.path = [self.current_node]

        self.flow_done = False

    def play_actions(self, actions):
        if self.flow_done: return
        for action in actions:
            if isinstance(action, (PromptUser)):
                continue
            if isinstance(action, GoBack):
                self.path.pop()
                self.current_node = self.path[-1]
                return
            elif isinstance(action, GoNext):
                if self.current_node.can_advance:
                    next_screen_id = self.current_node.next(self.ctx)
                    # print(next_screen_id)
                    self.current_node = list(filter(lambda s: s.id==next_screen_id, self.screens))[0]
                    self.ctx |= self.current_node.data
                    self.path.append(self.current_node)
                    return
            elif isinstance(action, BaseGoToNode):
                self.current_node = list(filter(lambda s: s.id==action.node_id, self.screens))[0]
                return
            elif isinstance(action, EndSession):
                self.flow_done = True
                return
            else:
                self.current_node.play_actions(action)
                # update ctx
                self.ctx |= self.current_node.data
                # return
        
    def current_action_model(self) -> BaseModel:
        # we should check if node is terminal (or has no begning) or not actually
        # before adding both Next and Back
        extra_actions = []
        # TODO: also add a gotonode action dynamically
        GoToNode = create_model(
            "GoToNode",
            node_id = (Literal[*[n.id for n in self.path]], Field(..., description="Node ID to go to")),
            __doc__="Goes to the chosen node that you have already visited in your path. Useful when you need to go back to a specific node to modify data or start-over.",
            __base__=BaseGoToNode
        )
        # if self.current_node != self.root_node:
        #     extra_actions.append(GoBack)
        if not self.current_node.is_terminal and self.current_node.can_advance:
            extra_actions.append(GoNext)
        if len(self.path) > 2:
            extra_actions.append(GoToNode)
        if self.current_node.is_terminal:
            extra_actions.append(EndSession)
        # return create_model(
        #     "ActionModel",
        #     __base__=self.current_node.action_model,
        #     **{k:(Optional[v], Field(default=None, description=d)) for k, d, v in extra_actions}
        # )
        DataFieldAction = create_model(
            "DataFieldAction",
            update= (str, Field(default=None, description="short ~3-5 word update for the user, avoid redundant updates.")),
            fields_actions = (list[Union[self.current_node.action_model]], Field(default=..., description="data field action to take")),
            __base__=BaseDataFieldAction
        )
        return Union[DataFieldAction, *extra_actions]
    
    def render(self):
        return f"""
Current Path: {" > ".join([n.title for n in self.path])}

{self.current_node.render(self.ctx)}""".strip()
    
    
