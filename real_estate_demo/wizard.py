import re
from typing import Callable, Literal, Union, Optional
from textwrap import dedent

from pydantic import BaseModel, Field, create_model

class GoNext(BaseModel):
    next: bool = Literal[True]

class GoBack(BaseModel):
    back: bool = Literal[True]


class InputField:
    def __init__(self, id: str, label: str=None, description: str=None, value=None):
        self.id = id
        self.description = description
        self.label = label if label is not None else self.id
        self.original_value = value.copy() if value is not None else None
        self.value = value
    
    def set_value(self, value):
        self.value = value
    
    def render(self):
        return dedent(f"""
{self.label} (Input Field)
{'[...]' if self.value is None else '['+self.value+']'}""").strip()


class RadioField:
    def __init__(self, id: str, label: str, description: str, options: list[str], value=None):
        self.id = id
        self.label = label
        self.description = description
        self.options = options
        self.original_value = value.copy() if value is not None else None
        self.value = value
    
    def set_value(self, value):
        self.value = value
    
    def render(self):
        str_options = "\n".join([f"( ) {o}" if self.value != o else f"(x) {o} <- currently selected" for o in self.options])
        return dedent(f"""
{self.label} (Radio Field)
{str_options}""").strip()


# class InformationField:
#     def __init__(self, id: str, label: str, information: str, description: str, value=None):
#         self.id = id
#         self.label = label
#         self.information = information
#         self.description = description


#     def set_value(self, value):
#         self.value = value

class Node:
    def __init__(self, id: str, title: str, instructions: str, fields: list, next: str|Callable=None):
        self.id = id
        self.title = title
        self.instructions = instructions
        self.fields = fields

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

        self.is_submitted = False

    def set_up_action_model(self):
        # dynamically create the pydantic model representing 
        # the available actions in the current screen

        fields_action_models = []
        for field in self.fields:
            if isinstance(field, InputField):
                pascal_action_name = f'Fill{"".join([w.title() for w in field.label.split(" ")])}'
                snake_case_action_name = f'fill_{"_".join([w.lower() for w in field.label.split(" ")])}'
                pascal_action_name = re.sub(r"[\?\!]", "", pascal_action_name)
                snake_case_action_name = re.sub(r"[\?\!]", "", snake_case_action_name)
                field_action_model = create_model(
                    pascal_action_name,
                    value=(str, Field(..., description="value to input"))
                )
                
            elif isinstance(field, RadioField):
                pascal_action_name = f'Select{"".join([w.title() for w in field.label.split(" ")])}'
                snake_case_action_name = f'select_{"_".join([w.lower() for w in field.label.split(" ")])}'
                pascal_action_name = re.sub(r"[\?\!]", "", pascal_action_name)
                snake_case_action_name = re.sub(r"[\?\!]", "", snake_case_action_name)
                field_action_model = create_model(
                    pascal_action_name,
                    value=(Literal[*field.options], Field(..., description="option to select"))
                )
            self.action_to_field[field_action_model] = field.id
            fields_action_models.append((snake_case_action_name, field, field_action_model))
        # fields_action_models.append(Next)
        self.action_model = create_model(
            "ActionModel",
            **{k:(Optional[v], Field(..., description=f.description)) for k, f, v in fields_action_models}
        )
    
    def play_actions(self, action: BaseModel):
        for k, v in action:
            if v is not None:
                if isinstance(v, GoNext):
                    self.is_submitted = True
                else:
                    action_cls = v.__class__
                    field_id = self.action_to_field[action_cls]
                    self.data[field_id] = v.value


    def render(self):
        body = ""
        for field in self.fields:
            field.set_value(self.data[field.id])
            body += field.render()
            body += '\n'
        return dedent(f"""
Node: {self.title}
Instructions: {self.instructions}
---

{body}""").strip()
    
    def reset(self):
        self.data = {}
        for field in self.fields:
            field.set_value(field.original_value)
            self.data[field.id] = field.value
        self.is_submitted = False
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

    def play_actions(self, action):
        for l, a in action:
            if a is not None:
                if isinstance(a, GoBack):
                    self.path.pop()
                    self.current_node = self.path[-1]
                    self.current_node.is_submitted = False
                    print(self.current_node.title)
                    return
        
        self.current_node.play_actions(action)
        # update ctx
        self.ctx |= self.current_node.data

        # check if the current screen has been submitted
        if self.current_node.is_submitted:
            next_screen_id = self.current_node.next(self.ctx)
            # print(next_screen_id)
            self.current_node = list(filter(lambda s: s.id==next_screen_id, self.screens))[0]
            self.ctx |= self.current_node.data
            self.path.append(self.current_node)
    
    def current_action_model(self) -> BaseModel:
        # we should check if node is terminal (or has no begning) or not actually
        # before adding both Next and Back
        extra_actions = []
        # TODO: also add a gotonode action dynamically
        GoToNode = ...
        if self.current_node != self.root_node:
            extra_actions.append(("go_back", "go to previous node", GoBack))
        if not self.current_node.is_terminal:
            extra_actions.append(("go_next", "go to the next node", GoNext))
        return create_model(
            "ActionModel",
            __base__=self.current_node.action_model,
            **{k:(Optional[v], Field(..., description=d)) for k, d, v in extra_actions}
        )
    
    def render(self):
        return f"""
Current Path: {" > ".join([n.title for n in self.path])}

{self.current_node.render()}""".strip()
    
    
