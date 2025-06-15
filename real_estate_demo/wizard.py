from typing import Callable, Literal, Union
from textwrap import dedent

from pydantic import BaseModel, Field, create_model

class Next(BaseModel):
    next: bool = Literal[True]

class Back(BaseModel):
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
        str_options = "\n".join([f"( ) {o}" if self.value != o else f"(x) {o}" for o in self.options])
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
    def __init__(self, id: str, title: str, instructions: str, fields: list, next: str|Callable):
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
        

        if isinstance(next, str):
            self.next = lambda ctx: next
        elif isinstance(next, dict):
            # assume there is one field in the screen
            self.next = lambda ctx: next[ctx[self.fields[0].id]]
        else:
            self.next = next

        self.is_submitted = False

    def set_up_action_model(self):
        # dynamically create the pydantic model representing 
        # the available actions in the current screen

        fields_action_models = []
        for field in self.fields:
            if isinstance(field, InputField):
                field_action_model = create_model(
                    f'Fill{"".join([w.title() for w in field.label.split(" ")])}',
                    value=(str, Field(..., description=field.description))
                )
                
            elif isinstance(field, RadioField):
                field_action_model = create_model(
                    f'Select{field.label.title()}',
                    value=(Literal[*field.options], Field(..., description=field.description))
                )
            self.action_to_field[field_action_model] = field.id
            fields_action_models.append(field_action_model)
        # fields_action_models.append(Next)
        self.action_model = Union[*fields_action_models]
    
    def play_actions(self, list_of_actions):
        for action in list_of_actions:
            if isinstance(action, Next):
                self.is_submitted = True
            else:
                action_cls = action.__class__
                field_id = self.action_to_field[action_cls]
                self.data[field_id] = action.value


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
        
        # this will hold all the data collected so far
        self.ctx = self.current_node.data
        self.path = [self.current_node]

    def play_actions(self, list_of_actions: ...):
        self.current_node.play_actions(list_of_actions)
        
        # update ctx
        self.ctx |= self.current_node.data

        # check if the current screen has been submitted
        if self.current_node.is_submitted:
            next_screen_id = self.current_node.next(self.ctx)
            # print(next_screen_id)
            self.current_node = list(filter(lambda s: s.id==next_screen_id, self.screens))[0]
            self.ctx |= self.current_node.data
            self.path.append(self.current_node)
    
    def current_action_model(self):
        # we should check if node is terminal (or has no begning) or not actually
        # before adding both Next and Back

        # TODO: also add a gotonode action dynamically
        GoToNode = ...
        return Union[self.current_node.action_model | Next | Back]
    
    def render(self):
        return f"""
Current Path: {" > ".join([n.title for n in self.path])}

{self.current_node.render()}"""
    
    
