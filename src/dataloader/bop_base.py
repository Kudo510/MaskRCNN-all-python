'''
A class should contain ll the called-in function like below. 
Have perpoerties, static, class methods
Write type of input vriables used in the class.
Use protected variables and private variables. as _ for preotected vars and __ for private vars. these cannot be accessed outside the class.- this is the encapsulation of OOP. 
    Protected vars are used when we want to use the var in the class and its child class only. not outside
    private vars are used when we want to use the var in the class only.

Use abstract class to define the class and methods.. the abstraction of OOP
Write normal method a in the class A and class B inherit class A, then in B rewrite content of method a. Then call B.a will be updated - Polymorphism of OOP
Inheritance of OOP u know alreday just inherit

Static method used like an utility function for the class- 10* rule put a function in a class when we are sure that we just use this function for this class only. Ohter wise just put in in util.py instead

Class method used when u want to change the state, the values of the self.vars of the class- just involve the calss itselfv.- it has cls instead of self in the function

property, setter, delete methods are used to get, set and delete the values of the class vars. they are used often for the protected and private varables/attributes. Cos they are not accessed directly out side class so to show them use getter, to change them use setter and to delte them use delete method(delter much less used).- 10* hay  - often seen setter, properties for protected vars ( of couse also works for private vars)

'''

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

class BOPBaseDataLoader:
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    data: List[Dict[str, Any]]
    data_len: int

    def __init__(self, input_data: Dict[str, Any], output_data: Dict[str, Any]):
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[Dict[str, Any]]:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass
    
    @abstractmethod
    def __call__(self, *args, **kwds):
        pass

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def __iter__(self):
        pass

    @abstractmethod
    def __next__(self):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @staticmethod
    @abstractmethod
    def a_static_method():
        pass

    @property
    @abstractmethod
    def a_property_method(self):
        pass
    
    @classmethod
    @abstractmethod
    def a_class_method(cls):
        pass
    