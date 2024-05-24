from numba import deferred_type, types, njit
from numba.experimental import jitclass

from numba.extending import type_callable, lower_builtin
import operator

@type_callable(operator.eq)
def eq(context):
    """Type the eq operator for jitclasses"""
    def typer(arg1, arg2):
        if isinstance(arg1, types.misc.ClassInstanceType) and isinstance(arg2, types.misc.ClassInstanceType):
            return types.bool_
    return typer

@lower_builtin(operator.eq, types.misc.ClassInstanceType, types.misc.ClassInstanceType)
def lower_node_eq(context, builder, sig, args):
    """Implement comparison operation for jitclasses by comparing their data pointers"""
    retty = sig.return_type
    # see numba.jitclass.base.InstanceDataModel
    get_dataptr = lambda val: builder.extract_value(val, 1)
    ptr1 = get_dataptr(args[0])
    ptr2 = get_dataptr(args[1])
    return builder.icmp_unsigned('==', ptr1, ptr2)

node_type = deferred_type()
node_spec = [("left", node_type),
             ("right", node_type),
             ("children", types.List(node_type))
            ]

@njit()
def get_empty_child():
    pass

@jitclass(node_spec)
class Node(object):
    def __init__(self):
        self.left = self
        self.right = self
        self.children = []
        
    def get_self(self):
        return self
    
    def equals(self, other):
        return other == self

node_type.define(Node.class_type.instance_type)

n1 = Node()
n2 = Node()
n1.equals(n1.left)
n1.equals(n1.right)
n1.left.equals(n1.right)
n1.equals(n2)