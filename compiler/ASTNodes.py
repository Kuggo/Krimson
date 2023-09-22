from copy import copy, deepcopy
from Instructions import *


# helper functions
def tuple_unpacking(ctx: 'Context', node: 'TupleLiteral', looking_for, error) -> list['VarDefineNode']:
    nodes = []
    for value in node.value:
        if isinstance(value, TupleLiteral):
            nodes += tuple_unpacking(ctx, value, looking_for, error)
        elif isinstance(value, looking_for):
            nodes.append(VarDefineNode(value, value.type))
        else:
            ctx.error(error, value)
    return nodes

def tuple_type_unpacking(ctx: 'Context', node: 'TupleType', looking_for, error) -> list['Type']:
    nodes = []
    for t in node.types:
        if isinstance(t, TupleType):
            nodes += tuple_type_unpacking(ctx, t, looking_for, error)
        elif isinstance(t, looking_for):
            nodes.append(t)
        else:
            ctx.error(error, t)
    return nodes


# Errors

class TypeError(Enum):
    no_attribute = 'type "{}" has no attribute "{}"'
    param_def_expected = 'Parameter definition expected'
    undefined_function = 'Undefined function for the given args'
    shadow_same_scope = 'variable "{}" was already defined in the current scope. It cannot be shadowed'
    exit_not_in_body = 'exit keyword found outside the specified body depth'
    skip_not_in_loop = 'skip keyword found outside the specified body depth'
    pos_int_expected = 'Positive integer expected'
    wrong_type = 'Expected type "{}" but got "{}"'
    ambiguous_type = 'Ambiguous name. Multiple variables with possible types "{}" could fit. Use `<name>: <type>` to specify which variable to use'
    cannot_infer_type = 'Cannot infer type of variable "{}". Possible types: {}. \nUse `<name>: <type>` in any usage of the variable to specify its type'
    unk_var = 'No associated type for name of variable "{}". \nUse `<name>: <type>` in any usage of the variable to specify its type'
    wrong_func_type = 'Expected function with input arguments of type "{}" but got "{}"'

    # Compile_time detected runtime errors
    out_of_bounds = 'Index is out of bounds'
    no_element = 'No element with such key'


# Literals directly supported by compiler

class TupleLiteral(Literal):
    def __init__(self, values: list['Node']):
        assert len(values) > 0
        super().__init__(values, set(), values[-1].location - values[0].location)
        return

    def update_literal(self, context, parent) -> None:
        for value in self.value:
            value.update(context, parent)
        return

    def type_check_literal(self, context: 'Context', expected_types=None) -> None:
        types = []
        for i, val in enumerate(self.value):
            self.value[i] = val.type_check(self.possible_types_at(expected_types, i))
            if self.value[i] is not None:
                types.append(self.value[i].type)

        self.type = TupleType(types)
        return

    @staticmethod
    def possible_types_at(types: Optional[set[Type]], index: int) -> Optional[set[Type]]:
        if types is None:
            return None

        possible_types = set()
        for t in types:
            if isinstance(t, TupleType) and len(t.types) > index:
                possible_types.add(t.types[index])

        return possible_types

    def __str__(self):
        return f'({", ".join([str(v) for v in self.value])})'

    def __repr__(self):
        return f'<({", ".join([f"{v.__repr__()}" for v in self.value])})>'


class ArrayLiteral(Literal):
    def __init__(self, values: list):
        assert len(values) > 0
        super().__init__(values, set(), values[-1].location - values[0].location)
        return

    def update_literal(self, context, parent) -> None:
        for value in self.value:
            value.update(context, parent)
        return

    def type_check_literal(self, context: 'Context', expected_types=None) -> None:
        t = None
        for i, val in enumerate(self.value):
            val = val.type_check(self.possible_types(expected_types))
            if val is None:
                continue
            self.value[i] = val

            if t is None:
                t = self.value[i].type

            if self.value[i].type != t:
                context.error(TypeError.wrong_type, t, self.value[i].type.name_tok.value, t)
                continue

        self.type = ArrayType(t)
        return

    @staticmethod
    def possible_types(types: Optional[set[Type]]) -> Optional[set[Type]]:
        if types is None:
            return None

        possible_types = set()
        for t in types:
            if isinstance(t, ArrayType):
                possible_types.add(t.arr_type)

        return possible_types

    def __str__(self):
        return f'[{", ".join([str(v) for v in self.value])}]'

    def __repr__(self):
        return f'<[{", ".join([f"{v.__repr__()}" for v in self.value])}]>'


class FunctionLiteral(Literal):
    def __init__(self, in_param: 'Node', out_param: 'Node', body: 'Node'):
        super().__init__(None, set())
        self.in_param: 'Node' = in_param
        self.out_param: 'Node' = out_param

        self.in_param_list: list[NameDefineNode] = []
        self.out_param_list: list[NameDefineNode] = []
        self.body: Node = body
        return

    def gen_param_list(self, ctx: 'Context', param: 'Node') -> list['NameDefineNode']:
        if isinstance(param, VarDefineNode):
            return [param]

        if isinstance(param, ValueNode):
            if isinstance(param.value, VoidLiteral):
                return []
            elif isinstance(param.value, TupleLiteral):
                return tuple_unpacking(ctx, param.value, VarDefineNode, TypeError.param_def_expected)

        ctx.error(TypeError.param_def_expected, param)
        return []

    def update_literal(self, context, parent) -> None:
        self.in_param.update(context, parent)
        self.out_param.update(context, parent)
        self.in_param_list = self.gen_param_list(context, self.in_param)
        self.out_param_list = self.gen_param_list(context, self.out_param)

        new_ctx = context.clone()
        for param in self.in_param_list:
            param.add_ctx(new_ctx)

        for param in self.out_param_list:
            param.add_ctx(new_ctx)

        self.body.update(new_ctx, parent)
        return

    def type_check_literal(self, context: 'Context', expected_types=None) -> None:
        self.in_param = self.in_param.type_check(self.possible_types_at(expected_types, True))
        self.out_param = self.out_param.type_check(self.possible_types_at(expected_types, False))
        self.body = self.body.type_check()
        if self.in_param is None or self.out_param is None or self.body is None:
            return

        self.type = FunctionType(self.in_param.type, self.out_param.type)
        return

    @staticmethod
    def possible_types_at(types: Optional[set[Type]], input_or_output = True) -> Optional[set[Type]]:
        if types is None:
            return None

        possible_types = set()
        for t in types:
            if isinstance(t, FunctionType):
                if input_or_output:
                    possible_types.add(t.arg)
                else:
                    possible_types.add(t.ret)

        return possible_types

    def __str__(self):
        return f'{self.in_param} -> {self.out_param} {self.body}'

    def __repr__(self):
        return f'<{self.in_param.__repr__()} -> {self.out_param.__repr__()} {self.body.__repr__()}>'


class ProductTypeLiteral(Literal):
    def __init__(self, values: list['TypeDefineNode']):
        assert len(values) > 0
        super().__init__(values, set(), values[-1].location - values[0].location)
        return

    def update_literal(self, context, parent) -> None:
        for value in self.value:
            value.update(context, parent)
        return

    def type_check_literal(self, context: 'Context', expected_types=None) -> None:
        types = []
        for i, val in enumerate(self.value):
            self.value[i] = val.type_check(self.possible_types_at(expected_types, i))
            if self.value[i] is not None:
                types.append(self.value[i].type)

        self.type = ProductType(types)
        return

    @staticmethod
    def possible_types_at(types: Optional[set[Type]], index: int) -> Optional[set[Type]]:
        if types is None:
            return None

        possible_types = set()
        for t in types:
            if isinstance(t, ProductType):
                possible_types.add(t.fields[index])

        return possible_types

    def __str__(self):
        return f'{{{", ".join([str(v) for v in self.value])}}}'

    def __repr__(self):
        return f'<{{{", ".join([f"{v.__repr__()}" for v in self.value])}}}>'



# AST level

class Context:
    """Compile Time context to process AST and type check"""
    def __init__(self, global_vars: Globals, up_scope: Optional['Context'] = None):
        self.up_scope: Optional[Context] = up_scope
        self.global_vars: Globals = global_vars

        if up_scope is None:
            self.scope_level: int = 0
            self.namespace: dict[tuple[str, Type], NameDefineNode] = {}
            self.types: set[TypeDefineNode] = set()
            self.funcs: set[FunctionLiteral] = set()
            self.errors: list[Error] = []
            self.processing_queue: PriorityQueue = PriorityQueue(1)
        else:
            self.scope_level: int = up_scope.scope_level + 1
            self.namespace: dict[tuple[str, Type], NameDefineNode] = copy(up_scope.namespace)
            self.types: set[TypeDefineNode] = up_scope.types
            self.funcs: set[FunctionLiteral] = up_scope.funcs
            self.errors: list[Error] = up_scope.errors    # not a copy. All errors will go to the same collection
            self.processing_queue: PriorityQueue = up_scope.processing_queue
        return

    def get_definition(self, key: tuple[str, Type]) -> Optional['NameDefineNode']:
        """Returns the definition of the given name in the current context or None if it is not defined"""
        if key in self.namespace:
            return self.namespace[key]
        else:
            return None

    def get_possible_types(self, name: str) -> set[Type]:
        """Returns the possible types of the given name in the current context"""
        types = set()
        for key in self.namespace:
            if key[0] == name:
                types.add(key[1])
        return types

    def clone(self) -> 'Context':
        c = Context(self.global_vars, self)
        return c

    def error(self, e: TypeError, node: ['Node', 'Type'], *args) -> None:
        self.errors.append(Error(e, node.location, self.global_vars, *args))
        return

    def __repr__(self):
        return f'vars: {self.namespace}\nfuncs: {self.funcs}\ntypes: {self.types}'


# Category Nodes (DO NOT construct these nodes directly!)

class Node:
    def __init__(self, location: FileRange, parent_node: Optional['Node'] = None, type: Optional[Type] = None):
        self.parent: Optional[Node] = parent_node
        self.context: Optional[Context] = None
        self.location: FileRange = location
        self.type: Optional[Type] = type
        return

    def update(self, ctx: Context, parent: 'Node') -> None:
        """Builds up the context of the node, and allows the node to change itself if needed"""
        self.context = ctx
        self.parent = parent
        ctx.processing_queue.enqueue(self, 0)
        return

    def type_check(self, expected_types: Optional[set[Type]] = None, *args) -> Optional['Node']:
        """Updates current node and recursively calls type_check on all the children nodes and compares its type with them.
        When updating the node may return a different node, so the parent node must update its children with the returned node.
        If there is a type mismatch, it will add an error to the context
        When None is returned that means the errors found are severe enough that the basic information of the node is not safe to use
        :param expected_types:
        :param args: Extra parameters for optimisations and for passing information between nodes.
                    the 1st one is an int for which position of the parent node the current node is.
                    The parent can be known by accessing ``self.parent``
        :return: The node itself or a new node if it was updated"""
        return self

    def get_typedef(self) -> Optional['TypeDefineNode']:
        """Gets the type definition of the type of the Node"""
        return self.context.get_definition((self.type.name_tok.value, self.type))

    def error(self, e: TypeError, *args) -> None:
        """Adds an error to the context"""
        self.context.error(e, self, *args)
        return

    def get_scope(self) -> Optional['ScopeNode']:
        parent = self.parent
        while not (parent is None or isinstance(parent, ScopeNode)):
            parent = parent.parent

        return parent

    def __repr__(self):
        pass


class NameDefineNode(Node):
    def __init__(self, name: 'VariableNode', type: Optional[Type] = None):
        super().__init__(name.location, type=type)
        self.name: VariableNode = name
        return

    def get_id(self) -> tuple[str, Type]:
        """Returns the identifier of the Definition node to be used as key in the dictionary of namespace"""
        return self.name.name_tok.value, self.type

    def add_ctx(self, ctx: Context) -> None:
        """Adds the definition to the current context dict"""
        ctx.namespace[self.get_id()] = self
        return

    def update(self, ctx: Context, parent: 'Node') -> None:
        self.context = ctx
        self.parent = parent
        self.add_ctx(ctx)
        self.name.update(ctx, self)
        return

    def get_size(self):
        return self.type.size


# Base NODES

class ValueNode(Node):
    def __init__(self, tok: Literal):
        super().__init__(tok.location)
        self.value: Literal = tok
        return

    def update(self, ctx: Context, parent: Optional[Node]) -> None:
        self.context = ctx
        self.parent = parent
        ctx.processing_queue.enqueue(self)

        self.value.update_literal(ctx, self)
        return

    def type_check(self, expected_types: Optional[set[Type]] = None, *args) -> Optional['ValueNode']:
        self.value.type_check_literal(self.context, expected_types)  # mutates itself not returns new node
        self.type = self.value.type
        return self

    def __str__(self):
        return f'{self.value}'

    def __repr__(self):
        return f'<{self.value.__repr__()}>'


class VariableNode(Node):
    def __init__(self, repr_tok: Token):
        super().__init__(repr_tok.location)
        self.name_tok: Token = repr_tok
        return

    def get_possible_types(self) -> set[Type]:
        return self.context.get_possible_types(self.name_tok.value)

    def type_check(self, expected_types: Optional[set[Type]] = None, *args) -> Optional['VariableNode']:
        """Recursively calls type_check on all the children nodes and compares its type with them.
        If there is a type mismatch, it will add an error to the context
        :param expected_types:
        :param args: """
        types = self.get_possible_types()

        if expected_types is not None:
            types.intersection_update(expected_types)
            if len(types) == 0:
                self.error(TypeError.wrong_type, expected_types, self.type)
                return None
        else:
            if len(types) == 0:
                self.error(TypeError.unk_var, self)
                return None

        if len(types) > 1:
            self.error(TypeError.ambiguous_type, expected_types)

        for t in types:     # unpacking the set
            self.type = t   # (in case of multiple types, a random one will be selected to try and recover from the error)
            break
        return self

    def __str__(self):
        return f'{self.name_tok.value}: {self.type}'

    def __repr__(self):
        return f'<{self.name_tok.value.__repr__()}>'


class ScopeNode(Node):
    def __init__(self, child_nodes: list[Node], location):
        super().__init__(location, type=VoidType())
        self.child_nodes: list[Node] = child_nodes
        return

    def update(self, ctx: Context, parent: Optional['Node']) -> None:
        self.context = ctx
        self.parent = parent
        new_ctx = ctx.clone()
        for node in self.child_nodes:
            node.update(new_ctx, self)
        return

    def type_check(self, expected_types: Optional[set[Type]] = None, *args) -> Optional['ScopeNode']:
        """Recursively calls type_check on all the children nodes and compares its type with them.
        If there is a type mismatch, it will add an error to the context
        :param expected_types:
        :param *args: """
        child_nodes = []
        for node in self.child_nodes:
            node = node.type_check()
            if node is not None:
                child_nodes.append(node)

        self.child_nodes = child_nodes
        return self

    def __str__(self):
        string = f'\n{self.context.scope_level * "  "}{{\n'
        for node in self.child_nodes:
            string += f'{node.context.scope_level * "  "}{node}\n'
        return string + f'{self.context.scope_level * "  "}}}'

    def __repr__(self):
        new_line = '\n'
        return f'{{\n{new_line.join([f"{node.__repr__()}" for node in self.child_nodes])}\n}}'


# Operation Nodes

class AssignNode(Node):
    def __init__(self, var: Node, value: Node):
        super().__init__(var.location)
        self.var: Node = var
        self.value: Node = value
        return

    def update(self, ctx: Context, parent: Optional[Node]) -> None:
        self.context = ctx
        self.parent = parent
        ctx.processing_queue.enqueue(self)

        self.var.update(ctx, self)
        self.value.update(ctx, self)
        return

    def type_check(self, expected_types: Optional[set[Type]] = None, *args) -> Optional[Node]:
        if self.var.type == Types.infer.value:
            self.value = self.value.type_check()
            if self.value is None:
                return None
            self.var = self.var.type_check({self.value.type})
        else:
            self.var = self.var.type_check()
            if self.var is None:
                return None
            self.value = self.value.type_check({self.var.type})

        if self.var is None or self.value is None:
            return self

        self.type = self.var.type
        return self

    def __str__(self):
        return f'{self.var} = {self.value}'

    def __repr__(self):
        return f'<{self.var.__repr__()} = {self.value.__repr__()}>'


class UnOpNode(Node):
    def __init__(self, op: Token, child: Node):
        super().__init__(op.location)
        self.op = op
        self.child = child
        return

    def update(self, ctx: Context, parent: Optional[Node]) -> None:
        self.context = ctx
        self.parent = parent
        ctx.processing_queue.enqueue(self)

        self.child.update(ctx, self)
        return

    def type_check(self, expected_types: Optional[set[Type]] = None, *args) -> Optional['FuncCallNode']:
        assert self.op.value in dunder_funcs
        func_name = dunder_funcs[self.op.value]

        return FuncCallNode(VariableNode(func_name), [self.child]).type_check(expected_types)
        # TODO: add edge cases for the primitives (they dont need to be converted to function calls)

    def __str__(self):
        return f'{self.op.value} {self.child}'

    def __repr__(self):
        return f'<{self.op.value} {self.child.__repr__()}>'


class BinOpNode(Node):
    def __init__(self, op: Token, left: Node, right: Node):
        super().__init__(op.location)
        self.left_child = left
        self.right_child = right
        self.op = op
        return

    def update(self, ctx: Context, parent: Optional[Node]) -> None:
        self.context = ctx
        self.parent = parent
        ctx.processing_queue.enqueue(self)

        self.left_child.update(ctx, self)
        self.right_child.update(ctx, self)
        return

    def type_check(self, expected_types: Optional[set[Type]] = None, *args) -> Optional['FuncCallNode']:
        assert self.op.value in dunder_funcs
        func_name = dunder_funcs[self.op.value]

        return FuncCallNode(VariableNode(func_name), [self.left_child, self.right_child]).type_check()

    def __str__(self):
        return f'{self.left_child} {self.op.value} {self.right_child}'

    def __repr__(self):
        return f'<{self.left_child.__repr__()} {self.op.value} {self.right_child.__repr__()}>'


class DotOperatorNode(Node):
    def __init__(self, repr_tok: Token, var: Node, field: VariableNode):
        super().__init__(repr_tok.location)
        self.var: Node = var
        self.field: VariableNode = field
        self.attribute: Optional[NameDefineNode] = None
        return

    def update(self, ctx: Context, parent: Optional[Node]) -> None:
        self.context = ctx
        self.parent = parent
        ctx.processing_queue.enqueue(self)

        self.var.update(ctx, self)
        return

    def type_check(self, expected_types: Optional[set[Type]] = None, *args) -> Optional['DotOperatorNode']:
        self.var = self.var.type_check()
        if self.var is None:
            return None

        # TODO need to find a way from the type, to get to its typedef

        typedef = self.var.get_typedef()
        if typedef is None:
            assert False    # if the type is not visible then it should have been caught in the previous step

        self.attribute = typedef.get_attribute(self.field.name_tok.value)
        if self.attribute is None:
            self.error(TypeError.no_attribute, typedef.name.name_tok.value, self.field.name_tok.value)
            return None

        self.type = self.attribute.type

        return self

    def __str__(self):
        return f'{self.var}.{self.field.name_tok.value}'

    def __repr__(self):
        return f'<{self.var.__repr__()}.{self.field.__repr__()}>'


class IndexOperatorNode(Node):
    def __init__(self, repr_tok: Token, collection: Node, index: Node):
        super().__init__(repr_tok.location)
        self.collection: Node = collection
        self.index: Node = index
        return

    def update(self, ctx: Context, parent: Optional[Node]) -> None:
        self.context = ctx
        self.parent = parent
        ctx.processing_queue.enqueue(self)

        self.collection.update(ctx, self)
        self.index.update(ctx, self)
        return

    def type_check(self, expected_types: Optional[set[Type]] = None, *args) -> Optional['IndexOperatorNode']:
        self.collection = self.collection.type_check([])
        self.index = self.index.type_check([])

        if self.collection is None or self.index is None:
            return None

        assert Operators.index.value.value in dunder_funcs
        fn_name = dunder_funcs[Operators.index.value.value]
        return FuncCallNode(VariableNode(fn_name), [self.collection, self.index]).type_check([])

    def __str__(self):
        return f'{self.collection}[{self.index}]'

    def __repr__(self):
        return f'<{self.collection.__repr__()}[{self.index.__repr__()}]>'


class FuncCallNode(Node):
    def __init__(self, func: Node, args: list[Node]):
        super().__init__(func.location)
        self.func: Node = func
        self.args: list[Node] = args
        self.func_literal: Optional[FunctionLiteral] = None
        return

    def update(self, ctx: Context, parent: Optional[Node]) -> None:
        self.context = ctx
        self.parent = parent
        ctx.processing_queue.enqueue(self)

        self.func.update(ctx, self)
        for arg in self.args:
            arg.update(ctx, self)
        return

    def type_check(self, expected_types: Optional[set[Type]] = None, *args) -> Optional['FuncCallNode']:
        self.func.type_check([])  # TODO need to check to see if we should send extra info to syntax sugar for funcs

        self.args = [arg.type_check([]) for arg in self.args]
        if self.func is None or None in self.args:
            return None

        assert isinstance(self.func.type, FunctionType)

        if isinstance(self.func.type.arg, TupleType):
            in_type = tuple_type_unpacking(self.context, self.func.type.arg, FunctionType, TypeError.wrong_type)
        else:
            in_type = [self.func.type.arg]

        in_func_type = TupleType([arg.type for arg in self.args])

        if in_type != in_func_type.types:
            self.error(TypeError.wrong_func_type, self.func, in_func_type, self.func.type.arg)

        return self

    def __str__(self):
        return f'{self.func}{self.args}'

    def __repr__(self):
        return f'<{self.func.__repr__()}({", ".join([f"{arg.__repr__()}" for arg in self.args])})>'


# Definition Nodes

class VarDefineNode(NameDefineNode):
    def __init__(self, var_name: VariableNode, var_type: Type):
        super().__init__(var_name, var_type)
        self.class_def: Optional[TypeDefineNode] = None
        return

    def update(self, ctx: Context, parent: Optional[Node]) -> None:
        self.context = ctx
        self.parent = parent
        ctx.processing_queue.enqueue(self)

        self.name.update(ctx, self)
        return

    def type_check(self, expected_types: Optional[set[Type]] = None, *args) -> Optional['VarDefineNode']:
        if self.type == Types.infer.value:
            if expected_types is None:
                self.error(TypeError.cannot_infer_type, self.name.name_tok.value, set())
                return None

            if len(expected_types) != 1:
                self.error(TypeError.cannot_infer_type, self.name.name_tok.value, expected_types)
                # any it's fine. we just want to recover from the error
            self.type = expected_types.pop()
        return self

    def __str__(self):
        return f'{self.name.name_tok.value}: {self.type}'

    def __repr__(self):
        return f'<{self.name.__repr__()}: {self.type.__repr__()}>'


# Type Nodes

class TypeDefineNode(NameDefineNode):
    def __init__(self, name: VariableNode, generics: Optional[list[VarDefineNode | VariableNode | ValueNode]] = None):
        super().__init__(name, copy(Types.type.value))
        self.generics: Optional[list[VarDefineNode | VariableNode | ValueNode]] = generics if generics is not None else []
        return

    def add_ctx(self, ctx: Context) -> None:
        """Adds the definition to the current context dict"""
        super().add_ctx(ctx)
        ctx.types.add(self)
        return

    def get_attribute(self, name: str) -> Optional[NameDefineNode]:
        """Returns the attribute of the type definition with the given name or None if it is not defined.
        Attribute can be a field or a variant type"""
        return None

    def update(self, ctx: Context, parent: Optional[Node]) -> None:
        self.context = ctx
        self.parent = parent
        ctx.processing_queue.enqueue(self, 0)

        self.name.update(ctx, self)
        return

    def type_check(self, expected_types: Optional[set[Type]] = None, *args) -> Optional['TypeDefineNode']:
        self.generics = [g.type_check([]) for g in self.generics]
        if None in self.generics:
            return None

        return self

    def __repr__(self):
        return f'type {self.name} {self.type}'


class ProductTypeDefineNode(TypeDefineNode):
    def __init__(self, name: VariableNode, fields: list[NameDefineNode], generics = None):
        super().__init__(name, generics)
        self.fields: dict[str, NameDefineNode] = dict(map(lambda e: (e.name.name_tok.value, e), fields))
        return

    def get_attribute(self, name: str) -> Optional[NameDefineNode]:
        if name in self.fields:
            return self.fields[name]
        return None

    def __repr__(self):
        string = "\n".join([f'{field}' for field in self.fields])
        return f'type {self.name} {self.type} {{\n{string}\n}}'


class TypeAliasDefineNode(TypeDefineNode):
    def __init__(self, name: VariableNode, other_type: Type, generics = None):
        super().__init__(name, generics)
        self.other_type: Type = other_type
        return

    def get_attribute(self, name: str) -> Optional[NameDefineNode]:
        typedef = self.context.get_definition((self.other_type.name_tok.value, self.other_type))    # TODO: types need to have their own id
        if not isinstance(typedef, TypeDefineNode):
            return None
        return typedef.get_attribute(name)

    def __repr__(self):
        return f'type {self.name} = {self.other_type}'


class SumTypeDefineNode(TypeDefineNode):
    def __init__(self, name: VariableNode, subtypes: list[TypeDefineNode], generics = None):
        super().__init__(name, generics)
        self.name: VariableNode = name
        self.variants: dict[str, TypeDefineNode] = dict(map(lambda e: (e.name.name_tok.value, e), subtypes))
        self.type: Type = self.make_type(name, subtypes)
        return

    @staticmethod
    def make_type(name: VariableNode, subtypes: list[TypeDefineNode]) -> Type:
        return SumType(name.name_tok, subtypes)

    def get_attribute(self, name: str) -> Optional[NameDefineNode]:
        if name in self.variants:
            return self.variants[name]
        return None

    def __repr__(self):
        string = ", ".join([f'{subtype}' for subtype in self.variants])
        return f'type {self.name} = {{{string}}}'


# Control Flow

class IfNode(Node):
    def __init__(self, repr_tok: Token, condition: Node, body: Node):
        super().__init__(repr_tok.location, type=VoidType())
        self.condition = condition
        self.body = body
        self.else_statement: Optional[ElseNode] = None
        return

    def update(self, ctx: Context, parent: Optional[Node]) -> Optional['IfNode']:
        self.context = ctx
        self.parent = parent

        self.condition = self.condition.update(ctx, self)

        self.body = self.body.update(ctx, self)
        if self.body is None or self.condition is None:
            return None

        if self.else_statement is not None:
            self.else_statement = self.else_statement.update(ctx, self)

        return self

    def __repr__(self):
        if self.else_statement is None:
            return f'if {self.condition} {self.body}'
        else:
            return f'if {self.condition} {self.body} \n{self.context.scope_level*"  "}{self.else_statement}'


class ElseNode(Node):
    def __init__(self, repr_tok: Token, body: Node, if_statement: Optional[IfNode]):
        super().__init__(repr_tok.location, type=VoidType())
        self.body = body
        self.if_statement: Optional[IfNode] = if_statement
        return

    def update(self, ctx: Context, parent: Optional[Node]) -> Optional['ElseNode']:
        self.context = ctx
        self.parent = parent

        self.body = self.body.update(ctx, self)
        if self.body is None:
            return None

        return self

    def __repr__(self):
        return f'else {self.body}'


class WhileNode(Node):
    def __init__(self, repr_tok: Token, condition: Node, body: Node):
        super().__init__(repr_tok.location, type=VoidType())
        self.condition = condition
        self.body = body
        return

    def update(self, ctx: Context, parent: Optional[Node]) -> Optional['WhileNode']:
        self.context = ctx
        self.parent = parent

        self.condition = self.condition.update(ctx, self)

        self.body = self.body.update(ctx, self)
        if self.body is None or self.condition is None:
            return None

        return self

    def __repr__(self):
        return f'while {self.condition} {self.body}'


class LoopModifierNode(Node):
    def __init__(self, repr_tok: Token, value: Optional[Token], error: TypeError):
        super().__init__(repr_tok.location, type=VoidType())
        self.loop: Optional[WhileNode] = None
        self.error: TypeError = error
        if value is None:
            self.value: Optional[Token] = Token(TT.LITERAL, 1)
        else:
            self.value: Optional[Token] = value

    def update(self, ctx: Context, parent: Optional[Node]) -> Optional['LoopModifierNode']:
        self.context = ctx
        self.parent = parent

        if self.value.value <= 0:
            ctx.error(TypeError.pos_int_expected, self)
            self.value.value = 1  # ignore and proceed as default

        i = self.value.value
        self.loop = self
        while i > 0 and self.loop is not None:
            self.loop = self.loop.get_scope().parent
            i -= 1

        if self.loop is None:
            ctx.error(self.error, self) # maybe improve this error message

        return self


class ExitNode(LoopModifierNode):
    def __init__(self, repr_tok: Token, value: Optional[Token] = None):
        super().__init__(repr_tok, value, TypeError.exit_not_in_body)

    def __repr__(self):
        if self.value is None:
            return f'exit'
        else:
            return f'exit {self.value.value}'


class SkipNode(LoopModifierNode):
    def __init__(self, repr_tok: Token, value: Optional[Token] = None):
        super().__init__(repr_tok, value, TypeError.skip_not_in_loop)

    def __repr__(self):
        if self.value is None:
            return f'skip'
        else:
            return f'skip {self.value.value}'


class MatchNode(Node):
    def __init__(self, repr_tok: Token, value: Node, cases: list['CaseNode']):
        assert len(cases) > 0
        super().__init__(cases[-1].location - repr_tok.location, type=VoidType())
        self.value: Node = value
        self.cases: list['CaseNode'] = cases
        return

    def update(self, ctx: Context, parent: Optional[Node]) -> Optional['MatchNode']:
        self.context = ctx
        self.parent = parent

        self.value = self.value.update(ctx, self)
        if self.value is None:
            return None

        self.cases = [case.update(ctx, self) for case in self.cases]
        if None in self.cases:
            return None

        return self

    def __repr__(self):
        string = f'match {self.value} {{\n'
        for case in self.cases:
            string += f'{case}\n'
        return string + '}'


class CaseNode(Node):
    def __init__(self, variant: VariableNode, body: Node):
        super().__init__(body.location - variant.location, type=VoidType())
        self.variant: VariableNode = variant
        self.body: Node = body
        return

    def update(self, ctx: Context, parent: Optional[Node]) -> Optional['CaseNode']:
        self.context = ctx
        self.parent = parent

        self.variant = self.variant.update(ctx, self)
        if self.variant is None:
            return None

        self.body = self.body.update(ctx, self)
        if self.body is None:
            return None

        return self

    def __repr__(self):
        return f'{self.variant} => {self.body}'


### Instruction Nodes (translation process)



# constants

dunder_funcs = {
    Operators.not_.value.value: Token(TT.IDENTIFIER, '__not__'),
    Operators.neg.value.value: Token(TT.IDENTIFIER, '__neg__'),
    Operators.mlt.value.value: Token(TT.IDENTIFIER, '__mlt__'),
    Operators.div.value.value: Token(TT.IDENTIFIER, '__div__'),
    Operators.mod.value.value: Token(TT.IDENTIFIER, '__mod__'),
    Operators.add.value.value: Token(TT.IDENTIFIER, '__add__'),
    Operators.sub.value.value: Token(TT.IDENTIFIER, '__sub__'),
    Operators.shr.value.value: Token(TT.IDENTIFIER, '__shr__'),
    Operators.shl.value.value: Token(TT.IDENTIFIER, '__shl__'),
    Operators.gt.value.value: Token(TT.IDENTIFIER, '__gt__'),
    Operators.gte.value.value: Token(TT.IDENTIFIER, '__gte__'),
    Operators.lt.value.value: Token(TT.IDENTIFIER, '__lt__'),
    Operators.lte.value.value: Token(TT.IDENTIFIER, '__lte__'),
    Operators.dif.value.value: Token(TT.IDENTIFIER, '__dif__'),
    Operators.equ.value.value: Token(TT.IDENTIFIER, '__equ__'),
    Operators.xor.value.value: Token(TT.IDENTIFIER, '__xor__'),
    Operators.and_.value.value: Token(TT.IDENTIFIER, '__and__'),
    Operators.or_.value.value: Token(TT.IDENTIFIER, '__or__'),
    Operators.index.value.value: Token(TT.IDENTIFIER, '__get__'),
}
