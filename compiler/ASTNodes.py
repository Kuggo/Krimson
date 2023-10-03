from copy import copy, deepcopy
from Instructions import *


# helper functions
def tuple_unpacking(ctx: 'Context', node: 'TupleLiteral', error) -> list['VariableNode']:
    nodes = []
    for value in node.value:
        if isinstance(value, TupleLiteral):
            nodes += tuple_unpacking(ctx, value, error)
        elif isinstance(value, VariableNode):
            nodes.append(VariableNode(value.name_tok, value.type))
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

def iterable_str(iterable):
    return f'{", ".join([str(i) for i in iterable])}'


# Errors

class TypeError(Enum):
    no_attribute = 'type "{!s}" has no attribute "{!s}"'
    param_def_expected = 'Parameter definition expected'
    undefined_function = 'Undefined function for the given args'
    shadow_same_scope = 'variable "{!s}" was already defined in the current scope. It cannot be shadowed'
    exit_not_in_body = 'exit keyword found outside the specified body depth'
    skip_not_in_loop = 'skip keyword found outside the specified body depth'
    pos_int_expected = 'Positive integer expected'
    wrong_type = 'Expected type "{!s}" but got "{!s}"'
    not_callable = 'Cannot call "{!s}" as it is not a function'
    not_assignable = 'Cannot assign values to "{!s}"'
    ambiguous_name = 'Ambiguous name. Multiple variables with possible types "{!s}" could fit. Use `<name>: <type>` to specify which variable to use'
    no_suitable_var = 'No suitable variable found for name "{!s}". Expected types: {!s}. Possible types: {!s}'
    cannot_infer_type = 'Cannot infer type of variable "{!s}". Possible types: {!s}. \nUse `<name>: <type>` in any usage of the variable to specify its type'
    unk_var = 'No associated type for name of variable "{!s}". \nUse `<name>: <type>` in any usage of the variable to specify its type'
    type_not_found0 = 'Type "{!s}" was defined using "{!s}" which is undefined/not visible in the current scope'
    type_not_found1 = 'Variable {!s} was defined as being of type "{!s}" which is undefined/not visible in the current scope'
    wrong_func_type = 'Expected function with input argument of type "{!s}" but got "{!s}"'
    product_type_lit_needs_name = 'Cannot define a value for a field of a product type literal without its id. Use `<id> = <value>` instead'
    field_name_expected = 'Field name expected, found {!s} instead'


# Types directly supported by the compiler

class ProductType(Type):
    def __init__(self, fields: set[tuple[str, Type]]):
        super().__init__()
        self.fields: set[tuple[str, Type]] = fields
        return

    def __eq__(self, other: 'ProductType'):
        return isinstance(other, ProductType) and self.name_tok == other.name_tok and self.fields == other.fields

    def __hash__(self):
        return sum([f.__hash__() for f in self.fields])

    def get_possible_attributes(self, name: str) -> set[Type]:
        return {t for n, t in self.fields if n == name}

    def get_label(self) -> str:
        return f'{self.name_tok.value}_{"_".join([f[1].get_label() for f in self.fields])}'

    def builtin_type(self):
        return True

    def __str__(self):
        return f'{{{", ".join([iterable_str(f) for f in self.fields])}}}'

    def __repr__(self):
        return f'<{self.name_tok.value}({", ".join([f"{f.__repr__()}" for f in self.fields])})>'


class SumType(Type):
    def __init__(self, types: list):
        super().__init__()
        self.types: list[Type] = types
        return

    def __eq__(self, other: 'SumType'):
        return isinstance(other, SumType) and self.types == other.types

    def __hash__(self):
        return sum([f.__hash__() for f in self.types])

    def get_possible_attributes(self, name: str) -> set[Type]:
        return {t for t in self.types if name == t.name_tok.value}

    def builtin_type(self):
        return True

    def __str__(self):
        return f'{{{" | ".join([str(t) for t in self.types])}}}'

    def __repr__(self):
        return f'<{{{" | ".join([f"{t.__repr__()}" for t in self.types])}}}>'



# Literals directly supported by compiler

class VoidLiteral(Literal):
    def __init__(self, location: FileRange):
        super().__init__('()', {VoidType()}, location)
        return

    def __str__(self):
        return f'()'

    def __repr__(self):
        return f'<void>'


class TupleLiteral(Literal):
    def __init__(self, values: list['Node']):
        assert len(values) > 0
        super().__init__(values, set(), values[-1].location - values[0].location)
        self.value: list[Node] = values
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
    def __init__(self, values: list, location: FileRange):
        self.value: list[Node] = values
        super().__init__(values, set(), location)
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
        super().__init__(None, set(), body.location - in_param.location)
        self.in_param: 'Node' = in_param
        self.out_param: 'Node' = out_param

        self.in_param_list: list[VariableNode] = []
        self.out_param_list: list[VariableNode] = []
        self.body: Node = body
        return

    def gen_param_list(self, ctx: 'Context', param: 'Node') -> list['VariableNode']:
        if isinstance(param, VariableNode):
            return [param]

        if isinstance(param, ValueNode):
            if isinstance(param.value, VoidLiteral):
                return []
            elif isinstance(param.value, TupleLiteral):
                return tuple_unpacking(ctx, param.value, TypeError.param_def_expected)

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
    def __init__(self, values: list['AssignNode'], location: FileRange):
        assert len(values) > 0
        self.value: list['AssignNode'] = values
        super().__init__(values, set(), location)
        self.fields: dict[tuple[str, Type], 'Node'] = {}
        return

    def update_literal(self, context, parent) -> None:
        new_ctx = context.clone()
        for value in self.value:
            value.update(new_ctx, parent)
        return

    def type_check_literal(self, context: 'Context', expected_types=None) -> None:
        if expected_types is None:
            pass
        types = set()
        for v in self.value:
            if isinstance(v, AssignNode):
                if isinstance(v.var, VariableNode):
                    t = self.possible_types_at(expected_types, v.var.name_tok.value)
                    v.value = v.value.type_check(t)
                    if v.value is None:
                        continue

                    if v.var.type is None or v.var.type == Types.infer.value:
                        v.var.type = v.value.type
                    self.fields[v.var.get_id()] = v.value
                    types.add(v.var.get_id())
                else:
                    context.error(TypeError.field_name_expected, v.location)
                continue
            context.error(TypeError.product_type_lit_needs_name, v.location)

        self.type = ProductType(types)
        return

    @staticmethod
    def possible_types_at(types: Optional[set[Type]], name: str) -> Optional[set[Type]]:
        if types is None:
            return None

        possible_types = set()
        for t in types:
            if isinstance(t, ProductType):
                for k, v in t.fields:
                    if k[0] == name:
                        possible_types.add(v)

        return possible_types

    def __str__(self):
        return f'{{{", ".join([str(v) for v in self.fields.values()])}}}'

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
            self.namespace: dict[tuple[str, Type], VariableNode] = {}
            self.types: set[TypeDefineNode] = set()
            self.funcs: set[FunctionLiteral] = set()
            self.errors: list[Error] = []
            self.processing_queue: PriorityQueue = PriorityQueue(1)
        else:
            self.scope_level: int = up_scope.scope_level + 1
            self.namespace: dict[tuple[str, Type], VariableNode] = copy(up_scope.namespace)
            self.types: set[TypeDefineNode] = up_scope.types
            self.funcs: set[FunctionLiteral] = up_scope.funcs
            self.errors: list[Error] = up_scope.errors    # not a copy. All errors will go to the same collection
            self.processing_queue: PriorityQueue = up_scope.processing_queue
        return

    def get_definition(self, key: tuple[str, Type]) -> Optional['VariableNode']:
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

    def __str__(self):
        return f'vars: {iterable_str(self.namespace)}\nfuncs: {iterable_str(self.funcs)}\ntypes: {iterable_str(self.types)}'

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
        return self.context.get_definition(self.type.get_id())

    def error(self, e: TypeError, *args) -> None:
        """Adds an error to the context"""
        self.context.error(e, self, *args)
        return

    def get_scope(self) -> Optional['ScopeNode']:
        parent = self.parent
        while not (parent is None or isinstance(parent, ScopeNode)):
            parent = parent.parent

        return parent

    @staticmethod
    def assignable() -> bool:
        """Returns whether the node can be used on the left side of an AssignNode"""
        return False

    def __repr__(self):
        pass


# Definition Nodes

class VariableNode(Node):
    def __init__(self, repr_tok: Token, t: Optional[Type] = None, location: Optional[FileRange] = None):
        super().__init__(repr_tok.location if location is None else location, type=t)
        self.name_tok: Token = repr_tok
        self.type_def: Optional[TypeDefineNode] = None
        return

    def get_id(self) -> tuple[str, Type]:
        """Returns the identifier of the Definition node to be used as key in the dictionary of namespace"""
        return self.name_tok.value, self.type

    def get_possible_types(self) -> set[Type]:
        typedefs = self.context.get_definition((self.name_tok.value, Types.infer.value)) # can only infer type once per name
        if typedefs is not None:
            assert isinstance(typedefs, set)
            for t in typedefs:
                t = t.type_check()
                if t is not None:
                    self.context.namespace[t.get_id()] = t
            self.context.namespace.pop((self.name_tok.value, Types.infer.value))    # remove the old definition
        return self.context.get_possible_types(self.name_tok.value)

    def add_ctx(self, ctx: Context) -> None:
        """Adds the definition to the current context dict"""
        if self.type == Types.infer.value:
            if self.get_id() in ctx.namespace:
                ts = ctx.namespace[self.get_id()]
                assert isinstance(ts, set)
                ts.add(self)
            else:
                ctx.namespace[self.get_id()] = {self}

        else:
            ctx.namespace[self.get_id()] = self
        return

    def update(self, ctx: Context, parent: Optional[Node]) -> None:
        self.context = ctx
        self.parent = parent
        ctx.processing_queue.enqueue(self)
        if self.type is not None:
            self.add_ctx(ctx)
        return

    def type_check(self, expected_types: Optional[set[Type]] = None, *args) -> Optional['VariableNode']:
        if self.type == Types.infer.value:
            if expected_types is None or len(expected_types) == 0:
                self.error(TypeError.cannot_infer_type, self.name_tok.value, set())
                return None

            if len(expected_types) > 1:
                self.error(TypeError.cannot_infer_type, self.name_tok.value, expected_types)
                # any it's fine. we just want to recover from the error
            self.type = expected_types.pop()

        if self.type is None:
            types = self.get_possible_types()

            if expected_types is not None:
                t = types.intersection(expected_types)
                if len(t) > 1:
                    self.error(TypeError.ambiguous_name, iterable_str(t))
                    return None
                elif len(t) == 0:
                    self.error(TypeError.no_suitable_var, self.name_tok.value, iterable_str(expected_types),
                               iterable_str(types))
                    return None
                self.type = t.pop()
                return self

            elif len(types) == 0:
                self.error(TypeError.unk_var, self)
                return None

            elif len(types) > 1:
                self.error(TypeError.ambiguous_name, iterable_str(expected_types))

            self.type = types.pop()  # unpacking the set
            # in case of multiple types, a random one will be selected to try and recover from the error

        if self.type.builtin_type():
            return self
        self.type_def = self.context.get_definition(self.type.get_id())
        if self.type_def is None and not self.type.builtin_type():
            self.error(TypeError.type_not_found1, self.name_tok.value, self.type.name_tok.value)
            return None
        return self

    @staticmethod
    def assignable() -> bool:
        """Returns whether the node can be used on the left side of an AssignNode"""
        return True

    def __str__(self):
        return f'{self.name_tok.value}'

    def __repr__(self):
        return f'<{self.name_tok.value.__repr__()}:{self.type.__repr__()}>'


class TypeDefineNode(VariableNode):
    def __init__(self, name: VariableNode, type: Optional[Type] = None, generics: Optional[list[Node]] = None, location: Optional[FileRange] = None):
        super().__init__(name.name_tok, copy(Types.type.value), location=location)
        self.generics: Optional[list[Node]] = generics if generics is not None else []
        self.type_val: Optional[Type] = type
        return

    def add_ctx(self, ctx: Context) -> None:
        """Adds the definition to the current context dict"""
        ctx.namespace[self.get_id()] = self
        ctx.types.add(self)
        return

    def get_possible_attributes(self, name: str) -> set[Type]:
        """Returns the attribute of the type definition with the given name or None if it is not defined.
        Attribute can be a field or a variant type"""
        if self.type_val is None:
            return set()

        return self.type_val.get_possible_attributes(name)

    def update(self, ctx: Context, parent: Optional[Node]) -> None:
        self.context = ctx
        self.parent = parent
        ctx.processing_queue.enqueue(self, 0)
        self.add_ctx(ctx)
        return

    def type_check(self, expected_types: Optional[set[Type]] = None, *args) -> Optional['TypeDefineNode']:
        self.generics = [g.type_check() for g in self.generics]
        if None in self.generics:
            return None

        if self.type_val is None:
            return self
        if not self.type_val.builtin_type():
            self.error(TypeError.type_not_found0, self.name_tok.value, self.type_val.name_tok.value)
            return self

        typedef = self.context.get_definition(self.type_val.get_id())
        while isinstance(typedef, TypeDefineNode):
            self.type_val = typedef.type_val
            typedef = self.context.get_definition(typedef.type_val.get_id())

        return self

    def __repr__(self):
        return f'type {self.name_tok} {self.type}'


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
        if expected_types is not None:
            types = {}
            for t in expected_types:
                typedef = self.context.get_definition(t.get_id())
                if isinstance(typedef, TypeDefineNode):     # it's an alias
                    types[typedef.type_val] = t
            self.value.type_check_literal(self.context, set(types.keys()))
            if self.value.type in types:
                self.value.type = types[self.value.type]
        else:
            self.value.type_check_literal(self.context, expected_types)
        self.type = self.value.type
        return self

    def __str__(self):
        return f'{self.value}'

    def __repr__(self):
        return f'<{self.value.__repr__()}>'


class ScopeNode(Node):
    def __init__(self, child_nodes: list[Node], location: FileRange):
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
        string = f'\n{max(self.context.scope_level-1, 0) * "  "}{{\n'
        for node in self.child_nodes:
            string += f'{node.context.scope_level * "  "}{node}\n'
        return string + f'{max(self.context.scope_level-1, 0) * "  "}}}'

    def __repr__(self):
        new_line = '\n'
        return f'{{\n{new_line.join([f"{node.__repr__()}" for node in self.child_nodes])}\n}}'


# Operation Nodes

class AssignNode(Node):
    def __init__(self, var: Node, value: Node):
        super().__init__(location = value.location - var.location)
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
        if isinstance(self.var, VariableNode) and self.var.type == Types.infer.value:
            self.value = self.value.type_check()
            if self.value is None:
                return None
            self.var = self.var.type_check({self.value.type})
        else:
            self.var = self.var.type_check()
            if self.var is None:
                return None
            if isinstance(self.var, FuncCallNode):  # it was an index
                return self.var
            self.value = self.value.type_check({self.var.type})

        if self.var is None or self.value is None:
            return self

        if self.var.type != self.value.type:
            self.error(TypeError.wrong_type, self.var.type, self.value.type)

        self.type = self.var.type
        if not self.var.assignable():
            self.error(TypeError.not_assignable, self.var)
        return self

    def __str__(self):
        return f'{self.var} = {self.value}'

    def __repr__(self):
        return f'<{self.var.__repr__()} = {self.value.__repr__()}>'


class UnOpNode(Node):
    def __init__(self, op: Token, child: Node):
        super().__init__(location = child.location - op.location)
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

        f = FuncCallNode(VariableNode(func_name), ValueNode(TupleLiteral([self.child])))
        f.update(self.context, self)
        return f.type_check(expected_types)
        # TODO: add edge cases for the primitives (they dont need to be converted to function calls)

    def __str__(self):
        return f'{self.op.value} {self.child}'

    def __repr__(self):
        return f'<{self.op.value} {self.child.__repr__()}>'


class BinOpNode(Node):
    def __init__(self, op: Token, left: Node, right: Node):
        super().__init__(location = right.location - left.location)
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

        f = FuncCallNode(VariableNode(func_name), ValueNode(TupleLiteral([self.left_child, self.right_child])))
        f.update(self.context, self)
        return f.type_check()

    def __str__(self):
        return f'{self.left_child} {self.op.value} {self.right_child}'

    def __repr__(self):
        return f'<{self.left_child.__repr__()} {self.op.value} {self.right_child.__repr__()}>'


class DotOperatorNode(Node):
    def __init__(self, var: Node, field: VariableNode):
        super().__init__(location = field.location - var.location)
        self.var: Node = var
        self.field: VariableNode = field
        return

    def update(self, ctx: Context, parent: Optional[Node]) -> None:
        self.context = ctx
        self.parent = parent
        ctx.processing_queue.enqueue(self)

        self.var.update(ctx, self)
        return

    def type_check(self, expected_types: Optional[set[Type]] = None, *args) -> Optional['Node']:
        self.var = self.var.type_check()
        if self.var is None:
            return None

        typedef = self.var.get_typedef()
        if typedef is None:
            # no need to throw error here because it will be caught when defining the type itself
            return None
        typedef = typedef.type_check()   # just making sure we are going to use the updated version of it
        possible_types = typedef.get_possible_attributes(self.field.name_tok.value)

        if len(possible_types) == 0:
            self.error(TypeError.no_attribute, self.var.type.name_tok.value, self.field.name_tok.value)
            return self.var     # trying to recover from the error

        elif len(possible_types) > 1:
            if expected_types is None:
                self.field.error(TypeError.ambiguous_name, iterable_str(possible_types))
            else:
                t = possible_types.intersection(expected_types)
                if len(t) > 1:
                    self.field.error(TypeError.ambiguous_name, iterable_str(possible_types))
                if len(t) >= 1:
                    possible_types = t

        self.type = possible_types.pop()
        self.field.type = self.type
        return self

    def __str__(self):
        return f'{self.var}.{self.field.name_tok.value}'

    def __repr__(self):
        return f'<{self.var.__repr__()}.{self.field.__repr__()}>'


class IndexOperatorNode(Node):
    def __init__(self, collection: Node, index: Node):
        super().__init__(location = index.location - collection.location)
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
        assert Operators.index.value.value in dunder_funcs
        fn_name = dunder_funcs[Operators.index.value.value]

        f = FuncCallNode(VariableNode(fn_name), ValueNode(TupleLiteral([self.collection, self.index])))
        f.update(self.context, self)
        return f.type_check()

    @staticmethod
    def assignable() -> bool:
        """Returns whether the node can be used on the left side of an AssignNode"""
        return True

    def __str__(self):
        return f'{self.collection}[{self.index}]'

    def __repr__(self):
        return f'<{self.collection.__repr__()}[{self.index.__repr__()}]>'


class FuncCallNode(Node):
    def __init__(self, func: Node, arg: Node):
        super().__init__(location = arg.location - func.location)
        self.func: Node = func
        self.arg: Node = arg
        self.func_literal: Optional[FunctionLiteral] = None
        return

    def update(self, ctx: Context, parent: Optional[Node]) -> None:
        self.context = ctx
        self.parent = parent
        ctx.processing_queue.enqueue(self)

        self.func.update(ctx, self)
        self.arg.update(ctx, self)
        return

    def type_check(self, expected_types: Optional[set[Type]] = None, *args) -> Optional['FuncCallNode']:
        self.func = self.func.type_check()
        if self.func is None:
            return None

        if not isinstance(self.func.type, FunctionType):
            self.error(TypeError.not_callable, self.func)
            return None

        self.arg = self.arg.type_check({self.func.type.arg})
        if self.arg is None:
            return None

        if self.func.type.arg != self.arg.type:
            self.error(TypeError.wrong_func_type, self.func.type.arg, self.arg.type)
            return None

        self.type = self.func.type.ret
        return self

    def __str__(self):
        return f'{self.func}{self.arg}'

    def __repr__(self):
        return f'<{self.func.__repr__()} {self.arg}>'


# Control Flow

class IfNode(Node):
    def __init__(self, repr_tok: Token, condition: Node, body: Node):
        super().__init__(location = body.location - repr_tok.location, type=VoidType())
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
        super().__init__(location = body.location - repr_tok.location, type=VoidType())
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
        super().__init__(location = body.location - repr_tok.location, type=VoidType())
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
        loc = repr_tok.location - value.location if value is not None else repr_tok.location
        super().__init__(location = loc, type=VoidType())
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
    def __init__(self, location: FileRange, value: Node, cases: list['CaseNode']):
        assert len(cases) > 0
        super().__init__(location, type=VoidType())
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
