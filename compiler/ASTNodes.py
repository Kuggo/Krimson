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
    unk_var = 'variable "{}" not defined or visible in scope'
    param_def_expected = 'Parameter definition expected'
    undefined_function = 'Undefined function for the given args'
    shadow_same_scope = 'variable "{}" was already defined in the current scope. It cannot be shadowed'
    exit_not_in_body = 'exit keyword found outside the specified body depth'
    skip_not_in_loop = 'skip keyword found outside the specified body depth'
    pos_int_expected = 'Positive integer expected'
    wrong_type = 'Expected type "{}" but got "{}"'
    wrong_func_type = 'Expected function with input arguments of type "{}" but got "{}"'

    # Compile_time detected runtime errors
    out_of_bounds = 'Index is out of bounds'
    no_element = 'No element with such key'


# Literals directly supported by compiler

class TupleLiteral(Literal):
    def __init__(self, values: list['ExpressionNode']):
        assert len(values) > 0
        super().__init__(values, None, values[-1].location - values[0].location)
        return

    def type_check_literal(self, ctx: 'Context', parent: Optional['Node']) -> None:
        types = []
        for i, val in enumerate(self.value):
            self.value[i] = val.type_check(ctx, parent)
            if self.value[i] is not None:
                types.append(self.value[i].type)

        self.literal_type = TupleType(types)
        return

    def __str__(self):
        return f'({", ".join([str(v) for v in self.value])})'

    def __repr__(self):
        return f'<({", ".join([f"{v.__repr__()}" for v in self.value])})>'


class VoidLiteral(Literal):
    def __init__(self, location: FileRange):
        super().__init__('void', VoidType(), location)
        return

    def type_check_literal(self, ctx: 'Context', parent: Optional['Node']) -> None:
        return

    def __str__(self):
        return f'void'

    def __repr__(self):
        return f'<void>'


class ArrayLiteral(Literal):
    def __init__(self, values: list):
        assert len(values) > 0
        super().__init__(values, None, values[-1].location - values[0].location)
        return

    def type_check_literal(self, ctx: 'Context', parent: Optional['Node']) -> None:
        t = None
        for i, val in enumerate(self.value):
            val = val.type_check(ctx, parent)
            if val is None:
                continue
            self.value[i] = val

            if t is None:
                t = self.value[i].type

            if self.value[i].type != t:
                ctx.error(TypeError.wrong_type, t, self.value[i].type.name_tok.value, t)
                continue

        self.literal_type = ArrayType(t)
        return

    def __str__(self):
        return f'[{", ".join([str(v) for v in self.value])}]'

    def __repr__(self):
        return f'<[{", ".join([f"{v.__repr__()}" for v in self.value])}]>'


class FunctionLiteral(Literal):
    def __init__(self, in_param: 'ExpressionNode', out_param: 'ExpressionNode', body: 'Node'):
        super().__init__(None, None)
        self.in_param: 'ExpressionNode' = in_param
        self.out_param: 'ExpressionNode' = out_param

        self.in_param_list: list[NameDefineNode] = []
        self.out_param_list: list[NameDefineNode] = []
        self.body: Node = body
        return

    def gen_param_list(self, ctx: 'Context', param: 'ExpressionNode') -> list['NameDefineNode']:
        if isinstance(param, VarDefineNode):
            return [param]

        if isinstance(param, ValueNode):
            if isinstance(param.value, VoidLiteral):
                return []
            elif isinstance(param.value, TupleLiteral):
                return tuple_unpacking(ctx, param.value, VarDefineNode, TypeError.param_def_expected)

        ctx.error(TypeError.param_def_expected, param)
        return []

    def type_check_literal(self, ctx: 'Context', parent: Optional['Node']) -> None:
        self.in_param = self.in_param.type_check(ctx, parent)
        self.out_param = self.out_param.type_check(ctx, parent)
        self.body = self.body.type_check(ctx, parent)
        if self.in_param is None or self.out_param is None or self.body is None:
            return

        self.literal_type = FunctionType(self.in_param.type, self.out_param.type)
        return

    def __str__(self):
        return f'{self.in_param} -> {self.out_param} {self.body}'

    def __repr__(self):
        return f'<{self.in_param.__repr__()} -> {self.out_param.__repr__()} {self.body.__repr__()}>'


class ProductTypeLiteral(Literal):
    def __init__(self, values: list['TypeDefineNode']):
        assert len(values) > 0
        super().__init__(values, None, values[-1].location - values[0].location)
        return

    def type_check_literal(self, ctx: 'Context', parent: Optional['Node']) -> None:
        types = []
        for i, val in enumerate(self.value):
            self.value[i] = val.type_check(ctx, parent)
            if self.value[i] is not None:
                types.append(self.value[i].type)

        self.literal_type = ProductType(types)
        return

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
            self.namespace: dict[str, NameDefineNode] = {}
            self.scope_namespace: set[str] = set()
            self.types: set[TypeDefineNode] = set()
            self.funcs: set[FunctionLiteral] = set()
            self.errors: list[Error] = []
        else:
            self.scope_level: int = up_scope.scope_level + 1
            self.namespace: dict[str, NameDefineNode] = copy(up_scope.namespace)
            self.scope_namespace: set[str] = set()
            self.types: set[TypeDefineNode] = up_scope.types
            self.funcs: set[FunctionLiteral] = up_scope.funcs
            self.errors: list[Error] = up_scope.errors    # not a copy. All errors will go to the same collection
        return

    def get_definition(self, name: str) -> Optional['NameDefineNode']:
        """Returns the definition of the given name in the current context or None if it is not defined"""
        if name in self.namespace:
            return self.namespace[name]
        else:
            return None

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
    def __init__(self, location: FileRange, parent_node=None):
        self.parent: Optional[Node] = parent_node
        self.context: Optional[Context] = None
        self.location: FileRange = location
        return

    def update(self, ctx: Context, parent: 'Node') -> None:
        """Builds up the context of the node, and allows the node to change itself if needed"""
        self.context = ctx
        self.parent = parent
        return

    def type_check(self, ctx: Context, parent: Optional['Node']) -> Optional['Node']:
        """Updates current node and recursively calls type_check on all the children nodes and compares its type with them.
        When updating the node may return a different node, so the parent node must update its children with the returned node.
        If there is a type mismatch, it will add an error to the context"""
        self.update(ctx, parent)
        return self

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


class StatementNode(Node):
    def get_type(self) -> Optional[Type]:
        return VoidType()


class ExpressionNode(StatementNode):
    def __init__(self, location: FileRange):
        super().__init__(location)
        self.type: Optional[Type] = None
        return

    def get_typedef(self) -> Optional['TypeDefineNode']:
        """Gets the type definition of the type of the Node"""
        return self.context.get_definition(self.type.name_tok.value)

    def update_type(self, ctx: Context):
        t = ctx.get_definition(self.type.name_tok.value)
        if t is not None:
            self.type = t.type

        return

    def get_size(self) -> int:
        return self.type.size


class NameDefineNode(StatementNode):
    def __init__(self, name: 'VariableNode', type: Optional[Type] = None):
        super().__init__(name.location)
        self.type: Optional[Type] = type
        self.name: VariableNode = name
        return

    def get_id(self) -> str:
        """Returns the identifier of the Definition node to be used as key in the dictionary of namespace"""
        return self.name.name_tok.value

    def add_ctx(self, ctx: Context) -> None:
        """Adds the definition to the current context dict"""
        if self.name.name_tok.value in ctx.scope_namespace:
            ctx.error(TypeError.shadow_same_scope, self.name.name_tok.value)
            return None
        ctx.namespace[self.get_id()] = self

        return

    def update(self, ctx: Context, parent: 'Node') -> None:
        self.context = ctx
        self.parent = parent
        self.add_ctx(ctx)
        return

    def get_size(self):
        return self.type.size


# Base NODES

class ValueNode(ExpressionNode):
    def __init__(self, tok: Literal):
        super().__init__(tok.location)
        self.value: Literal = tok
        return

    def type_check(self, ctx: Context, parent: Optional[Node]) -> Optional['ValueNode']:
        self.update(ctx, parent)
        self.value.type_check_literal(ctx, self)  # mutates itself not returns new node
        self.type = self.value.literal_type
        return self

    def __str__(self):
        return f'{self.value}'

    def __repr__(self):
        return f'<{self.value.__repr__()}>'


class VariableNode(ExpressionNode):
    def __init__(self, repr_tok: Token):
        super().__init__(repr_tok.location)
        self.name_tok: Token = repr_tok
        return

    def type_check(self, ctx: Context, parent: Optional['Node']) -> Optional['VariableNode']:
        """Recursively calls type_check on all the children nodes and compares its type with them.
        If there is a type mismatch, it will add an error to the context"""
        self.update(ctx, parent)
        var = self.context.get_definition(self.name_tok.value)
        if var is None:
            self.error(TypeError.unk_var, self.name_tok.value)
            return None
        self.type = var.type
        return self

    def __str__(self):
        return f'{self.name_tok.value}'

    def __repr__(self):
        return f'<{self.name_tok.value.__repr__()}>'


class ScopeNode(StatementNode):
    def __init__(self, child_nodes: list[Node], location):
        super().__init__(location)
        self.child_nodes: list[Node] = child_nodes
        return

    def process_body(self) -> None:
        ctx = self.context.clone()
        child_nodes = []
        for node in self.child_nodes:
            node = node.type_check(ctx, self)
            if node is not None:
                child_nodes.append(node)

        self.child_nodes = child_nodes
        return

    def type_check(self, ctx: Context, parent: Optional['Node']) -> Optional['ScopeNode']:
        """Recursively calls type_check on all the children nodes and compares its type with them.
        If there is a type mismatch, it will add an error to the context"""
        self.update(ctx, parent)
        self.process_body()
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

class AssignNode(ExpressionNode):
    def __init__(self, var: ExpressionNode, value: ExpressionNode):
        super().__init__(var.location)
        self.var: ExpressionNode = var
        self.value: ExpressionNode = value
        return

    def type_check(self, ctx: Context, parent: Optional['Node']) -> Optional[Node]:
        self.update(ctx, parent)
        self.var = self.var.type_check(ctx, self)
        self.value = self.value.type_check(ctx, self)

        if self.var is None or self.value is None:
            return None

        return self

    def __str__(self):
        return f'{self.var} = {self.value}'

    def __repr__(self):
        return f'<{self.var.__repr__()} = {self.value.__repr__()}>'


class UnOpNode(ExpressionNode):
    def __init__(self, op: Token, child: ExpressionNode):
        super().__init__(op.location)
        self.op = op
        self.child = child
        return

    def type_check(self, ctx: Context, parent: Optional['Node']) -> Optional['FuncCallNode']:
        self.update(ctx, parent)
        self.child = self.child.type_check(ctx, self)
        if self.child is None:
            return None

        assert self.op.value in dunder_funcs
        func_name = dunder_funcs[self.op.value]
        return FuncCallNode(VariableNode(func_name), [self.child]).type_check(ctx, parent)    # TODO: add edge cases for the primitives (they dont need to be converted to function calls)

    def __str__(self):
        return f'{self.op.value} {self.child}'

    def __repr__(self):
        return f'<{self.op.value} {self.child.__repr__()}>'


class BinOpNode(ExpressionNode):
    def __init__(self, op: Token, left: ExpressionNode, right: ExpressionNode):
        super().__init__(op.location)
        self.left_child = left
        self.right_child = right
        self.op = op
        return

    def type_check(self, ctx: Context, parent: Optional['Node']) -> Optional['FuncCallNode']:
        self.update(ctx, parent)
        self.left_child = self.left_child.type_check(ctx, self)

        self.right_child = self.right_child.type_check(ctx, self)
        if self.left_child is None or self.right_child is None:
            return None

        assert self.op.value in dunder_funcs
        func_name = dunder_funcs[self.op.value]
        return FuncCallNode(VariableNode(func_name), [self.left_child, self.right_child]).type_check(ctx, parent)

    def __str__(self):
        return f'{self.left_child} {self.op.value} {self.right_child}'

    def __repr__(self):
        return f'<{self.left_child.__repr__()} {self.op.value} {self.right_child.__repr__()}>'


class DotOperatorNode(VariableNode):
    def __init__(self, repr_tok: Token, var: ExpressionNode, field: VariableNode):
        super().__init__(repr_tok)
        self.var: ExpressionNode = var
        self.field: VariableNode = field
        self.attribute: Optional[NameDefineNode] = None
        return

    def type_check(self, ctx: Context, parent: Optional['Node']) -> Optional['DotOperatorNode']:
        self.update(ctx, parent)
        self.var = self.var.type_check(ctx, self)
        if self.var is None:
            return None

        typedef = self.var.get_typedef()
        if typedef is None:
            assert False    # if the type is not visible then it should have been caught in the previous step

        self.attribute = typedef.get_attribute(self.field.name_tok.value)
        if self.attribute is None:
            ctx.error(TypeError.no_attribute, typedef.name.name_tok.value, self.field.name_tok.value)
            return None

        self.type = self.attribute.type

        return self

    def __str__(self):
        return f'{self.var}.{self.field.name_tok.value}'

    def __repr__(self):
        return f'<{self.var.__repr__()}.{self.field.__repr__()}>'


class IndexOperatorNode(VariableNode):
    def __init__(self, repr_tok: Token, collection: ExpressionNode, index: ExpressionNode):
        super().__init__(repr_tok)
        self.collection: ExpressionNode = collection
        self.index: ExpressionNode = index
        return

    def type_check(self, ctx: Context, parent: Optional['Node']) -> Optional['IndexOperatorNode']:
        self.update(ctx, parent)
        self.collection = self.collection.type_check(ctx, self)
        self.index = self.index.type_check(ctx, self)

        if self.collection is None or self.index is None:
            return None

        assert Operators.index.value.value in dunder_funcs
        fn_name = dunder_funcs[Operators.index.value.value]
        return FuncCallNode(VariableNode(fn_name), [self.collection, self.index]).type_check(ctx, parent)

    def __str__(self):
        return f'{self.collection}[{self.index}]'

    def __repr__(self):
        return f'<{self.collection.__repr__()}[{self.index.__repr__()}]>'


class FuncCallNode(ExpressionNode):
    def __init__(self, func: ExpressionNode, args: list[ExpressionNode]):
        super().__init__(func.location)
        self.func: ExpressionNode = func
        self.args: list[ExpressionNode] = args
        self.func_literal: Optional[FunctionLiteral] = None
        return

    def type_check(self, ctx: Context, parent: Optional['Node']) -> Optional['FuncCallNode']:
        self.update(ctx, parent)
        self.func.type_check(ctx, self)

        self.args = [arg.type_check(ctx, self) for arg in self.args]
        if self.func is None or None in self.args:
            return None

        assert isinstance(self.func.type, FunctionType)

        if isinstance(self.func.type.arg, TupleType):
            in_type = tuple_type_unpacking(ctx, self.func.type.arg, FunctionType, TypeError.wrong_type)
        else:
            in_type = [self.func.type.arg]

        in_func_type = TupleType([arg.type for arg in self.args])

        if in_type != in_func_type.types:
            ctx.error(TypeError.wrong_func_type, self.func, in_func_type, self.func.type.arg)

        return self

    def __str__(self):
        return f'{self.func}{self.args}'

    def __repr__(self):
        return f'<{self.func.__repr__()}({", ".join([f"{arg.__repr__()}" for arg in self.args])})>'


# Definition Nodes

class VarDefineNode(NameDefineNode, ExpressionNode):
    def __init__(self, var_name: VariableNode, var_type: Type):
        super().__init__(var_name, var_type)
        self.class_def: Optional[TypeDefineNode] = None
        return

    def type_check(self, ctx: Context, parent: Optional['Node']) -> Optional['VarDefineNode']:
        self.update(ctx, parent)

        self.class_def = self.context.get_definition(self.name.name_tok.value)
        if self.class_def is None:
            self.error(TypeError.unk_var, self.name.name_tok.value)
            return None

        self.name = self.name.type_check(ctx, self.parent)
        if self.name is None:
            return None

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

    def type_check(self, ctx: Context, parent: Optional['Node']) -> Optional['TypeDefineNode']:
        self.update(ctx, parent)

        self.generics = [g.type_check(ctx, self) for g in self.generics]
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
        typedef = self.context.get_definition(self.other_type.name_tok.value)
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

class IfNode(StatementNode):
    def __init__(self, repr_tok: Token, condition: ExpressionNode, body: Node):
        super().__init__(repr_tok.location)
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


class ElseNode(StatementNode):
    def __init__(self, repr_tok: Token, body: Node, if_statement: Optional[IfNode]):
        super().__init__(repr_tok.location)
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


class WhileNode(StatementNode):
    def __init__(self, repr_tok: Token, condition: ExpressionNode, body: Node):
        super().__init__(repr_tok.location)
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


class LoopModifierNode(StatementNode):
    def __init__(self, repr_tok: Token, value: Optional[Token], error: TypeError):
        super().__init__(repr_tok.location)
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


class MatchNode(StatementNode):
    def __init__(self, repr_tok: Token, value: ExpressionNode, cases: list['CaseNode']):
        assert len(cases) > 0
        super().__init__(cases[-1].location - repr_tok.location)
        self.value: ExpressionNode = value
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


class CaseNode(StatementNode):
    def __init__(self, variant: VariableNode, body: Node):
        super().__init__(body.location - variant.location)
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
    Operators.b_not.value.value: Token(TT.IDENTIFIER, '__bnot__'),
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
    Operators.b_and.value.value: Token(TT.IDENTIFIER, '__band__'),
    Operators.b_xor.value.value: Token(TT.IDENTIFIER, '__bxor__'),
    Operators.b_or.value.value: Token(TT.IDENTIFIER, '__bor__'),
    Operators.and_.value.value: Token(TT.IDENTIFIER, '__and__'),
    Operators.or_.value.value: Token(TT.IDENTIFIER, '__or__'),
    Operators.index.value.value: Token(TT.IDENTIFIER, '__get__'),
}
