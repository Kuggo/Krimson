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

    # Compile_time detected runtime errors
    out_of_bounds = 'Index is out of bounds'
    no_element = 'No element with such key'


# Literals directly supported by compiler

class TupleLiteral(Literal):
    def __init__(self, values: list['ExpressionNode']):
        assert len(values) > 0
        super().__init__(values, None, values[-1].location - values[0].location)
        return

    @staticmethod
    def gen_type(values) -> TupleType:
        types = []
        for value in values:
            types.append(value.type)
        return TupleType(types)

    def __repr__(self):
        return f'({", ".join([str(v) for v in self.value])})'


class VoidLiteral(Literal):
    def __init__(self, location: FileRange):
        super().__init__('void', VoidType(), location)
        return

    def __repr__(self):
        return f'void'


class ArrayLiteral(Literal):
    def __init__(self, values: list):
        assert len(values) > 0
        super().__init__([], None, values[-1].location - values[0].location)
        self.values: list = values
        return

    def __repr__(self):
        return f'[{", ".join([str(v) for v in self.values])}]'


class FunctionLiteral(Literal):
    def __init__(self, in_param: 'ExpressionNode', out_param: 'ExpressionNode', body: 'Node'):
        super().__init__(None, None)
        self.in_param: 'ExpressionNode' = in_param
        self.out_param: 'ExpressionNode' = out_param

        self.in_param_list: list[NameDefineNode] = []
        self.out_param_list: list[NameDefineNode] = []
        self.body: Node = body
        # self.literal_type: FunctionType = FunctionType(in_param.type, out_param.type) # TODO this must be done in update phase
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

    def __repr__(self):
        return f'{self.in_param} -> {self.out_param} {self.body}'


class ProductTypeLiteral(Literal):
    def __init__(self, values: list['TypeDefineNode']):
        assert len(values) > 0
        super().__init__([], None, values[-1].location - values[0].location)
        self.values: list = values
        return



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
            self.funcs: set[FuncDefineNode] = set()
            self.errors: list[Error] = []
        else:
            self.scope_level: int = up_scope.scope_level + 1
            self.namespace: dict[str, NameDefineNode] = copy(up_scope.namespace)
            self.scope_namespace: set[str] = set()
            self.types: set[TypeDefineNode] = up_scope.types
            self.funcs: set[FuncDefineNode] = up_scope.funcs
            self.errors: list[Error] = up_scope.errors    # not a copy. All errors will go to the same collection
        return

    def get_definition(self, name: str) -> Optional['NameDefineNode']:
        if name in self.namespace:
            return self.namespace[name]
        else:
            return None

    def clone(self) -> 'Context':
        c = Context(self.global_vars, self)
        return c

    def error(self, e: TypeError, node: 'Node', *args) -> None:
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

    def update(self, ctx: Context, parent: 'Node') -> Optional['Node']:
        """Builds up the context of the node, and allows the node to change itself if needed"""
        self.context = ctx
        self.parent = parent
        return self

    def type_check(self) -> None:
        """Recursively calls type_check on all the children nodes and compares its type with them.
        If there is a type mismatch, it will add an error to the context"""
        pass

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

    def update_type(self, ctx: Context):
        t = ctx.get_definition(self.type.name.value)
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
        return self.name.var_name.value

    def add_ctx(self, ctx: Context) -> None:
        """Adds the definition to the current context dict"""
        if self.name.var_name.value in ctx.scope_namespace:
            ctx.error(TypeError.shadow_same_scope, self.name.var_name.value)
            return None
        ctx.namespace[self.get_id()] = self

        return

    def update(self, ctx: Context, parent: Optional[Node]) -> Optional['NameDefineNode']:
        self.context = ctx
        self.parent = parent
        self.add_ctx(ctx)
        return self

    def get_size(self):
        return self.type.size


# Base NODES

class ValueNode(ExpressionNode):
    def __init__(self, tok: Literal):
        super().__init__(tok.location)
        self.value: Literal = tok
        self.type: Type = tok.literal_type
        return

    def __repr__(self):
        return f'{self.value}'


class VariableNode(ExpressionNode):
    def __init__(self, repr_tok: Token):
        super().__init__(repr_tok.location)
        self.var_name: Token = repr_tok
        return

    def type_check(self) -> None:
        """Recursively calls type_check on all the children nodes and compares its type with them.
        If there is a type mismatch, it will add an error to the context"""
        var = self.context.get_definition(self.var_name.value)
        if var is None:
            self.error(TypeError.unk_var, self.var_name.value)
            return
        self.type = var.type
        return

    def __repr__(self):
        return f'{self.var_name.value}'


class ScopeNode(StatementNode):
    def __init__(self, child_nodes: list[Node], location):
        super().__init__(location)
        self.child_nodes: list[Node] = child_nodes
        return

    def process_body(self, ctx: Context) -> None:
        for node in self.child_nodes:
            if isinstance(node, NameDefineNode):
                node.add_ctx(ctx)

        child_nodes = []
        for node in self.child_nodes:
            node = node.update(ctx, self.parent)
            if node is not None:
                child_nodes.append(node)

        self.child_nodes = child_nodes
        return

    def update(self, ctx: Context, parent: Optional[Node]) -> Optional['ScopeNode']:
        self.context = ctx
        self.parent = parent
        new_ctx = ctx.clone()
        self.process_body(new_ctx)
        return self

    def type_check(self) -> None:
        """Recursively calls type_check on all the children nodes and compares its type with them.
        If there is a type mismatch, it will add an error to the context"""
        for node in self.child_nodes:
            node.type_check()
        return

    def __repr__(self):
        string = f'\n{self.context.scope_level * "  "}{{\n'
        for node in self.child_nodes:
            string += f'{node.context.scope_level * "  "}{node.__repr__()}\n'
        return string + f'{self.context.scope_level * "  "}}}'


# Operation Nodes

class AssignNode(ExpressionNode):
    def __init__(self, var: ExpressionNode, value: ExpressionNode):
        super().__init__(var.location)
        self.var: ExpressionNode = var
        self.value: ExpressionNode = value
        return

    def update(self, ctx: Context, parent: Optional[Node]) -> Optional['ExpressionNode']:
        self.context = ctx
        self.parent = parent

        self.var = self.var.update(ctx, self)
        if self.var is None:
            return None

        self.value = self.value.update(ctx, self)
        if self.value is None:
            return None

        return self

    def __repr__(self):
        return f'{self.var} = {self.value}'


class UnOpNode(ExpressionNode):
    def __init__(self, op: Token, child: ExpressionNode):
        super().__init__(op.location)
        self.op = op
        self.child = child
        return

    def update(self, ctx: Context, parent: Optional[Node]) -> Optional['FuncCallNode']:
        self.context = ctx
        self.parent = parent
        self.child = self.child.update(ctx, self)
        if self.child is None:
            return None

        if self.op.value in dunder_funcs:
            func_name = dunder_funcs[self.op.value]
            return FuncCallNode(VariableNode(func_name), [self.child]).update(ctx, self)
        else:
            assert False

    def __repr__(self):
        return f'{self.op.value} {self.child}'


class BinOpNode(ExpressionNode):
    def __init__(self, op: Token, left: ExpressionNode, right: ExpressionNode):
        super().__init__(op.location)
        self.left_child = left
        self.right_child = right
        self.op = op
        return

    def update(self, ctx: Context, parent: Optional[Node]) -> Optional['FuncCallNode']:
        self.context = ctx
        self.parent = parent
        self.left_child = self.left_child.update(ctx, self)
        self.right_child = self.right_child.update(ctx, self)

        if self.left_child is None or self.right_child is None:
            return None

        if self.op.value in dunder_funcs:
            fn_name = dunder_funcs[self.op.value]
            return FuncCallNode(VariableNode(fn_name), [self.left_child, self.right_child]).update(ctx, self)
        else:
            assert False

    def __repr__(self):
        return f'{self.left_child} {self.op.value} {self.right_child}'


class DotOperatorNode(VariableNode):
    def __init__(self, repr_tok: Token, var: ExpressionNode, field: VariableNode):
        super().__init__(repr_tok)
        self.var: ExpressionNode = var
        self.field: VariableNode = field
        return

    def update(self, ctx: Context, parent: Optional[Node]) -> Optional['DotOperatorNode']:
        self.context = ctx
        self.parent = parent

        self.var = self.var.update(ctx, self)
        if self.var is None:
            return None

        return self

    def __repr__(self):
        return f'{self.var}.{self.field.var_name.value}'


class IndexOperatorNode(VariableNode):
    def __init__(self, repr_tok: Token, collection: ExpressionNode, index: ExpressionNode):
        super().__init__(repr_tok)
        self.collection: ExpressionNode = collection
        self.index: ExpressionNode = index
        return

    def update(self, ctx: Context, parent: Optional[Node]) -> Optional['IndexOperatorNode']:
        self.context = ctx
        self.parent = parent
        self.collection = self.collection.update(ctx, self)
        self.index = self.index.update(ctx, self)

        if self.collection is None or self.index is None:
            return None

        #if use_dunder_func:
        #    fn_name = dunder_funcs[Operators.index.value.value]
        #    return FuncCallNode(VariableNode(fn_name), (self.collection, self.index)).update(ctx, self)

        return self

    def __repr__(self):
        return f'{self.collection}[{self.index}]'


class FuncCallNode(ExpressionNode):
    def __init__(self, func_name: ExpressionNode, args: list[ExpressionNode]):
        super().__init__(func_name.location)
        self.func_name: ExpressionNode = func_name
        self.args: list[ExpressionNode] = args
        self.func: Optional[FuncDefineNode] = None

    def update(self, ctx: Context, parent: Optional[Node]) -> Optional['FuncCallNode']:
        self.context = ctx
        self.parent = parent
        self.func_name.update(ctx, self)
        if self.func_name is None:
            return None

        self.args = [arg.update(ctx, self) for arg in self.args]
        if None in self.args:
            return None

        return self

    def __repr__(self):
        return f'{self.func_name}{self.args}'


# Definition Nodes

class VarDefineNode(NameDefineNode, ExpressionNode):
    def __init__(self, var_name: VariableNode, var_type: Type, value: Optional[ExpressionNode] = None):
        super().__init__(var_name, var_type)
        self.value: Optional[ExpressionNode] = value
        self.class_def: Optional[TypeDefineNode] = None
        return

    def update(self, ctx: Context, parent: Optional[Node]) -> Optional['VarDefineNode']:
        self.context = ctx
        self.parent = parent

        self.add_ctx(ctx)
        self.name = self.name.update(ctx, self.parent)
        if self.name is None:
            return None

        if self.value is not None:
            self.value = self.value.update(ctx, self.parent)
            if self.value is not None:
                self.name.type = self.type = self.value.type

        return self

    def __repr__(self):
        if self.value is None:
            return f'{self.name.var_name.value}: {self.type}'
        else:
            return f'{self.name.var_name.value}: {self.type} = {self.value}'


class FuncDefineNode(NameDefineNode):
    def __init__(self, func_name: VariableNode, func: 'FunctionLiteral'):
        super().__init__(func_name)
        self.func: FunctionLiteral = func
        self.body: Node = func.body
        return

    def add_ctx(self, ctx: Context) -> None:
        """Adds the definition to the current context dict"""
        super().add_ctx(ctx)
        ctx.funcs.add(self)
        return

    def update(self, ctx: Context, parent: Optional[Node]) -> Optional['FuncDefineNode']:
        self.context = ctx
        self.parent = parent
        self.body = self.body.update(ctx, self)
        if self.body is None:
            return None
        return self

    def __repr__(self):
        return f'{self.name}: {self.func.literal_type} = {self.func}'


# Type Nodes

class TypeDefineNode(NameDefineNode):
    def __init__(self, name: VariableNode, generics: Optional[list[VarDefineNode | VariableNode | ValueNode]] = None):
        super().__init__(name)
        self.generics: Optional[list[VarDefineNode | VariableNode | ValueNode]] = generics if generics is not None else []
        return

    @staticmethod
    def make_type(name: Token, fields: list[NameDefineNode]) -> Type:
        types = []
        for field in fields:
            types.append(field.type)
        return TypeDefType(name, types)

    def add_ctx(self, ctx: Context) -> None:
        """Adds the definition to the current context dict"""
        super().add_ctx(ctx)
        ctx.types.add(self)
        return

    def update(self, ctx: Context, parent: Optional[Node]) -> Optional['TypeDefineNode']:
        self.context = ctx
        self.parent = parent

        return self

    def __repr__(self):
        return f'type {self.name} {self.type}'


class ProductTypeDefineNode(TypeDefineNode):
    def __init__(self, name: VariableNode, fields: list[NameDefineNode], generics = None):
        super().__init__(name, generics)
        self.fields: list[NameDefineNode] = fields
        return

    def get_field(self, field_name: str) -> Optional[NameDefineNode]:
        for field in self.fields:
            if field.name.var_name == field_name:
                return field
        return None

    def __repr__(self):
        string = "\n".join([f'{field}' for field in self.fields])
        return f'type {self.name} {self.type} {{\n{string}\n}}'


class TypeAliasDefineNode(TypeDefineNode):
    def __init__(self, name: VariableNode, other_type: Type, generics = None):
        super().__init__(name, generics)
        self.other_type: Type = other_type
        return

    def __repr__(self):
        return f'type {self.name} = {self.other_type}'


class SumTypeDefineNode(TypeDefineNode):
    def __init__(self, name: VariableNode, subtypes: list[TypeDefineNode], generics = None):
        super().__init__(name, generics)
        self.name: VariableNode = name
        self.variants: list[TypeDefineNode] = subtypes
        self.type: Type = self.make_type(name, subtypes)
        return

    @staticmethod
    def make_type(name: VariableNode, subtypes: list[TypeDefineNode]) -> Type:
        return SumType(name.var_name, subtypes)

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
