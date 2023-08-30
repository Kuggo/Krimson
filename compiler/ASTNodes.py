from copy import copy, deepcopy
from Instructions import *


# helper functions
pass


# Errors

class TypeError(Enum):
    no_attribute = 'type "{}" has no attribute "{}"'
    unk_var = 'Undefined variable "{}"'
    unk_func = 'Function "{}" not defined or visible in scope'
    undefined_function = 'Undefined function for the given args'
    def_statement_expected = 'Declaration statement expected inside a class body'
    break_not_in_loop = 'break keyword found outside the specified loop body depth'
    skip_not_in_loop = 'skip keyword found outside the specified loop body depth'
    pos_int_expected = 'Positive integer expected'

    # Compile_time detected runtime errors
    out_of_bounds = 'Index is out of bounds'
    no_element = 'No element with such key'


# Literals directly supported by compiler

class TupleLiteral(Literal):
    def __init__(self, values: list):
        assert len(values) > 0
        super().__init__([], None, values[-1].location - values[0].location)
        self.values: list = values
        return


class ArrayLiteral(Literal):
    def __init__(self, values: list):
        assert len(values) > 0
        super().__init__([], None, values[-1].location - values[0].location)
        self.values: list = values
        return


class FunctionLiteral(Literal):
    def __init__(self, in_param, out_param, body):
        super().__init__(None, None)
        self.in_param = in_param
        self.out_param = out_param
        self.body = body
        return


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
            self.types: set[TypeDefineNode] = set()
            self.funcs: set[FuncDefineNode] = set()
            self.errors: list[Error] = []
        else:
            self.scope_level: int = up_scope.scope_level + 1
            self.namespace: dict[str, NameDefineNode] = copy(up_scope.namespace)
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
        self.up_scope = c
        return c

    def error(self, e: TypeError, node: 'Node', *args) -> None:
        self.errors.append(Error(e, node.location, self.global_vars, *args))
        return

    def __repr__(self):
        return f'vars: {self.namespace}\nfuncs: {self.funcs}\ntypes: {self.types}'


# Category Nodes (DO NOT construct these nodes directly!)

class Node:
    def __init__(self, location: FileRange, parent_node=None):
        self.parent: Optional[ScopeNode] = parent_node
        self.context: Optional[Context] = None
        self.location: FileRange = location
        return

    def update(self, ctx: Context, parent: 'Node') -> Optional['Node']:
        self.context = ctx.scope_level
        self.parent = parent
        return self

    def __repr__(self):
        pass


class ExpressionNode(Node):
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


class NameDefineNode(Node):
    def __init__(self, name: 'VariableNode', type: Optional[Type] = None):
        super().__init__(name.location)
        self.type: Optional[Type] = type
        self.name: VariableNode = name
        return

    def get_id(self) -> str:
        """Returns the identifier of the Definition node to be used as key in the dictionary of namespace"""
        return self.name.name.value

    def add_ctx(self, ctx: Context) -> None:
        """Adds the definition to the current context dict"""
        pass

    def update(self, ctx: Context, parent: Optional[Node]) -> Optional['NameDefineNode']:
        self.context = ctx
        self.parent = parent
        self.add_ctx(ctx)
        return self

    def get_size(self):
        return self.type.size


# Base NODES

class ValueNode(ExpressionNode):
    def __init__(self, tok: Token):
        super().__init__(tok.location)
        self.value: Token = tok
        return

    def update(self, ctx: Context, parent: Optional[Node]) -> Optional['ValueNode']:
        self.context = ctx
        self.parent = parent
        self.type = self.convert_py_type()
        return self

    def get_size(self) -> int:
        if isinstance(self.value.value, list):
            size = 0
            for item in self.value.value:
                size += item.get_size()
            return size
        else:
            return 1    # primitive size is 1

    def convert_py_type(self) -> 'Type':
        if isinstance(self.value, Literal):
            self.value.literal_type.size = self.get_size()
            return self.value.literal_type
        else:
            assert False

    def gen_ir(self, left=True) -> list[Instruction]:
        ir = []

        if isinstance(self.value, Literal):
            if self.value.literal_type == Types.array.value or self.value.literal_type == Types.str.value:
                for item in reversed(self.value.value):
                    # TODO: make a method for copying multiword values, and pushing them to the stack too
                    i: Instruction = copy(Operations.imm_a.value)
                    i.imm = item
                    ir.append(i)

            elif self.value.literal_type.size == 1:
                if left:
                    i: Instruction = copy(Operations.imm_a.value)
                else:
                    i: Instruction = copy(Operations.imm_b.value)
                i.imm = self.value.value
                ir.append(i)
            else:
                assert False

        else:
            assert False

        return ir

    def __repr__(self):
        return f'{self.value.value}'


class VariableNode(ExpressionNode):
    def __init__(self, repr_tok: Token):
        super().__init__(repr_tok.location)
        self.name: Token = repr_tok
        return

    def update(self, ctx: Context, parent: Optional[Node], use_dunder_func=True, func=False) -> Optional['VariableNode']:
        self.context = ctx
        self.parent = parent
        var = ctx.get_definition(self.name.value)
        if isinstance(var, VarDefineNode):
            self.type = var.type
            # self.offset = OffsetNode(Registers.BP, 0)

        elif isinstance(var, MacroDefineNode):
            value = copy(var.value)
            return value.update(ctx, self.parent)

        elif isinstance(var, TypeDefineNode):
            self.type = Types.type

        elif isinstance(var, FuncDefineNode):
            self.type = var.type

        else:
            if not func:
                ctx.error(TypeError.unk_var, self, self.name.value)
            return None
        return self

    def __repr__(self):
        return f'{self.name}'


class ScopeNode(Node):
    def __init__(self, child_nodes: list[Node], location):
        super().__init__(location)
        self.child_nodes: list[Node] = child_nodes
        return

    def process_body(self, ctx: Context) -> None:
        for node in self.child_nodes:
            if isinstance(node, NameDefineNode) and not isinstance(node, VarDefineNode):
                node.update(ctx, self.parent)

        child_nodes = []
        for node in self.child_nodes:
            if isinstance(node, NameDefineNode) and not isinstance(node, VarDefineNode):
                continue

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

    def __repr__(self):
        string = f'\n{self.context.scope_level * "  "}{{\n'
        for node in self.child_nodes:
            string += f'{node.context.scope_level * "  "}{node.__repr__()}\n'
        return string + f'{self.context.scope_level * "  "}}}'


# Operation Nodes

class AssignNode(ExpressionNode):
    def __init__(self, var: VariableNode, value: ExpressionNode):
        super().__init__(var.location)
        self.var: VariableNode = var
        self.value: ExpressionNode = value
        return

    def update(self, ctx: Context, parent: Optional[Node]) -> Optional['ExpressionNode']:
        self.context = ctx
        self.parent = parent

        return self

    def __repr__(self):
        return f'{self.var.name.value} = {self.value}'


class UnOpNode(ExpressionNode):
    def __init__(self, op: Token, child: ExpressionNode):
        super().__init__(op.location)
        self.op = op
        self.child = child
        return

    def update(self, ctx: Context, parent: Optional[Node]) -> Optional['FuncCallNode']:
        self.context = ctx
        self.parent = parent
        self.child = self.child.update(ctx, self.parent)
        if self.child is None:
            return None

        if self.op.value in dunder_funcs:
            func_name = dunder_funcs[self.op.value]
            return FuncCallNode(VariableNode(func_name), (self.child,)).update(ctx, self.parent)
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
        self.left_child = self.left_child.update(ctx, self.parent)
        self.right_child = self.right_child.update(ctx, self.parent)

        if self.left_child is None or self.right_child is None:
            return None

        if self.op.value in dunder_funcs:
            fn_name = dunder_funcs[self.op.value]
            return FuncCallNode(VariableNode(fn_name), (self.left_child, self.right_child)).update(ctx, self.parent)
        else:
            assert False

    def __repr__(self):
        return f'{self.left_child} {self.op.value} {self.right_child}'


class DotOperatorNode(VariableNode):
    def __init__(self, repr_tok: Token, var: ExpressionNode, field: VariableNode):
        super().__init__(repr_tok)
        self.name = field.name
        self.var: ExpressionNode = var
        self.field: VariableNode = field
        return

    def update(self, ctx: Context, parent: Optional[Node], use_dunder_func=True, func=False) -> Optional['DotOperatorNode']:
        self.context = ctx
        self.parent = parent
        return

    def __repr__(self):
        return f'{self.var}.{self.field}'


class IndexOperatorNode(VariableNode):
    def __init__(self, repr_tok: Token, collection: ExpressionNode, index: ExpressionNode):
        super().__init__(repr_tok)
        self.collection: ExpressionNode = collection
        self.index: ExpressionNode = index
        return

    def update(self, ctx: Context, parent: Optional[Node], use_dunder_func=True, func=False) -> Optional['IndexOperatorNode']:
        self.context = ctx
        self.parent = parent
        self.collection = self.collection.update(ctx, self.parent)
        self.index = self.index.update(ctx, self.parent)

        if self.collection is None or self.index is None:
            return None

        if use_dunder_func:
            fn_name = dunder_funcs[Operators.index.value.value]
            return FuncCallNode(VariableNode(fn_name), (self.collection, self.index)).update(ctx, self.parent)

        return self

    def __repr__(self):
        return f'{self.collection}[{self.index}]'


class FuncCallNode(ExpressionNode):
    def __init__(self, func_name: VariableNode, args: tuple[ExpressionNode, ...]):
        super().__init__(func_name.location)
        self.func_name: VariableNode = func_name
        self.args: tuple[ExpressionNode, ...] = args
        self.func: Optional[FuncDefineNode] = None

    def update(self, ctx: Context, parent: Optional[Node]) -> Optional['FuncCallNode']:
        self.context = ctx
        self.parent = parent
        self.func_name.update(ctx, self.parent, func=True)
        return self

    def __repr__(self):
        return f'{self.func_name}{self.args}'


# Definition Nodes

class MacroDefineNode(NameDefineNode):
    def __init__(self, name: 'VariableNode', expression: ExpressionNode):
        super().__init__(name)
        self.value: ExpressionNode = expression

    def __repr__(self):
        return f'(macro) {self.name} = {self.value}'


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
        """if self.value is None:
            return f'loc:({self.offset}) {self.name.repr_token.value}: {self.type}'
        else:
            return f'loc:({self.offset}) {self.name.repr_token.value}: {self.type} = {self.value}'"""
        if self.value is None:
            return f'loc:(TODO) {self.name.name.value}: {self.type}'
        else:
            return f'loc:(TODO) {self.name.name.value}: {self.type} = {self.value}'


class FuncDefineNode(NameDefineNode):
    def __init__(self, func_name: VariableNode, body: 'IsolatedScopeNode'):
        super().__init__(func_name)
        self.func_type = None
        self.params: tuple[NameDefineNode, ...] = tuple()
        self.ret_param = None
        self.body: IsolatedScopeNode = body
        return

    def get_func_label(self) -> str:
        string = self.name.name.value
        for param in self.params:
            string += f'.{param.type.get_type_label()}'
        return string

    def update(self, ctx: Context, parent: Optional[Node]) -> Optional['FuncDefineNode']:
        self.context = ctx
        self.parent = parent

        return self

    def inline_func(self, func_call: FuncCallNode) -> ScopeNode:
        pass

    def __repr__(self):
        if len(self.params) == 0:
            return f'{self.name}: {self.type} = null -> {self.ret_param} {self.body}'
        elif len(self.params) == 1:
            return f'{self.name}: {self.type} = {self.params.__repr__()} -> {self.ret_param} {self.body}'
        else:
            return f'{self.name}: {self.type} = ({self.params.__repr__()[1:-1]}) -> {self.ret_param} {self.body}'


class IsolatedScopeNode(ScopeNode):
    def __init__(self, child_nodes: list[Node], location: FileRange):
        super().__init__(child_nodes, location)
        return

    @classmethod
    def new_from_old(cls, old: ScopeNode) -> 'IsolatedScopeNode':
        return cls(old.child_nodes, old.location)

    def update(self, ctx: Context, parent: Optional[Node]) -> Optional['IsolatedScopeNode']:
        self.context = ctx
        self.parent = parent
        self.process_body(ctx)
        return self


class TypeDefineNode(NameDefineNode):
    def __init__(self, name: VariableNode, fields: list[NameDefineNode]):
        super().__init__(name)
        self.fields: list[NameDefineNode] = fields
        return

    @staticmethod
    def make_type(name: Token, fields: list[NameDefineNode]) -> Type:
        types = []
        for field in fields:
            types.append(field.type)
        return TypeDefType(name, types)

    def update(self, ctx: Context, parent: Optional[Node]) -> Optional['TypeDefineNode']:
        self.context = ctx
        self.parent = parent

        return self

    def get_field(self, field_name: str) -> Optional[NameDefineNode]:
        for field in self.fields:
            if field.name.name == field_name:
                return field
        return None

    def __repr__(self):
        string = "\n".join([f'{field}' for field in self.fields])
        return f'type {self.name} {self.type} {{\n{string}\n}}'


class TypeAliasDefineNode(TypeDefineNode):
    def __init__(self, name: VariableNode, other_type: Type):
        super().__init__(name, [])
        self.other_type: Type = other_type
        return

    def __repr__(self):
        return f'type {self.name} = {self.other_type}'


class SumTypeDefineNode(TypeDefineNode):
    def __init__(self, name: VariableNode, subtypes: list[TypeDefineNode]):
        super().__init__(name, [])
        self.name: VariableNode = name
        self.variants: list[TypeDefineNode] = subtypes
        self.type: Type = self.make_type(name, subtypes)
        return

    @staticmethod
    def make_type(name: VariableNode, subtypes: list[TypeDefineNode]) -> Type:
        return SumType(name.name, subtypes)

    def __repr__(self):
        string = ", ".join([f'{subtype}' for subtype in self.variants])
        return f'type {self.name} = {{{string}}}'


# Control Flow

class IfNode(Node):
    def __init__(self, repr_tok: Token, condition: ExpressionNode, body: Node):
        super().__init__(self.location)
        self.condition = condition
        self.body = body
        self.else_statement: Optional[ElseNode] = None
        return

    def update(self, ctx: Context, parent: Optional[Node]) -> Optional['IfNode']:
        self.context = ctx
        self.parent = parent

        self.condition = self.condition.update(ctx, self.parent)

        self.body = self.body.update(ctx, self.parent)
        if self.body is None or self.condition is None:
            return None

        if self.else_statement is not None:
            self.else_statement = self.else_statement.update(ctx, self.parent)

        return self

    def __repr__(self):
        if self.else_statement is None:
            return f'if {self.condition} {self.body}'
        else:
            return f'if {self.condition} {self.body} \n{self.context.scope_level*"  "}{self.else_statement}'


class ElseNode(Node):
    def __init__(self, repr_tok: Token, body: Node, if_statement: Optional[IfNode]):
        super().__init__(self.location)
        self.body = body
        self.if_statement: Optional[IfNode] = if_statement
        return

    def update(self, ctx: Context, parent: Optional[Node]) -> Optional['ElseNode']:
        self.context = ctx
        self.parent = parent

        self.body = self.body.update(ctx, self.parent)
        if self.body is None:
            return None

        return self

    def __repr__(self):
        return f'else {self.body}'


class WhileNode(Node):
    def __init__(self, repr_tok: Token, condition: ExpressionNode, body: Node):
        super().__init__(self.location)
        self.condition = condition
        self.body = body
        return

    def get_loop(self) -> Optional['WhileNode']:
        parent = self.parent
        while not (parent is None or isinstance(parent, WhileNode)):
            parent = parent.parent

        return parent

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
        super().__init__(self.location)
        self.loop: Optional[WhileNode] = None
        self.error: TypeError = error
        if value is None:
            self.value: Optional[Token] = Token(TT.LITERAL, 1)
        else:
            self.value: Optional[Token] = value

    def get_loop(self) -> Optional[WhileNode]:
        parent = self.parent
        while not (parent is None or isinstance(parent, WhileNode)):
            parent = parent.parent

        return parent

    def update(self, ctx: Context, parent: Optional[Node]) -> Optional['LoopModifierNode']:
        self.context = ctx
        self.parent = parent

        if self.value.value <= 0:
            ctx.error(TypeError.pos_int_expected, self)
            self.value.value = 1  # ignore and proceed as default

        i = self.value.value
        self.loop = self
        while i > 0 and self.loop is not None:
            self.loop = self.loop.get_loop()
            i -= 1

        if self.loop is None:
            ctx.error(self.error, self) # maybe improve this error message

        return self


class BreakNode(LoopModifierNode):
    def __init__(self, repr_tok: Token, value: Optional[Token] = None):
        super().__init__(repr_tok, value, TypeError.break_not_in_loop)

    def __repr__(self):
        if self.value is None:
            return f'break'
        else:
            return f'break {self.value.value}'


class SkipNode(LoopModifierNode):
    def __init__(self, repr_tok: Token, value: Optional[Token] = None):
        super().__init__(repr_tok, value, TypeError.skip_not_in_loop)

    def __repr__(self):
        if self.value is None:
            return f'skip'
        else:
            return f'skip {self.value.value}'


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

self_tok = Token(TT.IDENTIFIER, 'self')
constructor_name_tok = Token(TT.IDENTIFIER, 'new')
