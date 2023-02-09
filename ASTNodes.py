from enum import Enum
from Constants import Token, TT, Error, global_vars
from typing import Optional


# helper functions

def convert_py_type(tok: Token) -> 'Type':
    if isinstance(tok, int):
        if tok < 0:
            return Type(Token(TT.IDENTIFIER, 'int'))
        else:
            return Type(Token(TT.IDENTIFIER, 'nat'))
    elif isinstance(tok, list):
        return Type(Token(TT.IDENTIFIER, 'array'))
    elif isinstance(tok, dict):
        return Type(Token(TT.IDENTIFIER, 'dict'))
    elif isinstance(tok, set):
        return Type(Token(TT.IDENTIFIER, 'set'))
    else:
        assert False


# Type

class Type:
    def __init__(self, name: Token, generics: Optional[tuple['Type']] = None):
        self.name: Token = name
        self.generics: Optional[tuple['Type']] = generics
        self.definition: Optional[ClassDefineNode] = None

    def __eq__(self, other: 'Type'):
        return self.name == other.name and self.generics == self.generics

    def is_subtype(self, super_type: 'Type'):
        if self.name == super_type.name:
            return True

        for gen, sup_gen in zip(self.generics, super_type.generics):
            if not gen.is_subtype(sup_gen):
                return False

        return True

    def __repr__(self):
        if self.generics is None:
            return f'{self.name.value}'
        else:
            return f'{self.name.value}[{self.generics.__repr__()[1:-1]}]'


# Errors

class TypeError(Enum):
    pass


# AST level

class Context:
    """Compile Time context to process AST and type check"""
    def __init__(self, down_scope: Optional['Context'] = None):
        self.down_scope: Optional[Context] = down_scope
        self.errors: list[Error] = []

        if down_scope is None:
            self.stack_map: dict[str, int] = {}
            self.funcs: dict[tuple, FuncDefineNode] = {}
            self.types: dict[Type, ClassDefineNode] = {}
            self.vars: dict[str, (MacroDefineNode, VariableNode)] = {}
        else:
            self.stack_map: dict[str, int] = down_scope.stack_map.copy()
            self.funcs: dict[tuple, FuncDefineNode] = down_scope.funcs.copy()
            self.types: dict[Type, ClassDefineNode] = down_scope.types.copy()
            self.vars: dict[str, (MacroDefineNode, VariableNode)] = down_scope.vars.copy()
        return

    def get_var(self, var: 'VariableNode') -> Optional['VarDefineNode']:
        if var.name in self.vars:
            return self.vars[var.name]
        else:
            return None

    def stack_location(self, var: 'VariableNode') -> int:
        if var.name in self.stack_map:
            return self.stack_map[var.name]
        else:
            allocations = set(self.stack_map.values())
            for i, j in enumerate(allocations):
                if i != j:
                    self.stack_map[var.name] = i
                    return i

            self.stack_map[var.name] = len(allocations)
            return self.stack_map[var.name]

    def stack_dealloc(self, var: 'VariableNode') -> None:
        del self.stack_map[var.name]

    def clone(self) -> 'Context':
        c = Context(self)
        self.down_scope = c
        return c

    def error(self, e: TypeError, node: 'Node', *args) -> None:
        self.errors.append(Error(e, node.repr_token.start, node.repr_token.end, node.repr_token.line,
                                 global_vars.PROGRAM_LINES[node.repr_token.line - 1], *args))
        return


# Category Nodes (DO NOT construct these nodes!)

class Node:
    def __init__(self, repr_tok: Token, parent_node=None):
        self.parent: Optional[ScopeNode] = parent_node
        self.scope_level = 0
        self.repr_token = repr_tok
        return

    def update(self, ctx: Context) -> Optional['Node']:
        pass

    def __repr__(self):
        return f'<{self.repr_token}>'


class ExpressionNode(Node):
    def __init__(self, repr_tok: Token):
        super().__init__(repr_tok)
        self.type: Optional[Type] = None
        return

    def get_type(self):
        return self.type


class NameDefineNode(Node):
    def __init__(self, repr_tok: Token, name: 'VariableNode'):
        super().__init__(repr_tok)
        self.name: VariableNode = name

    def get_id(self):
        return self.name

    def add_ctx(self, ctx: Context):
        pass


# Base NODES

class ValueNode(ExpressionNode):
    def __init__(self, tok: Token):
        super().__init__(tok)

    def update(self, ctx: Context) -> Optional['Node']:
        self.type = convert_py_type(self.repr_token)
        self.type.definition = ctx.types[self.type]
        return self

    def __repr__(self):
        return f'{self.repr_token.value}'


class VariableNode(ExpressionNode):
    def __init__(self, repr_tok: Token):
        super().__init__(repr_tok)
        self.name: str = repr_tok.value
        self.offset: int = 0
        return

    def __repr__(self):
        return f'{self.name}'


class ScopeNode(Node):
    def __init__(self, start_tok: Token, child_nodes: list[Node]):
        super().__init__(start_tok)
        self.child_nodes: list[Node] = child_nodes
        return

    def update(self, ctx: Context) -> Optional['Node']:
        new_ctx = ctx.clone()
        for node in self.child_nodes:
            node.scope_level = self.scope_level + 1
            if isinstance(node, NameDefineNode):
                node.add_ctx(new_ctx)

        for i, node in enumerate(self.child_nodes):

            self.child_nodes[i] = node.update(new_ctx)

        return

    def __repr__(self):
        string = f'\n{self.scope_level*"  "}{{\n'
        for node in self.child_nodes:
            string += f'{(self.scope_level+1)*"  "}{node.__repr__()}\n'
        return string + f'{self.scope_level*"  "}}}'


# Operation Nodes

class AssignNode(ExpressionNode):
    def __init__(self, var: VariableNode, value: ExpressionNode):
        super().__init__(var.repr_token)
        self.var: VariableNode = var
        self.value: ExpressionNode = value
        return

    def __repr__(self):
        return f'{self.var.repr_token.value} = {self.value}'


class UnOpNode(ExpressionNode):
    def __init__(self, op: Token, child: ExpressionNode):
        super().__init__(op)
        self.op = op
        self.child = child
        return

    def __repr__(self):
        return f'{self.op.value} {self.child}'


class BinOpNode(ExpressionNode):
    def __init__(self, op: Token, left: ExpressionNode, right: ExpressionNode):
        super().__init__(op)
        self.left_child = left
        self.right_child = right
        self.op = op
        return

    def __repr__(self):
        return f'{self.left_child} {self.op.value} {self.right_child}'


class DotOperatorNode(VariableNode):
    def __init__(self, repr_tok: Token, var: VariableNode, field: Token):
        super().__init__(repr_tok)
        self.var: VariableNode = var
        self.field: Token = field

    def __repr__(self):
        return f'{self.var}.{self.field}'


class IndexOperatorNode(VariableNode):
    def __init__(self, repr_tok: Token, collection: ExpressionNode, index: ExpressionNode):
        super().__init__(repr_tok)
        self.collection: ExpressionNode = collection
        self.index: ExpressionNode = index

    def __repr__(self):
        return f'{self.collection}[{self.index}]'


class FuncCallNode(ExpressionNode):
    def __init__(self, repr_tok: Token, func_name: VariableNode, args: tuple[ExpressionNode]):
        super().__init__(repr_tok)
        self.name: VariableNode = func_name
        self.args: tuple[ExpressionNode] = args

    def __repr__(self):
        return f'{self.name}{self.args}'


# Definition Nodes

class MacroDefineNode(NameDefineNode):
    def __init__(self, repr_tok: Token, name: 'VariableNode', expression: ExpressionNode):
        super().__init__(repr_tok, name)
        self.value: ExpressionNode = expression

    def __repr__(self):
        return f'(macro) {self.name} = {self.value}'


class VarDefineNode(NameDefineNode, ExpressionNode):
    def __init__(self, repr_tok: Token, var_type: Type, var_name: VariableNode, value: Optional[ExpressionNode]):
        super().__init__(repr_tok, var_name)
        self.var_type: Type = var_type
        self.value: Optional[ExpressionNode] = value

    def add_ctx(self, ctx: Context) -> None:
        ctx.vars[self.name.name] = self

    def __repr__(self):
        if self.value is None:
            return f'{self.var_type} {self.name.repr_token.value}'
        else:
            return f'{self.var_type} {self.name.repr_token.value} = {self.value}'


class FuncDefineNode(NameDefineNode):
    def __init__(self, repr_tok: Token, ret_type: Type, func_name: VariableNode, args: tuple[VarDefineNode], body: Node):
        super().__init__(repr_tok, func_name)
        self.ret_type: Type = ret_type
        self.args: tuple[VarDefineNode] = args
        self.body: Node = body

    def get_id(self) -> tuple:
        return self.name, self.args

    def add_ctx(self, ctx: Context) -> None:
        ctx.funcs[self.get_id()] = self
        return

    def __repr__(self):
        return f'{self.ret_type} {self.name}({self.args.__repr__()[1:-1]}) {self.body}'


class ClassDefineNode(NameDefineNode):
    def __init__(self, repr_tok: Token, name: 'VariableNode', class_type: Type, body: Node):
        super().__init__(repr_tok, name)
        self.body: Node = body
        self.type: Type = class_type
        self.type.definition = self

    def add_ctx(self, ctx: Context) -> None:
        ctx.types[self.type] = self
        return

    def __repr__(self):
        return f'class {self.name} {self.body}'


# Control Flow

class IfNode(Node):
    def __init__(self, repr_tok: Token, condition: ExpressionNode, body: Node):
        super().__init__(repr_tok)
        self.condition = condition
        self.body = body
        self.else_statement: Optional[ElseNode] = None

    def __repr__(self):
        if self.else_statement is None:
            return f'if {self.condition} {self.body}'
        else:
            return f'if {self.condition} {self.body} \n{self.scope_level*"  "}{self.else_statement}'
            # maybe add: ```` between body and else, to make formatting better


class ElseNode(Node):
    def __init__(self, repr_tok: Token, body: Node, if_statement: Optional[IfNode]):
        super().__init__(repr_tok)
        self.body = body
        self.if_statement: Optional[IfNode] = if_statement

    def __repr__(self):
        return f'else {self.body}'


class WhileNode(Node):
    def __init__(self, repr_tok: Token, condition: ExpressionNode, body: Node):
        super().__init__(repr_tok)
        self.condition = condition
        self.body = body

    def __repr__(self):
        return f'while {self.condition} {self.body}'


class DoWhileNode(Node):
    def __init__(self, repr_tok: Token, condition: ExpressionNode, body: Node):
        super().__init__(repr_tok)
        self.condition = condition
        self.body = body

    def __repr__(self):
        return f'do {self.body} while {self.condition}'


class ReturnNode(Node):
    def __init__(self, repr_tok: Token, value: Optional[ExpressionNode] = None):
        super().__init__(repr_tok)
        self.value = value

    def __repr__(self):
        if self.value is None:
            return f'return'
        else:
            return f'return {self.value}'


class BreakNode(Node):
    def __init__(self, repr_tok: Token, value: int = 1):
        super().__init__(repr_tok)
        self.value = value

    def __repr__(self):
        if self.value is None:
            return f'break'
        else:
            return f'break {self.value}'


class SkipNode(Node):
    def __init__(self, repr_tok: Token, value: int = 1):
        super().__init__(repr_tok)
        self.value = value

    def __repr__(self):
        if self.value is None:
            return f'skip'
        else:
            return f'skip {self.value}'
