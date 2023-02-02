from Constants import Token
from typing import Optional


class Type:
    def __init__(self, name: Token, generics: Optional[list['Type']] = None):
        self.name: Token = name
        self.generics: Optional[list['Type']] = generics

    def __repr__(self):
        if self.generics is None:
            return f'{self.name.value}'
        else:
            return f'{self.name.value}<{self.generics.__repr__()[1:-1]}>'


# Category Nodes (DO NOT construct these nodes!)

class Node:
    def __init__(self, repr_tok: Token, parent_node=None):
        self.parent: Optional[ScopeNode] = parent_node
        self.repr_token = repr_tok
        return

    def __repr__(self):
        return f'<{self.repr_token}>'


class ExpressionNode(Node):
    def __init__(self, repr_tok: Token):
        super().__init__(repr_tok)
        return


class NameDefineNode(Node):
    def __init__(self, repr_tok: Token):
        super().__init__(repr_tok)


# Base NODES

class ValueNode(ExpressionNode):
    def __init__(self, tok: Token):
        super().__init__(tok)

    def __repr__(self):
        return f'{self.repr_token.value}'


class VariableNode(ExpressionNode):
    def __init__(self, repr_tok: Token):
        super().__init__(repr_tok)
        return

    def __repr__(self):
        return f'{self.repr_token.value}'


class ScopeNode(Node):
    def __init__(self, start_tok: Token, child_nodes: list[Node]):
        super().__init__(start_tok)
        self.child_nodes: list[Node] = child_nodes
        return

    def __repr__(self):
        string = '{\n'
        for node in self.child_nodes:
            string += node.__repr__() + '\n'
        return string + '}'


# Operation Nodes

class AssignNode(ExpressionNode):
    def __init__(self, var: VariableNode, value: ExpressionNode):
        super().__init__(var.repr_token)
        self.var: VariableNode = var
        self.value: ExpressionNode = value
        return

    def __repr__(self):
        return f'<{self.var.repr_token.value} = {self.value}>'


class UnOpNode(ExpressionNode):
    def __init__(self, op: Token, child: ExpressionNode):
        super().__init__(op)
        self.op = op
        self.child = child
        return

    def __repr__(self):
        return f'<{self.op.value} {self.child}>'


class BinOpNode(ExpressionNode):
    def __init__(self, op: Token, left: ExpressionNode, right: ExpressionNode):
        super().__init__(op)
        self.left_child = left
        self.right_child = right
        self.op = op
        return

    def __repr__(self):
        return f'<{self.left_child} {self.op.value} {self.right_child}>'


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


# Definition Nodes

class MacroDefineNode(NameDefineNode):  # TODO left undone for now
    pass


class VarDefineNode(NameDefineNode, ExpressionNode):
    def __init__(self, repr_tok: Token, var_type: Type, var: VariableNode, value: Optional[ExpressionNode]):
        super().__init__(repr_tok)
        self.var: VariableNode = var
        self.var_type: Type = var_type
        self.value: Optional[ExpressionNode] = value

    def __repr__(self):
        if self.value is None:
            return f'<{self.var_type} {self.var.repr_token.value}>'
        else:
            return f'<{self.var_type} {self.var.repr_token.value} = {self.value}>'


class FuncDefineNode(NameDefineNode):
    pass


class ClassDefineNode(NameDefineNode):
    pass


# Control Flow

class IfNode(Node):
    def __init__(self, repr_tok: Token, condition: ExpressionNode, body: Node):
        super().__init__(repr_tok)
        self.condition = condition
        self.body = body


class ElseNode(Node):
    def __init__(self, repr_tok: Token, body: Node):
        super().__init__(repr_tok)
        self.body = body


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
