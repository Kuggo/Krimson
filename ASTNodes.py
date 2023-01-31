from Lexer import Token
from typing import Optional


class Node:
    def __init__(self, repr_tok: Token, parent_node=None):
        self.parent: Optional[ScopedNode] = parent_node
        self.repr_token = repr_tok
        return

    def __repr__(self):
        return f'<{self.repr_token}>'


class ScopedNode(Node):
    def __init__(self, start_tok: Token, child_nodes: list[Node]):
        super().__init__(start_tok)
        self.child_nodes: list[Node] = child_nodes
        return

    def __repr__(self):
        string = '{\n'
        for node in self.child_nodes:
            string += node.__repr__() + '\n'
        return string + '}'


class ExpressionNode(Node):
    def __init__(self, repr_tok: Token):
        super().__init__(repr_tok)
        return


class ValueNode(ExpressionNode):
    def __init__(self, tok: Token):
        super().__init__(tok)

    def __repr__(self):
        return f'{self.repr_token.value}'


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


class VariableNode(ExpressionNode):
    def __init__(self, repr_tok: Token):
        super().__init__(repr_tok)
        return

    def __repr__(self):
        return f'{self.repr_token.value}'


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


class AssignNode(ExpressionNode):
    def __init__(self, var: VariableNode, value: ExpressionNode):
        super().__init__(var.repr_token)
        self.var: VariableNode = var
        self.value: ExpressionNode = value
        return

    def __repr__(self):
        return f'<{self.var.repr_token.value} = {self.value}>'


class VarDefineNode(ExpressionNode):
    def __init__(self, repr_tok: Token, var_type: Token, var: VariableNode, value: Optional[ExpressionNode]):
        super().__init__(repr_tok)
        self.var: VariableNode = var
        self.var_type: Token = var_type
        self.value: Optional[ExpressionNode] = value

    def __repr__(self):
        if self.value is None:
            return f'<{self.var_type.value} {self.var.repr_token.value}>'
        else:
            return f'<{self.var_type.value} {self.var.repr_token.value} = {self.value}>'
