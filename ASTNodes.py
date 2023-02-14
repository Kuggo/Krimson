from enum import Enum
from Constants import Token, TT, Error, global_vars, Operators
from typing import Optional
from copy import copy


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


# helper functions

def convert_py_type(tok: Token) -> 'Type':
    if isinstance(tok.value, int):
        if tok.value < 0:
            return Type(Token(TT.IDENTIFIER, 'int'))
        else:
            return Type(Token(TT.IDENTIFIER, 'nat'))
    elif isinstance(tok.value, str):
        return Type(Token(TT.IDENTIFIER, 'bool'))
    elif isinstance(tok.value, list):
        return Type(Token(TT.IDENTIFIER, 'array'))
    elif isinstance(tok.value, dict):
        return Type(Token(TT.IDENTIFIER, 'dict'))
    elif isinstance(tok.value, set):
        return Type(Token(TT.IDENTIFIER, 'set'))
    else:
        assert False


# Type

class Type:
    def __init__(self, name: Token, generics: Optional[tuple['Type', ...]] = None):
        self.name: Token = name
        self.generics: Optional[tuple['Type', ...]] = generics
        self.size = 1

    def __eq__(self, other: 'Type'):
        return self.name == other.name and self.generics == self.generics

    def __hash__(self):
        return self.name.value.__hash__()

    def is_subtype(self, super_type: 'Type'):
        if self.name == super_type.name:
            return True

        for gen, sup_gen in zip(self.generics, super_type.generics):
            if not gen.is_subtype(sup_gen):
                return False

        return True

    def get_type_label(self) -> str:
        if self.generics is None:
            return self.name.value

        string = self.name.value
        for gen in self.generics:
            string += f'.{gen.get_type_label()}'
        return string

    def __repr__(self):
        if self.generics is None:
            return f'{self.name.value}'
        else:
            return f'{self.name.value}[{self.generics.__repr__()[1:-1]}]'


class_type = Type(Token(TT.IDENTIFIER, 'class'))


# Errors

class TypeError(Enum):
    no_attribute = 'type "{}" has no attribute {}"'
    unk_var = 'Undefined variable "{}"'
    undefined_function = 'Undefined function for the given args'
    def_statement_expected = 'Declaration statement expected inside a class body'


# AST level

class Context:
    """Compile Time context to process AST and type check"""
    def __init__(self, up_scope: Optional['Context'] = None):
        self.up_scope: Optional[Context] = up_scope

        if up_scope is None:
            self.scope_level: int = 0
            self.errors: list[Error] = []
            self.stack_map: dict[str, int] = {}
            self.funcs: dict[tuple, FuncDefineNode] = {}
            self.types: dict[Type, ClassDefineNode] = {}
            self.vars: dict[str, (MacroDefineNode, VarDefineNode)] = {}
        else:
            self.scope_level: int = up_scope.scope_level + 1
            self.errors: list[Error] = up_scope.errors    # not a copy. All errors will go to the same collection
            self.stack_map: dict[str, int] = up_scope.stack_map.copy()
            self.funcs: dict[tuple[str, tuple[Type, ...]], FuncDefineNode] = up_scope.funcs.copy()
            self.types: dict[Type, ClassDefineNode] = up_scope.types.copy()
            self.vars: dict[str, (MacroDefineNode, VariableNode)] = up_scope.vars.copy()
        return

    def get_var(self, var_name: str) -> Optional['VarDefineNode']:
        if var_name in self.vars:
            return self.vars[var_name]
        else:
            return None

    def get_func(self, func: 'FuncCallNode') -> Optional['FuncDefineNode']:
        func_id = func.get_id()
        if func_id in self.funcs:
            return self.funcs[func_id]
        else:
            return None

    def get_class(self, t: Type) -> Optional['ClassDefineNode']:
        if t in self.types:
            return self.types[t]
        else:
            return None

    def get_class_by_name(self, name: str) -> Optional['ClassDefineNode']:
        return self.get_class(Type(Token(TT.IDENTIFIER, name)))

    def get_definition(self, name: str) -> Optional['NameDefineNode']:
        var = self.get_var(name)
        if var is not None:
            return var

        var = self.get_class_by_name(name)
        if var is not None:
            return var

        for f in self.funcs.keys():
            if f[0] == name:
                return self.funcs[f]

        return None

    def has_field(self, field: str) -> bool:
        if field in self.vars:
            return True

        for t in self.types.keys():
            if t.name.value == field:
                return True

        for f in self.funcs.keys():
            if f[0] == field:
                return True

        return False

    def stack_location(self, var: 'VariableNode') -> int:
        if var.name in self.stack_map:
            return self.stack_map[var.name]
        else:
            size = var.get_size()
            i = 0
            last_index = 0
            for index in sorted(self.stack_map.values()):
                if last_index + size <= index:
                    self.stack_map[var.name] = last_index
                    return last_index

            self.stack_map[var.name] = i
            return i

    def stack_dealloc(self, var: 'VariableNode') -> None:
        if var.name in self.stack_map:
            del self.stack_map[var.name]
        else:
            assert False

    def clone(self) -> 'Context':
        c = Context(self)
        self.up_scope = c
        return c

    def error(self, e: TypeError, node: 'Node', *args) -> None:
        self.errors.append(Error(e, node.repr_token.start, node.repr_token.end, node.repr_token.line,
                                 global_vars.PROGRAM_LINES[node.repr_token.line - 1], *args))
        return

    def __repr__(self):
        return f'vars: {self.vars}\nfuncs: {self.funcs}\ntypes: {self.types}'


# Category Nodes (DO NOT construct these nodes!)

class Node:
    def __init__(self, repr_tok: Token, parent_node=None):
        self.parent: Optional[ScopeNode] = parent_node
        self.scope_level = 0
        self.repr_token = repr_tok
        return

    def update(self, ctx: Context) -> Optional['Node']:
        self.scope_level = ctx.scope_level
        return self

    def alloc_vars(self, ctx: Context) -> None:
        """In reverse order, it traverses the AST and when it finds a variable it finds a location for it on the stack.
        That value is saved on ``self.offset``.

        If it finds the variable declaration node of that variable it deallocates the space being used

        :param ctx:
        :return: """
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

    def get_size(self) -> int:
        return self.type.size


class NameDefineNode(Node):
    def __init__(self, repr_tok: Token, name: 'VariableNode'):
        super().__init__(repr_tok)
        self.name: VariableNode = name
        return

    def get_id(self) -> str:
        """Returns the identifier of the Definition node to be used as key in the dictionary of namespace"""
        return self.name.name

    def add_ctx(self, ctx: Context) -> None:
        """Adds the definition to the current context dict"""
        pass

    def update(self, ctx: Context) -> Optional['NameDefineNode']:
        self.scope_level = ctx.scope_level
        self.add_ctx(ctx)
        return self


# Base NODES

class ValueNode(ExpressionNode):
    def __init__(self, tok: Token):
        super().__init__(tok)
        return

    def update(self, ctx: Context) -> Optional['ValueNode']:
        self.scope_level = ctx.scope_level
        self.type = convert_py_type(self.repr_token)
        t = ctx.get_class(self.type)
        if t is not None:
            self.type.size = t.size
        return self

    def __repr__(self):
        return f'{self.repr_token.value}'


class VariableNode(ExpressionNode):
    def __init__(self, repr_tok: Token):
        super().__init__(repr_tok)
        self.name: str = repr_tok.value
        self.offset: int = 0
        return

    def update(self, ctx: Context, use_dunder_func=False) -> Optional['VariableNode']:
        self.scope_level = ctx.scope_level
        var = ctx.get_definition(self.name)
        if isinstance(var, VarDefineNode):
            self.type = var.get_type()
            return self

        elif isinstance(var, MacroDefineNode):
            value = copy(var.value)
            return value.update(ctx)

        elif isinstance(var, ClassDefineNode):
            self.type = class_type
            return self

        else:
            ctx.error(TypeError.unk_var, self, self.repr_token.value)
            return None

    def alloc_vars(self, ctx: Context) -> None:
        self.offset = ctx.stack_location(self)
        return

    def __repr__(self):
        return f'{self.name}'


class ScopeNode(Node):
    def __init__(self, start_tok: Token, child_nodes: list[Node]):
        super().__init__(start_tok)
        self.child_nodes: list[Node] = child_nodes
        return

    def process_body(self, ctx: Context) -> None:
        for node in self.child_nodes:
            if isinstance(node, NameDefineNode) and not isinstance(node, VarDefineNode):
                node.update(ctx)

        child_nodes = []
        for node in self.child_nodes:
            if isinstance(node, NameDefineNode) and not isinstance(node, VarDefineNode):
                continue

            node = node.update(ctx)
            if node is not None:
                child_nodes.append(node)

        self.child_nodes = child_nodes
        return

    def update(self, ctx: Context) -> Optional['Node']:
        self.scope_level = ctx.scope_level
        new_ctx = ctx.clone()
        self.process_body(new_ctx)
        return self

    def alloc_vars(self, ctx: Context) -> None:
        for node in reversed(self.child_nodes):
            node.alloc_vars(ctx)
        return

    def __repr__(self):
        string = f'\n{self.scope_level*"  "}{{\n'
        for node in self.child_nodes:
            string += f'{node.scope_level*"  "}{node.__repr__()}\n'
        return string + f'{self.scope_level*"  "}}}'


# Operation Nodes

class AssignNode(ExpressionNode):
    def __init__(self, var: VariableNode, value: ExpressionNode):
        super().__init__(var.repr_token)
        self.var: VariableNode = var
        self.value: ExpressionNode = value
        return

    def update(self, ctx: Context) -> Optional['AssignNode']:
        self.scope_level = ctx.scope_level
        self.var = self.var.update(ctx)     # cannot be None
        self.value = self.value.update(ctx)
        if self.var is None or self.value is None:
            return None

        # TODO while no refs this is impossible:
        # if isinstance(self.var, FuncCallNode) and self.var.name == '__get__':
        #     self.var.name = '__set__'   # the index is not to get but to assign to

        self.type = self.value.get_type()
        return self

    def alloc_vars(self, ctx: Context) -> None:
        self.value.alloc_vars(ctx)
        self.var.alloc_vars(ctx)
        return

    def __repr__(self):
        return f'{self.var.repr_token.value} = {self.value}'


class UnOpNode(ExpressionNode):
    def __init__(self, op: Token, child: ExpressionNode):
        super().__init__(op)
        self.op = op
        self.child = child
        return

    def update(self, ctx: Context) -> Optional['FuncCallNode']:
        self.scope_level = ctx.scope_level
        self.child = self.child.update(ctx)
        if self.child is None:
            return None

        if self.op.value in dunder_funcs:
            func_name = dunder_funcs[self.op.value]
            return FuncCallNode(self.repr_token, VariableNode(func_name), (self.child,)).update(ctx)
        else:
            assert False

    def alloc_vars(self, ctx: Context) -> None:
        self.child.alloc_vars(ctx)
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

    def update(self, ctx: Context) -> Optional['FuncCallNode']:
        self.scope_level = ctx.scope_level
        self.left_child = self.left_child.update(ctx)
        self.right_child = self.right_child.update(ctx)

        if self.left_child is None or self.right_child is None:
            return None

        if self.op.value in dunder_funcs:
            fn_name = dunder_funcs[self.op.value]
            return FuncCallNode(self.repr_token, VariableNode(fn_name), (self.left_child, self.right_child)).update(ctx)
        else:
            assert False

    def alloc_vars(self, ctx: Context) -> None:
        self.right_child.alloc_vars(ctx)
        self.left_child.alloc_vars(ctx)
        return

    def __repr__(self):
        return f'{self.left_child} {self.op.value} {self.right_child}'


class DotOperatorNode(VariableNode):
    def __init__(self, repr_tok: Token, var: VariableNode, field: VariableNode):
        super().__init__(repr_tok)
        self.name = field.name
        self.var: VariableNode = var
        self.field: VariableNode = field
        return

    def update(self, ctx: Context, use_dunder_func=False) -> Optional['DotOperatorNode']:
        self.scope_level = ctx.scope_level
        self.var = self.var.update(ctx)
        if self.var is None:
            return None

        if self.var.type == class_type:
            t = ctx.get_class_by_name(self.var.name)
        else:
            t = ctx.get_class(self.var.type)

        if t is None:
            return None

        field = t.get_field(self.field.name)

        if isinstance(field, VarDefineNode):
            self.offset = field.offset
            self.type = field.type
            return self

        elif isinstance(field, ClassDefineNode):
            self.offset = t.body.offset_map[self.field.name]
            self.type = class_type
            return self

        elif isinstance(field, FuncDefineNode):
            self.type = field.func_to_type()
            return self

        else:
            ctx.error(TypeError.no_attribute, self, self.var.type.name, self.field.name)
            return None

    def alloc_vars(self, ctx: Context) -> None:
        self.var.alloc_vars(ctx)
        return

    def __repr__(self):
        return f'{self.var}.{self.field}'


class IndexOperatorNode(VariableNode):
    def __init__(self, repr_tok: Token, collection: ExpressionNode, index: ExpressionNode):
        super().__init__(repr_tok)
        self.collection: ExpressionNode = collection
        self.index: ExpressionNode = index
        return

    def update(self, ctx: Context, use_dunder_func=False) -> Optional['IndexOperatorNode']:
        self.scope_level = ctx.scope_level
        self.collection = self.collection.update(ctx)
        self.index = self.index.update(ctx)

        if self.collection is None or self.index is None:
            return None

        if use_dunder_func:
            fn_name = dunder_funcs[Operators.index.value.value]
            return FuncCallNode(self.repr_token, VariableNode(fn_name), (self.collection, self.index)).update(ctx)
        else:
            return self

    def alloc_vars(self, ctx: Context) -> None:
        self.index.alloc_vars(ctx)
        self.collection.alloc_vars(ctx)
        return

    def __repr__(self):
        return f'{self.collection}[{self.index}]'


class FuncCallNode(ExpressionNode):
    def __init__(self, repr_tok: Token, func_name: VariableNode, args: tuple[ExpressionNode, ...]):
        super().__init__(repr_tok)
        self.func_name: VariableNode = func_name
        self.args: tuple[ExpressionNode, ...] = args

    def get_id(self) -> tuple[str, tuple[Type, ...]]:
        args: list[Type] = []
        for arg in self.args:
            if arg.type is not None:
                args.append(arg.type)
        return self.func_name.name, tuple(args)

    def update(self, ctx: Context) -> Optional['FuncCallNode']:
        self.scope_level = ctx.scope_level
        self.func_name = self.func_name.update(ctx)
        if self.func_name is None:
            return None

        args = []
        if isinstance(self.func_name, DotOperatorNode):  # syntax sugar
            t = ctx.get_class(self.func_name.var.type)
            if t is not None and t != ctx.get_class_by_name(self.func_name.var.name):   # it's an instance and not the class itself
                args.append(self.func_name.var)

        for arg in self.args:
            args.append(arg.update(ctx))
        self.args = tuple(args)

        func = ctx.get_func(self)
        if func is None:
            ctx.error(TypeError.undefined_function, self)
            return None
        return self

    def alloc_vars(self, ctx: Context) -> None:
        for arg in reversed(self.args):
            arg.alloc_vars(ctx)
        return

    def __repr__(self):
        return f'{self.func_name}{self.args}'


# Definition Nodes

class MacroDefineNode(NameDefineNode):
    def __init__(self, repr_tok: Token, name: 'VariableNode', expression: ExpressionNode):
        super().__init__(repr_tok, name)
        self.value: ExpressionNode = expression

    def add_ctx(self, ctx: Context) -> None:
        ctx.vars[self.name.name] = self
        return

    def __repr__(self):
        return f'(macro) {self.name} = {self.value}'


class VarDefineNode(NameDefineNode, ExpressionNode):
    def __init__(self, repr_tok: Token, var_type: Type, var_name: VariableNode, value: Optional[ExpressionNode]):
        super().__init__(repr_tok, var_name)
        self.type: Type = var_type
        self.value: Optional[ExpressionNode] = value
        self.offset: int = 0

    def add_ctx(self, ctx: Context) -> None:
        ctx.vars[self.name.name] = self
        return

    def update(self, ctx: Context) -> Optional['VarDefineNode']:
        self.scope_level = ctx.scope_level
        if self.value is not None:
            self.value = self.value.update(ctx)
            # if self.value is None:
            #     return None
        self.add_ctx(ctx)
        self.name = self.name.update(ctx)

        if self.name is None:
            return None

        return self

    def alloc_vars(self, ctx: Context) -> None:
        if self.value is not None:
            self.value.alloc_vars(ctx)
        self.offset = ctx.stack_location(self.name)
        ctx.stack_dealloc(self.name)    # first occurrence of this variable, prior to this point, this slot can be used
        return

    def __repr__(self):
        if self.value is None:
            return f'{self.type} {self.name.repr_token.value}'
        else:
            return f'loc:{self.offset} {self.type} {self.name.repr_token.value} = {self.value}'


class FuncDefineNode(NameDefineNode):
    def __init__(self, repr_tok: Token, ret_type: Type, func_name: VariableNode, args: tuple[VarDefineNode, ...], body: 'IsolatedScopeNode'):
        super().__init__(repr_tok, func_name)
        self.ret_type: Type = ret_type
        self.args: tuple[VarDefineNode, ...] = args
        self.body: IsolatedScopeNode = body

    def get_id(self) -> tuple[str, tuple[Type, ...]]:
        args: list[Type] = []
        for arg in self.args:
            if arg.type is not None:
                args.append(arg.type)
        return self.name.name, tuple(args)

    def func_to_type(self) -> Type:
        args: list[Type] = []
        for arg in self.args:
            if arg.type is not None:
                args.append(arg.type)

        return Type(self.name.repr_token, tuple(args))

    def get_func_label(self) -> str:
        string = self.name.name
        for arg in self.args:
            string += f'.{arg.type.get_type_label()}'
        return string

    def add_ctx(self, ctx: Context) -> None:
        ctx.funcs[self.get_id()] = self
        return

    def update(self, ctx: Context) -> Optional['FuncDefineNode']:
        self.scope_level = ctx.scope_level
        self.add_ctx(ctx)

        args = []
        for arg in self.args:
            arg = arg.update(ctx)
            if arg is not None:
                args.append(arg)
        self.args = tuple(args)

        self.body = self.body.update(ctx)
        if self.body is None:
            return None

        self.body.alloc_vars(ctx)   # allocating local variables on the stack

        i = 0   # allocating arguments on the stack, above the BP
        for arg in reversed(self.args):
            arg.offset = i
            i += arg.get_size()

        return self

    def inline_func(self, func_call: FuncCallNode) -> ScopeNode:
        pass    # TODO

    def __repr__(self):
        return f'{self.ret_type} {self.name}({self.args.__repr__()[1:-1]}) {self.body}'


class IsolatedScopeNode(ScopeNode):
    def __init__(self, start_tok: Token, child_nodes: list[Node]):
        super().__init__(start_tok, child_nodes)
        return

    @classmethod
    def new_from_old(cls, old: ScopeNode) -> 'IsolatedScopeNode':
        return cls(old.repr_token, old.child_nodes)

    def create_context(self, ctx: Context) -> Context:
        """Creates a new context for a scope node via ctx.clone(), but isolates the variables defined in upper scope
        :param ctx: Context of the above scope
        :return: the new context of the current scope"""
        new_ctx = ctx.clone()

        new_ctx.stack_map = {}  # erasing pre existing variables
        new_ctx.vars = {}       # same here
        return new_ctx

    def update(self, ctx: Context) -> Optional['IsolatedScopeNode']:
        self.scope_level = ctx.scope_level
        new_ctx = self.create_context(ctx)
        self.process_body(new_ctx)
        return self


class ClassDefineNode(NameDefineNode):
    def __init__(self, repr_tok: Token, name: 'VariableNode', c_type: Type, body: 'ClassBodyNode'):
        super().__init__(repr_tok, name)
        self.body: ClassBodyNode = body
        self.type: Type = c_type
        self.size: int = 0

        self.body.class_def = self
        return

    def update(self, ctx: Context) -> Optional['ClassDefineNode']:
        self.scope_level = ctx.scope_level
        self.add_ctx(ctx)

        self.body = self.body.update(ctx)
        if self.body is None:
            return None

        self.type.size = self.size
        return self

    def add_ctx(self, ctx: Context) -> None:
        ctx.types[self.type] = self
        return

    def get_id(self) -> Type:
        return self.type

    def get_var(self, var_name: str) -> Optional[VarDefineNode]:
        if var_name in self.body.vars:
            return self.body.vars[var_name]
        else:
            return None

    def get_func(self, func: 'FuncCallNode') -> Optional['FuncDefineNode']:
        func_id = func.get_id()
        if func_id in self.body.funcs:
            return self.body.funcs[func_id]
        else:
            return None

    def get_class(self, t: Type) -> Optional['ClassDefineNode']:
        if t in self.body.types:
            return self.body.types[t]
        else:
            return None

    def get_class_by_name(self, name: str) -> Optional['ClassDefineNode']:
        return self.get_class(Type(Token(TT.IDENTIFIER, name)))

    def get_field(self, field: str) -> Optional[NameDefineNode]:
        if field in self.body.vars:
            return self.body.vars[field]

        t = self.get_class_by_name(field)
        if t is not None:
            return t

        for f in self.body.funcs.keys():
            if f[0] == field:
                return self.body.funcs[f]

        return None

    def __repr__(self):
        return f'class {self.name} {self.body}'


class ClassBodyNode(ScopeNode):
    def __init__(self, start_tok: Token, child_nodes: list[Node]):
        super().__init__(start_tok, child_nodes)
        self.funcs: dict[tuple, FuncDefineNode] = {}
        self.types: dict[Type, ClassDefineNode] = {}
        self.vars: dict[str, (MacroDefineNode, VarDefineNode)] = {}
        self.offset_map: dict[str, int] = {}
        self.size: int = 0
        self.class_def: Optional[ClassDefineNode] = None
        return

    @classmethod
    def new_from_old(cls, old: ScopeNode) -> 'ClassBodyNode':
        return cls(old.repr_token, old.child_nodes)

    def process_body(self, ctx: Context) -> None:
        child_nodes = []
        for node in self.child_nodes:
            if isinstance(node, NameDefineNode):
                node = node.update(ctx)

                if isinstance(node, VarDefineNode):
                    self.vars[node.name.name] = node
                    self.offset_map[node.name.name] = self.size
                    self.size += node.type.size

                elif isinstance(node, MacroDefineNode):
                    self.vars[node.get_id()] = node

                elif isinstance(node, ClassDefineNode):
                    self.types[node.get_id()] = node

                elif isinstance(node, FuncDefineNode):
                    self.funcs[node.get_id()] = node

                if node is not None:
                    child_nodes.append(node)
            else:
                ctx.error(TypeError.def_statement_expected, node)

        self.child_nodes = child_nodes
        return

    def create_context(self, ctx: Context) -> Context:
        """Creates a new context for a scope node via ctx.clone(), but isolates the variables defined in upper scope.
        Additionally, functions defined in the class's body will be visible on the class's definition scope.

        :param ctx: Context of the above scope
        :return: the new context of the current scope"""
        new_ctx = ctx.clone()

        new_ctx.stack_map = {}      # erasing pre existing variables
        new_ctx.vars = {}           # same here

        new_ctx.funcs = ctx.funcs   # methods need to be accessible outside of class
        return new_ctx

    def update(self, ctx: Context) -> Optional['ClassBodyNode']:
        self.scope_level = ctx.scope_level
        new_ctx = self.create_context(ctx)
        self.process_body(new_ctx)
        return self


# Control Flow

class IfNode(Node):
    def __init__(self, repr_tok: Token, condition: ExpressionNode, body: Node):
        super().__init__(repr_tok)
        self.condition = condition
        self.body = body
        self.else_statement: Optional[ElseNode] = None
        return

    def update(self, ctx: Context) -> Optional['IfNode']:
        self.scope_level = ctx.scope_level

        self.condition = self.condition.update(ctx)

        self.body = self.body.update(ctx)
        if self.body is None or self.condition is None:
            return None

        if self.else_statement is not None:
            self.else_statement = self.else_statement.update(ctx)

        return self

    def alloc_vars(self, ctx: Context) -> None:
        if self.else_statement is not None:
            self.else_statement.alloc_vars(ctx)
        self.body.alloc_vars(ctx)
        self.condition.alloc_vars(ctx)
        return

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
        return

    def update(self, ctx: Context) -> Optional['ElseNode']:
        self.scope_level = ctx.scope_level

        self.body = self.body.update(ctx)
        if self.body is None:
            return None

        return self

    def alloc_vars(self, ctx: Context) -> None:
        self.body.alloc_vars(ctx)
        return

    def __repr__(self):
        return f'else {self.body}'


class WhileNode(Node):
    def __init__(self, repr_tok: Token, condition: ExpressionNode, body: Node):
        super().__init__(repr_tok)
        self.condition = condition
        self.body = body
        return

    def update(self, ctx: Context) -> Optional['WhileNode']:
        self.scope_level = ctx.scope_level

        self.condition = self.condition.update(ctx)

        self.body = self.body.update(ctx)
        if self.body is None or self.condition is None:
            return None

        return self

    def alloc_vars(self, ctx: Context) -> None:
        self.body.alloc_vars(ctx)
        self.condition.alloc_vars(ctx)
        return

    def __repr__(self):
        return f'while {self.condition} {self.body}'


class DoWhileNode(WhileNode):
    def __init__(self, repr_tok: Token, condition: ExpressionNode, body: Node):
        super().__init__(repr_tok, condition, body)
        return

    def __repr__(self):
        return f'do {self.body} while {self.condition}'


class ReturnNode(Node):
    def __init__(self, repr_tok: Token, value: Optional[ExpressionNode] = None):
        super().__init__(repr_tok)
        self.value = value
        return

    def update(self, ctx: Context) -> Optional['Node']:
        self.scope_level = ctx.scope_level
        self.value = self.value.update(ctx)
        if self.value is None:
            return None
        return self

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
