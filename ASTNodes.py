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
    static_not_in_class = 'Static modifier cannot be applied outside a Class definition scope'

    # Compile_time detected runtime errors
    out_of_bounds = 'Index is out of bounds'
    no_element = 'No element with such key'


# AST level

class Context:
    """Compile Time context to process AST and type check"""
    def __init__(self, up_scope: Optional['Context'] = None):
        self.up_scope: Optional[Context] = up_scope

        if up_scope is None:
            self.scope_level: int = 0
            self.errors: list[Error] = []
            self.stack_map: dict[str, tuple[int, int]] = {}
            self.funcs: dict[tuple, FuncDefineNode] = {}
            self.types: dict[Type, ClassDefineNode] = {}
            self.vars: dict[str, (MacroDefineNode, VarDefineNode)] = {}
        else:
            self.scope_level: int = up_scope.scope_level + 1
            self.errors: list[Error] = up_scope.errors    # not a copy. All errors will go to the same collection
            self.stack_map: dict[str, tuple[int, int]] = up_scope.stack_map.copy()
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

        return self.get_class_by_name(name)

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
            return self.stack_map[var.name][0]
        else:
            size = var.get_size()
            index = - size

            for var_index, var_size in sorted(self.stack_map.values(), reverse=True):
                if index >= var_index + var_size:
                    self.stack_map[var.name] = (index, size)
                    return index

                index = var_index - size

            self.stack_map[var.name] = (index, size)
            return index

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

    def update(self, ctx: Context, parent: 'Node') -> Optional['Node']:
        self.scope_level = ctx.scope_level
        self.parent = parent
        return self

    def alloc_vars(self, ctx: Context) -> None:
        """In reverse order, it traverses the AST and when it finds a variable it finds a location for it on the stack.
        That value is saved on ``self.offset``.

        If it finds the variable declaration node of that variable it deallocates the space being used

        :param ctx:
        :return: """
        pass

    def get_up_class_def(self) -> Optional['ClassDefineNode']:
        return self.parent.get_up_class_def()

    def gen_ir(self):
        pass

    def __repr__(self):
        return f'<{self.repr_token}>'


class ExpressionNode(Node):
    def __init__(self, repr_tok: Token):
        super().__init__(repr_tok)
        self.type: Optional[Type] = None
        return

    def update_type(self, ctx: Context):
        if self.type in ctx.types:
            t = ctx.types[self.type].type
            if t is not None:
                self.type = t
        return

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

    def update(self, ctx: Context, parent: Optional[Node]) -> Optional['NameDefineNode']:
        self.scope_level = ctx.scope_level
        self.parent = parent
        self.add_ctx(ctx)
        return self


# Base NODES

class ValueNode(ExpressionNode):
    def __init__(self, tok: Token):
        super().__init__(tok)
        return

    def update(self, ctx: Context, parent: Optional[Node]) -> Optional['ValueNode']:
        self.scope_level = ctx.scope_level
        self.parent = parent
        self.type = self.convert_py_type()
        return self

    def get_size(self) -> int:
        if isinstance(self.repr_token.value, list):
            size = 0
            for item in self.repr_token.value:
                size += item.get_size()
            return size
        else:
            return 1    # primitive size is 1

    def convert_py_type(self) -> 'Type':
        if isinstance(self.repr_token, Literal):
            if isinstance(self.repr_token.value, list):
                self.repr_token.literal_type.size = self.get_size()

            return self.repr_token.literal_type
        elif isinstance(self.repr_token.value, Registers):
            return any_type
        else:
            assert False

    def __repr__(self):
        return f'{self.repr_token.value}'


class VariableNode(ExpressionNode):
    def __init__(self, repr_tok: Token):
        super().__init__(repr_tok)
        self.name: str = repr_tok.value
        self.offset: OffsetNode = OffsetNode(Registers.SP, 0)
        return

    def update(self, ctx: Context, parent: Optional[Node], use_dunder_func=True, func=False) -> Optional['VariableNode']:
        self.scope_level = ctx.scope_level
        self.parent = parent
        var = ctx.get_definition(self.name)
        if isinstance(var, VarDefineNode):
            self.type = var.type
            self.offset = OffsetNode(Registers.BP, 0)

        elif isinstance(var, MacroDefineNode):
            value = copy(var.value)
            return value.update(ctx, self.parent)

        elif isinstance(var, ClassDefineNode):
            self.type = class_type

        elif isinstance(var, FuncDefineNode):
            self.type = var.func_to_type()

        else:
            if not func:
                ctx.error(TypeError.unk_var, self, self.repr_token.value)
            return None
        return self

    def alloc_vars(self, ctx: Context) -> None:
        self.offset.offset = ctx.stack_location(self)
        return

    def get_name(self) -> Node:
        return self

    def get_absolute_location(self) -> 'OffsetNode':
        return self.offset

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
        self.scope_level = ctx.scope_level
        self.parent = parent
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

    def update(self, ctx: Context, parent: Optional[Node]) -> Optional['ExpressionNode']:
        self.scope_level = ctx.scope_level
        self.parent = parent
        ref = RefNode(self.var)
        self.var = self.var.update(ctx, self.parent)  # cannot be None
        self.value = self.value.update(ctx, self.parent)
        if self.var is None or self.value is None:
            return None

        self.update_type(ctx)

        if isinstance(self.var, FuncCallNode) and self.var.name == '__get__':
            self.var.name = '__set__'   # the index is not to get but to assign to
            self.var.args = (ref,) + self.var.args
            return self.var

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

    def update(self, ctx: Context, parent: Optional[Node]) -> Optional['FuncCallNode']:
        self.scope_level = ctx.scope_level
        self.parent = parent
        self.child = self.child.update(ctx, self.parent)
        if self.child is None:
            return None

        if self.op.value in dunder_funcs:
            func_name = dunder_funcs[self.op.value]
            return FuncCallNode(self.repr_token, VariableNode(func_name), (self.child,)).update(ctx, self.parent)
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

    def update(self, ctx: Context, parent: Optional[Node]) -> Optional['FuncCallNode']:
        self.scope_level = ctx.scope_level
        self.parent = parent
        self.left_child = self.left_child.update(ctx, self.parent)
        self.right_child = self.right_child.update(ctx, self.parent)

        if self.left_child is None or self.right_child is None:
            return None

        if self.op.value in dunder_funcs:
            fn_name = dunder_funcs[self.op.value]
            return FuncCallNode(self.repr_token, VariableNode(fn_name), (self.left_child, self.right_child)).update(ctx, self.parent)
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

    def update(self, ctx: Context, parent: Optional[Node], use_dunder_func=True, func=False) -> Optional['DotOperatorNode']:
        self.scope_level = ctx.scope_level
        self.parent = parent
        self.var = self.var.update(ctx, self.parent)
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
            self.offset = OffsetNode(self.var.offset, field.offset.offset)
            self.type = field.type

        elif isinstance(field, ClassDefineNode):
            self.type = class_type

        elif isinstance(field, FuncDefineNode):
            self.type = field.func_to_type()

        else:
            ctx.error(TypeError.no_attribute, self, self.var.type.name.value, self.field.name)
            return None

    def alloc_vars(self, ctx: Context) -> None:
        self.var.alloc_vars(ctx)
        return

    def get_name(self) -> Node:
        return Node(self.field.repr_token)

    def __repr__(self):
        return f'{self.var}.{self.field}'


class IndexOperatorNode(VariableNode):
    def __init__(self, repr_tok: Token, collection: ExpressionNode, index: ExpressionNode):
        super().__init__(repr_tok)
        self.collection: ExpressionNode = collection
        self.index: ExpressionNode = index
        return

    def update(self, ctx: Context, parent: Optional[Node], use_dunder_func=True, func=False) -> Optional['IndexOperatorNode']:
        self.scope_level = ctx.scope_level
        self.parent = parent
        self.collection = self.collection.update(ctx, self.parent)
        self.index = self.index.update(ctx, self.parent)

        if self.collection is None or self.index is None:
            return None

        optimised = self.optimise_ct(ctx)
        if optimised is not None:
            return optimised

        if use_dunder_func:
            fn_name = dunder_funcs[Operators.index.value.value]
            return FuncCallNode(self.repr_token, VariableNode(fn_name), (self.collection, self.index)).update(ctx, self.parent)
        else:
            return self

    def optimise_ct(self, ctx: Context):
        if not isinstance(self.index, ValueNode):
            return None

        if self.collection.type == Types.dict.value:
            if isinstance(self.collection, ValueNode):  # can access the element at compile time
                if self.index.repr_token.value in self.collection.repr_token.value:
                    return self.collection.repr_token.value[self.index.repr_token.value]
                else:
                    ctx.error(TypeError.no_element, self)

        elif self.collection.type in {Types.str.value, Types.array.value}:
            if isinstance(self.index.repr_token.value, int):
                if not (0 <= abs(self.index.repr_token.value) < len(self.collection.repr_token.value)): # bound check
                    ctx.error(TypeError.out_of_bounds, self.index)

                if isinstance(self.collection, ValueNode):  # can access the element at compile time
                    return self.collection.repr_token.value[self.index.repr_token.value]
                else:
                    return self    # can return an unsafe array access, and it will be just fine

        return None

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
        self.func: Optional[FuncDefineNode] = None

    def get_id(self) -> tuple[str, tuple[Type, ...]]:
        args: list[Type] = []
        for arg in self.args:
            if arg.type is not None:
                args.append(arg.type)
            continue
        return self.func_name.name, tuple(args)

    def update(self, ctx: Context, parent: Optional[Node]) -> Optional['FuncCallNode']:
        self.scope_level = ctx.scope_level
        self.parent = parent
        self.func_name.update(ctx, self.parent, func=True)

        args = []
        if self.func_name.type is not None and self.func_name.type == class_type:   # it's a constructor
            ref = RefNode(self.func_name)
            ref.offset = OffsetNode(Registers.SP, 0)
            ref.type = ctx.get_class_by_name(self.func_name.name).type  # adding type field for func signature
            args.append(ref)
            self.func_name.name = constructor_name_tok.value

        if isinstance(self.func_name, DotOperatorNode):     # syntax sugar
            if self.func_name.var.type != class_type:       # it's an instance and not the class itself
                ref = RefNode(self.func_name.var)
                ref.type = self.func_name.var.type
                args.append(ref)

        for arg in self.args:
            args.append(arg.update(ctx, self.parent))
        self.args = tuple(args)

        self.func = ctx.get_func(self)
        if self.func is None:
            ctx.error(TypeError.undefined_function, self.func_name.get_name())
            return None
        self.type = self.func.ret_type
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
    def __init__(self, repr_tok: Token, var_type: Type, var_name: VariableNode, value: Optional[ExpressionNode] = None, static: Optional[Token] = None):
        super().__init__(repr_tok, var_name)
        self.type: Type = var_type
        self.value: Optional[ExpressionNode] = value
        self.offset: OffsetNode = OffsetNode(Registers.BP, 0)
        self.class_def: Optional[ClassDefineNode] = None
        self.static_tok: Optional[Token] = static
        self.static: bool = static is not None

    def add_ctx(self, ctx: Context) -> None:
        ctx.vars[self.name.name] = self
        return

    def update(self, ctx: Context, parent: Optional[Node]) -> Optional['VarDefineNode']:
        self.scope_level = ctx.scope_level
        self.parent = parent

        if self.static and self.class_def is None:
            ctx.error(TypeError.static_not_in_class, Node(self.static_tok))
            self.static = False

        self.add_ctx(ctx)
        self.name = self.name.update(ctx, self.parent)
        if self.name is None:
            return None

        if self.value is not None:
            self.value = self.value.update(ctx, self.parent)
            if self.value is not None:
                self.name.type = self.type = self.value.type

        return self

    def alloc_vars(self, ctx: Context) -> None:
        if self.value is not None:
            self.value.alloc_vars(ctx)
        self.offset.offset = ctx.stack_location(self.name)
        ctx.stack_dealloc(self.name)    # first occurrence of this variable, prior to this point, this slot can be used
        return

    def __repr__(self):
        if self.value is None:
            return f'loc:({self.offset}) {self.type} {self.name.repr_token.value}'
        else:
            return f'loc:({self.offset}) {self.type} {self.name.repr_token.value} = {self.value}'


class FuncDefineNode(NameDefineNode):
    def __init__(self, repr_tok: Token, ret_type: Type, func_name: VariableNode, params: tuple[VarDefineNode, ...], body: 'IsolatedScopeNode', static: Optional[Token] = None):
        super().__init__(repr_tok, func_name)
        self.ret_type: Type = ret_type
        self.ret_dest: Optional[VariableNode] = None
        self.params: tuple[VarDefineNode, ...] = params
        self.body: IsolatedScopeNode = body
        self.class_def: Optional[ClassDefineNode] = None
        self.static_tok: Optional[Token] = static
        self.static: bool = static is not None
        return

    def get_id(self) -> tuple[str, tuple[Type, ...]]:
        params: list[Type] = []
        for param in self.params:
            if param.type is not None:
                params.append(param.type)
        return self.name.name, tuple(params)

    def func_to_type(self) -> Type:
        params: list[Type] = []
        for param in self.params:
            if param.type is not None:
                params.append(param.type)

        return Type(self.name.repr_token, tuple(params))

    def get_func_label(self) -> str:
        string = self.name.name
        for param in self.params:
            string += f'.{param.type.get_type_label()}'
        return string

    def add_ctx(self, ctx: Context) -> None:
        ctx.funcs[self.get_id()] = self
        return

    def create_context(self, ctx: Context) -> Context:
        """Creates a new context for parameters and body via ctx.clone(), but isolates the variables defined in upper scope
        :param ctx: Context of the above scope
        :return: the new context of the current scope"""
        new_ctx = ctx.clone()

        new_ctx.stack_map = {}  # erasing pre existing variables
        new_ctx.vars = {}       # same here
        return new_ctx

    def update(self, ctx: Context, parent: Optional[Node]) -> Optional['FuncDefineNode']:
        self.scope_level = ctx.scope_level
        self.parent = parent

        lower_ctx = self.create_context(ctx)

        params = []
        if self.static and self.class_def is None:
            ctx.error(TypeError.static_not_in_class, Node(self.static_tok))
            self.static = False
        elif not self.static and self.class_def is not None:
            self_param = self.get_self_param()
            self_param.add_ctx(lower_ctx)
            params.append(self_param)

        for param in self.params:
            param = param.update(lower_ctx, self)
            if param is not None:
                params.append(param)
        self.params = tuple(params)

        self.add_ctx(ctx)

        self.body = self.body.update(lower_ctx, self)
        if self.body is None:
            return None

        self.body.alloc_vars(lower_ctx)   # allocating local variables on the stack

        i = 0   # allocating arguments on the stack, above the BP
        for param in reversed(self.params):
            param.offset = OffsetNode(Registers.BP, i)
            i += param.get_size()

        self.ret_dest = VariableNode(self.repr_token)
        self.ret_dest.offset = OffsetNode(Registers.BP, i)
        self.ret_dest.type = copy(any_type)
        return self

    def get_self_param(self) -> VarDefineNode:
        return VarDefineNode(self_tok, self.class_def.type, VariableNode(self_tok))

    def inline_func(self, func_call: FuncCallNode) -> ScopeNode:
        body: list[Node] = []
        for param, arg in zip(self.params, func_call.args):
            param = deepcopy(param)
            param.value = arg
            body.append(param)

        if isinstance(self.body, ScopeNode):
            for node in self.body.child_nodes:
                if isinstance(node, ReturnNode):
                    body.append(AssignNode(self.ret_dest, deepcopy(node.value)))
                    body.append(BreakNode(func_call.repr_token))
                else:
                    body.append(deepcopy(node))
        elif isinstance(self.body, ReturnNode):
            body.append(AssignNode(self.ret_dest, deepcopy(self.body.value)))
            body.append(BreakNode(func_call.repr_token))
        else:
            body.append(deepcopy(self.body))


        body_node = ScopeNode(body[0].repr_token, body)
        body_node.scope_level = func_call.scope_level
        body_node.parent = func_call.parent
        return body_node

    def __repr__(self):
        return f'{self.ret_type} {self.name}({self.params.__repr__()[1:-1]}) {self.body}'


class IsolatedScopeNode(ScopeNode):
    def __init__(self, start_tok: Token, child_nodes: list[Node]):
        super().__init__(start_tok, child_nodes)
        return

    @classmethod
    def new_from_old(cls, old: ScopeNode) -> 'IsolatedScopeNode':
        return cls(old.repr_token, old.child_nodes)

    def update(self, ctx: Context, parent: Optional[Node]) -> Optional['IsolatedScopeNode']:
        self.scope_level = ctx.scope_level
        self.parent = parent
        self.process_body(ctx)
        return self


class ClassDefineNode(NameDefineNode):
    def __init__(self, repr_tok: Token, name: 'VariableNode', c_type: Type, body: 'ClassBodyNode'):
        super().__init__(repr_tok, name)
        self.body: ClassBodyNode = body
        self.type: Type = c_type
        self.size: int = 0

        self.body.class_def = self
        return

    def get_up_class_def(self) -> Optional['ClassDefineNode']:
        return self

    def update(self, ctx: Context, parent: Optional[Node]) -> Optional['ClassDefineNode']:
        self.scope_level = ctx.scope_level
        self.parent = parent
        self.add_ctx(ctx)

        self.body = self.body.update(ctx, self)
        if self.body is None:
            return None

        self.type.size = self.size = self.body.size
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
    def __init__(self, start_tok: Token, child_nodes: list[Node], static: Optional[Token] = None):
        super().__init__(start_tok, child_nodes)
        self.funcs: dict[tuple, FuncDefineNode] = {}
        self.types: dict[Type, ClassDefineNode] = {}
        self.class_def: Optional[ClassDefineNode] = None

        self.vars: dict[str, (MacroDefineNode, VarDefineNode)] = {}
        self.offset_map: dict[str, int] = {}
        self.size: int = 0

        self.static_vars: dict[str, (MacroDefineNode, VarDefineNode)] = {}
        self.static_offset_map: dict[str, int] = {}
        self.static_size: int = 0

        self.static_tok: Optional[Token] = static
        self.static: bool = static is not None
        return

    @classmethod
    def new_from_old(cls, old: ScopeNode, static) -> 'ClassBodyNode':
        return cls(old.repr_token, old.child_nodes, static)

    def process_body(self, ctx: Context) -> None:
        child_nodes = []
        for node in self.child_nodes:
            if isinstance(node, VarDefineNode):
                if not node.static:
                    node.static = self.static
                    node.static_tok = self.static_tok

                node.class_def = self.class_def
                node = node.update(ctx, self.parent)
                if node is None:
                    continue
                if node.static:
                    self.static_vars[node.get_id()] = node
                    self.static_offset_map[node.get_id()] = self.static_size
                    self.static_size += node.type.size
                else:
                    self.vars[node.get_id()] = node
                    self.offset_map[node.get_id()] = self.size
                    self.size += node.type.size

            elif isinstance(node, MacroDefineNode):
                node = node.update(ctx, self.parent)
                if node is None:
                    continue
                self.vars[node.get_id()] = node

            elif isinstance(node, ClassDefineNode):
                node = node.update(ctx, self.parent)
                if node is None:
                    continue
                self.types[node.get_id()] = node

            elif isinstance(node, FuncDefineNode):
                if not node.static:
                    node.static = self.static
                    node.static_tok = self.static_tok

                node.class_def = self.class_def
                node = node.update(ctx, self.parent)
                if node is None:
                    continue
                self.funcs[node.get_id()] = node

            else:
                ctx.error(TypeError.def_statement_expected, node)
                continue

            child_nodes.append(node)

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

    def update(self, ctx: Context, parent: Optional[Node]) -> Optional['ClassBodyNode']:
        self.scope_level = ctx.scope_level
        self.parent = parent
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

    def update(self, ctx: Context, parent: Optional[Node]) -> Optional['IfNode']:
        self.scope_level = ctx.scope_level
        self.parent = parent

        self.condition = self.condition.update(ctx, self.parent)

        self.body = self.body.update(ctx, self.parent)
        if self.body is None or self.condition is None:
            return None

        if self.else_statement is not None:
            self.else_statement = self.else_statement.update(ctx, self.parent)

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


class ElseNode(Node):
    def __init__(self, repr_tok: Token, body: Node, if_statement: Optional[IfNode]):
        super().__init__(repr_tok)
        self.body = body
        self.if_statement: Optional[IfNode] = if_statement
        return

    def update(self, ctx: Context, parent: Optional[Node]) -> Optional['ElseNode']:
        self.scope_level = ctx.scope_level
        self.parent = parent

        self.body = self.body.update(ctx, self.parent)
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

    def get_loop(self) -> Optional['WhileNode']:
        parent = self.parent
        while not (parent is None or isinstance(parent, WhileNode)):
            parent = parent.parent

        return parent

    def update(self, ctx: Context, parent: Optional[Node]) -> Optional['WhileNode']:
        self.scope_level = ctx.scope_level
        self.parent = parent

        self.condition = self.condition.update(ctx, self)

        self.body = self.body.update(ctx, self)
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

    def update(self, ctx: Context, parent: Optional[Node]) -> Optional['Node']:
        self.scope_level = ctx.scope_level
        self.parent = parent
        self.value = self.value.update(ctx, self.parent)
        if self.value is None:
            return None
        return self

    def __repr__(self):
        if self.value is None:
            return f'return'
        else:
            return f'return {self.value}'


class LoopModifierNode(Node):
    def __init__(self, repr_tok: Token, value: Optional[Token], error: TypeError):
        super().__init__(repr_tok)
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
        self.scope_level = ctx.scope_level
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

class RefNode(VariableNode):
    def __init__(self, var: VariableNode):
        super().__init__(var.repr_token)
        self.var: VariableNode = var
        self.type = any_type
        return

    def alloc_vars(self, ctx: Context) -> None:
        self.var.alloc_vars(ctx)
        return

    def __repr__(self):
        return f'&{self.name}'


class InstructionNode(Node):
    def __init__(self, repr_tok: Token):
        super().__init__(repr_tok)
        return


class OffsetNode(InstructionNode):
    def __init__(self, base, offset: int):
        if isinstance(base, Registers):
            super().__init__(base.value)
        else:
            super().__init__(base.repr_tok)
        self.base = base
        self.offset: int = offset
        return

    def __add__(self, other):
        return OffsetNode(self.base, self.offset+other)

    def __repr__(self):
        sign = '' if self.offset < 0 else '+'

        if isinstance(self.base, Registers):
            return f'reg:{self.base.value.value}{sign}{self.offset}'
        else:
            return f'{self.base}{sign}{self.offset}'


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
