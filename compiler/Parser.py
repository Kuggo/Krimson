from ASTNodes import *

# functions

def get_void_value(tok: Token) -> ValueNode:
    return ValueNode(VoidLiteral(tok.location))


class SyntaxError(Enum):
    """Enum containing all the custom error messages that the parser may generate."""
    identifier_expected = 'Identifier expected'
    symbol_expected = "'{}' expected"
    symbol_expected_before = "'{}' expected before '{}'"
    expression_expected = 'Expression expected'
    type_expected = 'type expected'
    typedef_expected = 'type declaration expected'
    cannot_assign_to_non_var = '"{}" is not a variable that can be assigned a value to'
    if_expected = 'if keyword expected before else'
    statement_expected = 'Statement expected'
    arg_expected = 'Function argument definition expected'
    not_a_func = 'Cannot call "{}"'
    cannot_index_multiple = "Cannot index with multiple values. ']' expected"
    cannot_assign_to_arg = "Cannot assign values to function argument definitions"
    declaration_expected = "Declaration statement expected"
    func_literal_expected = "Function literal expected"
    generic_expected = "Generic type/value expected (Needs to be known at compile-time)"


class Parser:
    """Parses a collection of tokens generated by the Lexer

    By convention, each function should take care of its tokens and leave it ready for the next function to start
    reading the next token right away"""

    def __init__(self, tokens: list[Token], global_vars: Globals):
        self.tokens: list[Token] = tokens + [Separators.rcb.value, Separators.eof.value]
        """the token collection"""

        self.global_vars: Globals = global_vars
        """Object that holds the variables related to the input code"""

        self.i: int = 0
        """index on the token input of the current token"""

        self.peak: Token = self.tokens[self.i]
        """current token pointed by index"""

        self.length = len(self.tokens)
        """length of the input token collection. To avoid calling ``len()`` on ``self.has_next()`` multiple times"""

        self.last = None
        """previous token pointed by index"""

        self.ast: Optional[ScopeNode] = None
        """output ScopeNode generated upon calling ``self.parse()``"""

        self.errors: list[Error] = []
        """collection of Errors found by the Parser upon calling ``self.parse()``"""
        return

    def parse(self):
        """
        Entry function for the whole parsing process. The program is parsed as a scoped node (stream of statements).

        program : scope
        """
        if len(self.tokens) > 0:
            self.ast = self.scope()
        return

    # rules
    def scope(self) -> ScopeNode:
        """
        Parses the next statements until it finds its end and returns the ScopeNode
        
        The end is the ``}`` separator
        
        scope : {statement}

        :return: ScopeNode
        """

        start_tok = self.peak
        statements = self.repeat_until_symbol(Separators.rcb.value.value, Parser.statement)
        self.advance()

        return ScopeNode(statements, self.peak.location - start_tok.location)

    def statement(self) -> Optional[Node]:
        """
        Parses the next statement until it find its end and returns the Node corresponding to the statement, or None if
        the statement contains syntax errors

        The end is a separator token such as ',' ';' ':', a new keyword, or the end of an expression

        statement : expression
                  | while_statement
                  | if_statement
                  | exit_statement
                  | skip_statement
                  | scope_statement
                  | assign_statement
                  | var_define_statement
                  | func_define_statement
                  | class_define_statement

        :param: last_statement: The last statement processed on the scope
        :return: Node or None if an error occurred
        """
        if self.peak.tt == TT.KEYWORD:
            if self.peak.value == 'while':
                return self.while_statement()
            elif self.peak.value == 'if':
                return self.if_statement()
            elif self.peak.value == 'else':
                self.error(SyntaxError.if_expected, self.peak.location)
                return None
            elif self.peak.value == 'exit':
                return self.exit_statement()
            elif self.peak.value == 'skip':
                return self.skip_statement()
            elif self.peak.value == 'match':
                return self.match_statement()
            else:
                assert False

        elif self.peak == Separators.lcb.value:
            self.advance()
            return self.scope()

        else:
            return self.expression()

    def expression(self, precedence=-1) -> Optional[ExpressionNode]:
        left = self.parse_prefix()
        if left is None:
            return None

        node = left

        while self.peak.value in OPERATOR_PRECENDENCE and \
                (OPERATOR_PRECENDENCE[self.peak.value] > precedence or
                 (self.peak.value in RIGHT_ASSOCIATIVE_OPERATORS and OPERATOR_PRECENDENCE[self.peak.value] == precedence)):

            node = self.parse_infix(left)
            if node is not None:
                left = node

        while self.peak.value in END_OF_EXPRESSION:
            self.advance()

        return node

    def parse_prefix(self) -> Optional[ExpressionNode]:
        token = self.peak
        self.advance()

        if isinstance(token, Literal):
            return ValueNode(token)

        elif token.tt == TT.IDENTIFIER:
            return VariableNode(token)

        elif token.tt == TT.OPERATOR and token.value in UNARY_OPERATORS:
            expression = self.expression(precedence=OPERATOR_PRECENDENCE[token.value])
            if expression is None:
                return None
            return UnOpNode(token, expression)

        elif token == Separators.lpa.value:
            if self.peak == Separators.rpa.value:
                self.advance()
                return get_void_value(self.last)
            expression = self.expression()
            if expression is None:
                return None
            self.advance()
            return expression

        elif token == Separators.lbr.value:   # array literal
            return self.array_literal()

        elif token == Separators.lcb.value:   # product type instance
            values = self.repeat_until_symbol(Separators.rcb.value.value, Parser.expression,
                                              SyntaxError.expression_expected, OPERATOR_PRECENDENCE[Separators.comma.value.value])
            self.advance()
            if len(values) == 0:
                return None
            return ValueNode(ProductTypeLiteral(values))

        self.rollback()
        self.error(SyntaxError.expression_expected, self.peak.location)
        return None

    def parse_infix(self, left: ExpressionNode) -> Optional[ExpressionNode]:
        token = self.peak
        self.advance()

        if token == Separators.lpa.value: # function call
            return self.make_function_call(left)

        elif token == Operators.dot.value: # attribute access
            if self.peak.tt != TT.IDENTIFIER:
                self.error(SyntaxError.identifier_expected, self.peak.location)
                return None
            right = VariableNode(self.peak)
            return DotOperatorNode(token, left, right)

        elif token == Separators.colon.value: # name define
            return self.name_define_statement(left)

        elif token == Operators.fn.value:
            return self.func_literal(left)

        elif token == Separators.lbr.value:   # indexing
            return self.index(left)

        elif token == Separators.comma.value:   # tuple
            return self.tuple_literal(left)

        elif token == Operators.assign.value:
            right = self.expression(precedence=OPERATOR_PRECENDENCE[token.value])
            if right is None:
                return None
            return AssignNode(left, right)

        elif token.value not in OPERATOR_PRECENDENCE:
            self.rollback()
            return None

        right = self.expression(precedence=OPERATOR_PRECENDENCE[token.value])
        if right is None:
            return None
        return BinOpNode(token, left, right)

    def make_function_call(self, function: ExpressionNode) -> Optional[FuncCallNode]:
        args = self.repeat_until_symbol(Separators.rpa.value.value, Parser.expression, SyntaxError.expression_expected,
                                            OPERATOR_PRECENDENCE[Separators.comma.value.value])
        self.advance()
        return FuncCallNode(function, args)

    def index(self, left) -> Optional[IndexOperatorNode]:
        """
        Parses the next expression until the end of indexing was found.

        If no expression is found, or more than 1 expression is found, then an error is detected.

        :param left: ExpressionNode to be indexed
        :return: ExpressionNode or None if an error occurred
        """
        tok = self.last

        index = self.expression()
        self.advance()
        return IndexOperatorNode(tok, left, index)

    # types

    def type(self, precedence=-1) -> Optional[Type]:
        left = self.type_prefix()
        if left is None:
            return None

        t = left

        while self.peak.value in TYPE_OP_PRECEDENCE and (TYPE_OP_PRECEDENCE[self.peak.value] > precedence or
                 (self.peak.value in RIGHT_ASSOCIATIVE_OPERATORS and TYPE_OP_PRECEDENCE[self.peak.value] == precedence)):

            t = self.type_infix(left)
            if t is not None:
                left = t

        while self.peak.value in END_OF_EXPRESSION:
            self.advance()

        return t

    def type_prefix(self) -> Optional[Type]:
        tok = self.peak
        self.advance()

        if tok.tt == TT.LITERAL:
            assert tok == Types.void.value
            return VoidType()

        elif tok.tt == TT.IDENTIFIER:
            return Type(tok)

        elif tok.value in UNARY_OPERATORS:
            t = self.type(precedence=OPERATOR_PRECENDENCE[tok.value])
            if t is None:
                return None
            return # We don't have any unary operators for types

        elif tok == Separators.lpa.value:
            if self.peak == Separators.rpa.value:
                self.advance()
                return VoidType()
            t = self.type()
            if t is None:
                return None
            self.advance()
            return t

        elif tok == Separators.lbr.value:  # array type
            start_tok = self.peak
            t = self.repeat_until_symbol(Separators.rbr.value.value, Parser.type, SyntaxError.type_expected,
                                         OPERATOR_PRECENDENCE[Separators.lbr.value.value])

            if len(t) == 0:
                self.error(SyntaxError.expression_expected, start_tok.location)
                return None

            if len(t) > 1:
                self.error(SyntaxError.symbol_expected, t[1].location, Separators.lbr.value.value)
                # error but accept the first expression

            self.advance()
            return ArrayType(t[0])

        self.rollback()
        self.error(SyntaxError.type_expected, self.peak.location)
        return None

    def type_infix(self, left: Type) -> Optional[Type]:
        t = self.peak

        if t == Separators.comma.value:
            return self.tuple_type(left)

        elif t == Separators.lbr.value:
            generics = []
            while self.peak != Separators.rpa.value:
                t = self.type(precedence=OPERATOR_PRECENDENCE[Separators.comma.value.value])
                if t is None:
                    return None
                generics.append(t)
            # left.generics = generics  # TODO: remove comment when generics are added
            return

        elif t == Operators.fn.value:
            return self.func_type(left)

        elif t == Operators.opt.value:
            pass    # TODO

        elif t.value not in OPERATOR_PRECENDENCE:
            return None

        # self.type(precedence=OPERATOR_PRECENDENCE[t.value])
        # later supporting | for in place sum types?
        return None

    def tuple_type(self, left) -> Optional[TupleType]:
        types = [left]
        while self.peak == Separators.comma.value and self.preview() != Separators.rbr.value:
            self.advance()
            t = self.type(OPERATOR_PRECENDENCE[Separators.comma.value.value])
            if t is not None:
                types.append(t)

        return TupleType(types)

    def func_type(self, left: Type) -> Optional[FunctionType]:
        tok = self.peak
        self.advance()
        right = self.type(precedence=OPERATOR_PRECENDENCE[Operators.fn.value.value])
        if right is None:
            self.error(SyntaxError.type_expected, tok.location)
            return None

        return FunctionType(left, right)

    # defines

    def name_define_statement(self, name: ExpressionNode) -> Optional[NameDefineNode]:
        """
        Parses the next statement (that does not start with a keyword) without knowing what kind of statement it is.
        The difference is spotted on how the statement goes.

        - It is a func_define if: IDENTIFIER [:] fn Type
        - It is a type_define if: IDENTIFIER [:] type Type
        - It is a var_define if not the above: IDENTIFIER [:] Type

        :return: var_define_statement | func_define_statement | None if an expression/error occurred
        """

        if not isinstance(name, VariableNode):
            self.error(SyntaxError.identifier_expected, name.location)
            return None

        t = self.type(OPERATOR_PRECENDENCE[Separators.colon.value.value])
        if t is None:
            return None

        if t == Types.type.value:  # it's a typedef
            return self.type_define_statement(name)

        # it's not using special syntax to define a new name, so it can be any variable type
        return VarDefineNode(name, t)

    def type_define_statement(self, name: VariableNode) -> Optional[TypeDefineNode]:
        """
        Parses the next type declaration until its end is found, and returns its Node
        :param name: name of the new type
        :return: TypeDefineNode or None if an error occurred
        """

        if self.peak != Operators.assign.value:
            self.error(SyntaxError.symbol_expected, self.peak.location, Operators.assign.value.value)
            return None
        self.advance()

        if isinstance(name, IndexOperatorNode): # it has generics
            if isinstance(name.index, ValueNode) and isinstance(name.index.value, TupleLiteral):
                generics = name.index.value.value
            elif isinstance(name.index, ValueNode) or isinstance(name.index, VariableNode):
                generics = [name.index]
            else:
                self.error(SyntaxError.expression_expected, name.index.location)
                return None
        else:
            generics = None

        if self.peak != Separators.lcb.value:  # type alias
            t = self.type()
            if t is None:
                return None
            return TypeAliasDefineNode(name, t, generics)

        self.advance()
        if self.peak == Operators.b_or.value:  # accepting the first one for stylishness
            self.advance()
            return self.sum_type(name, generics)

        if self.peak.tt != TT.IDENTIFIER:
            self.error(SyntaxError.identifier_expected, self.peak.location)
            return None

        if self.preview() != Separators.colon.value or self.preview(2) == Types.type.value.name_tok:  # it's a sum type
            return self.sum_type(name, generics)

        fields = self.repeat_until_symbol(Separators.rcb.value.value, Parser.field_define, SyntaxError.declaration_expected)
        self.advance()
        return ProductTypeDefineNode(name, fields, generics)

    def field_define(self) -> Optional[VarDefineNode]:
        name = self.peak
        if name.tt != TT.IDENTIFIER:
            return None

        self.advance()

        if self.peak == Separators.colon.value:
            self.advance()
        else:
            pass  # type inference

        t = self.type()
        if t is None:
            return None

        return VarDefineNode(VariableNode(name), t)

    def sum_type(self, name: VariableNode, generics=None) -> Optional[SumTypeDefineNode]:
        def variant() -> Optional[TypeDefineNode]:
            if self.peak.tt != TT.IDENTIFIER:
                return None
            variant_name = VariableNode(self.peak)
            self.advance()

            if self.peak == Separators.colon.value and self.preview() == Types.type.value:
                self.advance(2)

            if self.peak == Operators.b_or.value or self.peak == Separators.rcb.value:   # no size variant
                return TypeDefineNode(variant_name)

            return self.type_define_statement(variant_name)

        variants = []
        while self.peak != Separators.rcb.value:
            v = variant()
            if v is not None:
                variants.append(v)
            else:
                self.advance()
            if self.peak == Operators.b_or.value:
                self.advance()

        self.advance()
        return SumTypeDefineNode(name, variants, generics)

    # keywords

    def while_statement(self) -> Optional[WhileNode]:
        """
        Parses the next while statement until it finds its end

        :return: WhileNode or None if an error was found
        """
        start = self.peak
        self.advance()

        condition = self.expression()
        if condition is None:
            self.error(SyntaxError.expression_expected, self.peak.location)
            return None

        body = self.statement()
        if body is None:
            return None

        return WhileNode(start, condition, body)

    def if_statement(self) -> Optional[IfNode]:
        """
        Parses the next if statement until it finds its end. It does not check for else statement after it is done.

        :return: IfNode or None if an error was found
        """
        start = self.peak
        self.advance()

        condition = self.expression()
        if condition is None:
            self.error(SyntaxError.expression_expected, self.peak.location)
            return None

        body = self.statement()
        if body is None:
            return None

        if_node = IfNode(start, condition, body)

        if self.peak == Keywords.else_.value:
            self.else_statement(if_node)

        return if_node

    def else_statement(self, if_node: Optional[Node]) -> Optional[ElseNode]:
        """
        Parses the next else statement until its end

        :param: if_node: the if statement the else statement is bound to
        :return: ElseNode or None if error was found (no previous if statement error)
        """

        start = self.peak
        self.advance()
        body = self.statement()
        if body is None:
            return None

        else_node = ElseNode(start, body, if_node)
        if_node.else_statement = else_node

        return else_node

    def exit_statement(self) -> ExitNode:
        """
        Parses the exit statement until it finds its end.

        exit_statement : 'exit' [literal]

        :return: ExitNode
        """

        start = self.peak
        self.advance()
        value = None

        if self.peak.tt == TT.LITERAL and isinstance(self.peak.value, int):
            value = self.peak
            self.advance()

        return ExitNode(start, value)

    def skip_statement(self) -> SkipNode:
        """
        Parses the skip statement until it finds its end.

        skip_statement : 'skip' [literal]

        :return: SkipNode
        """

        start = self.peak
        self.advance()
        value = None
        if self.peak.tt == TT.LITERAL and isinstance(self.peak.value, int):
            value = self.peak
            self.advance()

        return SkipNode(start, value)

    def match_statement(self) -> Optional[MatchNode]:
        tok = self.peak
        self.advance()

        value = self.expression()
        if value is None:
            self.error(SyntaxError.expression_expected, tok.location)
            return None

        if self.peak == Separators.lcb.value:
            self.advance()
            cases = self.repeat_until_symbol(Separators.rcb.value.value, Parser.match_case, SyntaxError.statement_expected)
            self.advance()
        else:
            case = self.match_case()
            if case is None:
                return None
            cases = [case]

        return MatchNode(tok, value, cases)

    def match_case(self) -> Optional[CaseNode]:
        variant = self.peak
        if variant.tt != TT.IDENTIFIER:
            self.error(SyntaxError.identifier_expected, variant.location)
            return None
        self.advance()

        # do we need any separator?
        if self.peak == Separators.colon.value or self.peak == Operators.fn.value:  # allowing : and -> atm
            self.advance()

        body = self.statement()
        if body is None:
            return None

        return CaseNode(VariableNode(variant), body)

    # literals

    def func_literal(self, left: ExpressionNode) -> Optional[ValueNode]:
        """Parses the next function literal until it finds its end

        func_literal : VarDefineNode '->' VarDefineNode statement
        """

        tok = self.peak
        right = self.expression()
        if right is None:
            self.error(SyntaxError.expression_expected, tok.location)
            return None

        tok = self.peak
        body = self.statement()
        if body is None:
            self.error(SyntaxError.statement_expected, tok.location)
            return None

        func = FunctionLiteral(left, right, body)
        return ValueNode(func)

    def tuple_literal(self, left: ExpressionNode) -> Optional[ValueNode]:
        values = [left]
        self.rollback()
        while self.peak == Separators.comma.value and self.preview() != Separators.rbr.value:
            self.advance()
            t = self.expression(OPERATOR_PRECENDENCE[Separators.comma.value.value])
            if t is not None:
                values.append(t)

        return ValueNode(TupleLiteral(values))

    def array_literal(self) -> Optional[ValueNode]:
        """
        Parses and constructs an Array Literal from the next tokens and returns it.

        array : '[' {expression} ']'

        :return: ValueNode array literal
        """

        elements = self.repeat_until_symbol(Separators.lbr.value.value, Parser.expression,
                                            SyntaxError.expression_expected, OPERATOR_PRECENDENCE[Separators.comma.value.value])
        self.advance()
        if len(elements) == 0:
            return None

        return ValueNode(ArrayLiteral(elements))

    # utils

    def repeat_until_symbol(self, end_symbol: str, method, error: Optional[SyntaxError] = None, *args) -> list:
        """
        Parses an unknown amount of statements/expressions.

        Given the Separator token to stop parsing at, this method will continue to loop for multiple occurrences of the
        rule specified in the method passed. If error is specified then an error will be generated if the rule was not
        found.

        :param end_symbol: the separator character the pattern is meant to end at
        :param method: the Parser method that checks/parses the rule.
        :param error: if specified, what error message will be generated by not finding the rule expected
        :return: list of nodes generated by the method
        """

        output = []
        while self.has_next() and self.peak.value != end_symbol:
            tok = self.peak
            element = method(self, *args)

            if element is None:
                if error is not None:
                    self.error(error, tok.location)
            else:
                output.append(element)

            while self.has_next() and self.peak.value in END_OF_EXPRESSION:
                self.advance()

        if not self.has_next():
            self.error(SyntaxError.symbol_expected, self.last.location, end_symbol)

        return output

    def error(self, error: SyntaxError, location: FileRange, *args) -> None:
        """
        Creates a new Error and adds it to the error collection of tokens ``self.errors``

        :param error: the error found
        :param location: the token where it occurred
        :param args: extra arguments for custom error message formatting
        """
        self.errors.append(Error(error, location, self.global_vars, *args))

    def advance(self, i=1) -> None:
        """
        Moves the index to i tokens ahead and updates all state variables

        :param i: number of tokens to advance
        """
        if self.i + i < len(self.tokens):
            self.i += i
            self.last = self.peak
            self.peak = self.tokens[self.i]
        else:
            self.peak = Separators.eof.value

    def rollback(self, i=1) -> None:
        """
        Moves the index to i tokens behind and updates all state variables

        :param i: number of tokens to rollback
        """
        self.i -= i
        if self.i > 0:
            self.last = self.tokens[self.i - 1]
        else:
            self.last = None
            self.i = 0
        self.peak = self.tokens[self.i]
        return

    def preview(self, i=1) -> Token:
        """
        returns the token after ``i`` positions of current token

        :param i: offset of the position to look ahead
        :return: the token at the specified position
        """

        if self.has_next(i):
            return self.tokens[self.i + i]
        else:
            return Separators.eof.value

    def has_next(self, i=0) -> bool:
        """
        Checks if there are more tokens to be processed ``i`` tokens after the current index

        :param i: offset of current index
        :return: True if there are more tokens to be processed, False otherwise
        """
        return self.peak != Separators.eof.value and self.i + i < self.length
