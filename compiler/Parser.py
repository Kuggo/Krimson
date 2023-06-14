from ASTNodes import *


class SyntaxError(Enum):
    """Enum containing all the custom error messages that the parser may generate."""
    identifier_expected = 'Identifier expected'
    symbol_expected = "'{}' expected"
    symbol_expected_before = "'{}' expected before '{}'"
    expression_expected = 'Expression expected'
    type_expected = 'type expected'
    cannot_assign_to_non_var = '"{}" is not a variable that can be assigned a value to'
    if_expected = 'if keyword expected before else'
    statement_expected = 'Statement expected'
    arg_expected = 'Function argument definition expected'
    not_a_func = 'Cannot call "{}"'
    cannot_index_multiple = "Cannot index with multiple values. ']' expected"
    cannot_assign_to_arg = "Cannot assign values to function argument definitions"
    # class_body_not_a_scope = "Class body must be defined between curly brackets { }"
    declaration_expected = "Declaration statement expected"


class Parser:
    """Parses a collection of tokens generated by the Lexer

    By convention, each function should take care of its tokens and leave it ready for the next function to start
    reading the next token right away"""

    def __init__(self, tokens: list[Token]):
        self.tokens: list[Token] = tokens + [Separators.rcb.value, Separators.eof.value]
        """the token collection"""

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

        return ScopeNode(start_tok, statements)

    def statement(self) -> Optional[Node]:
        """
        Parses the next statement until it find its end and returns the Node corresponding to the statement, or None if
        the statement contains syntax errors

        The end is a separator token such as ',' ';' ':', a new keyword, or the end of an expression

        statement : expression
                  | while_statement
                  | if_statement
                  | return_statement
                  | break_statement
                  | skip_statement
                  | scope_statement
                  | var_assign_statement
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
                self.error(SyntaxError.if_expected, self.peak)
                return None
            elif self.peak.value == 'return':
                return self.return_statement()
            elif self.peak.value == 'break':
                return self.break_statement()
            elif self.peak.value == 'skip':
                return self.skip_statement()
            else:
                assert False

        elif self.peak == Separators.lcb.value:
            self.advance()
            return self.scope()

        else:
            index = self.i
            statement = self.make_name_define_statement()

            if statement is None:
                self.i = index - 1  # rollback to before the type
                self.advance()
                return self.expression()
            else:
                return statement

    def expression(self) -> Optional[ExpressionNode]:
        """
        Parses the next expression until it finds its end and returns an AST of the expression

        The end is defined by a separator token such as ',' ';' ':', a keyword, or by finding 2 tokens not connected
        with an operator

        expression : value
                   | unary_operator expression
                   | expression binary_operator expression
                   | function_call

        :param: end_separator: Optional[str] char separator that can be expected to end the expression
        :return: ExpressionNode
        """

        def is_last_tok_value(last_operand_operator) -> bool:
            """
            Detects if the last non separator token was a variable/literal, within a certain range

            :param last_operand_operator: the last Node/Token that has relevance to the expression
            :return: True if the last non separator token was a variable/literal, False otherwise
            """
            if last_operand_operator is None:
                return False

            elif isinstance(last_operand_operator, Token) and last_operand_operator.tt == TT.OPERATOR:
                return False

            else:
                return True

        def must_pop(op_stack, tok: Token) -> bool:
            if len(op_stack) == 0:
                return False
            p1 = OPERATOR_PRECENDENCE[tok.value]
            p2 = OPERATOR_PRECENDENCE[op_stack[-1].value]
            return p1 < p2 or (p1 == p2 and tok.value not in RIGHT_ASSOCIATIVE_OPERATORS)


        def shunting_yard() -> Optional[list[(Token, Node)]]:
            """
            Parses an infix expression into a postfix expression using the shunting yard algorithm.

             The expression is the stream of tokens until an end is found. End is defined as a keyword, repetition of
             operands or separator symbols.

            :return: list of operands/operators or None if an error was found
            """
            last_op = None
            expression_queue: list[(Token, Node)] = []
            operator_stack: list[Token] = []
            while self.has_next():
                if self.peak.tt == TT.KEYWORD or \
                        (self.peak.tt in (TT.IDENTIFIER, TT.LITERAL) and is_last_tok_value(last_op)):
                    break

                if self.peak.tt in (TT.LITERAL, TT.IDENTIFIER):
                    expression_queue.append(self.peak)
                    last_op = self.peak

                elif self.peak.tt == TT.SEPARATOR:
                    if self.peak.value == '(':
                        if self.last.tt == TT.IDENTIFIER:  # it's a function call
                            self.advance()
                            args = self.repeat_until_symbol(')', Parser.expression, SyntaxError.expression_expected)
                            func = Operators.func.value

                            while must_pop(operator_stack, func):
                                expression_queue.append(operator_stack.pop())

                            operator_stack.append(func)
                            expression_queue.append(args)  # function arguments will be lists
                            last_op = args
                        else:
                            operator_stack.append(self.peak)

                    elif self.peak.value == ')':
                        while len(operator_stack) > 0 and operator_stack[-1].value != '(':
                            expression_queue.append(operator_stack.pop())
                        if len(operator_stack) > 0:
                            operator_stack.pop()
                        else:
                            break

                    elif self.peak.value == '[':
                        if is_last_tok_value(last_op):
                            index = self.make_index(operator_stack)
                            if index is not None:
                                expression_queue.append(index)
                                last_op = None

                        else:
                            arr = self.array_literal()
                            expression_queue.append(arr)  # arrays/dict/set and other data structures will be ValueNodes
                            last_op = arr

                    elif self.peak.value == ']':
                        while len(operator_stack) > 0 and operator_stack[-1].value != '[]':
                            expression_queue.append(operator_stack.pop())

                        if len(operator_stack) > 0:
                            expression_queue.append(operator_stack.pop())
                        else:
                            break

                    elif self.peak.value == '{':
                        if is_last_tok_value(last_op):
                            break
                        else:
                            dict_set = self.dict_set_literal()  # arrays/index and other data structures will be ValueNodes
                            expression_queue.append(dict_set)
                            last_op = dict_set

                    else:
                        break

                elif self.peak.tt == TT.OPERATOR:
                    if self.peak == Operators.sub.value and not is_last_tok_value(last_op): # its unary minus
                        self.peak.value = '-'   # changing its value

                    while must_pop(operator_stack, self.peak):
                        expression_queue.append(operator_stack.pop())
                    operator_stack.append(self.peak)
                    last_op = self.peak

                else:
                    assert False

                self.advance()

            while len(operator_stack) > 0:
                operator = operator_stack.pop()
                if operator == Separators.lpa.value:
                    self.error(SyntaxError.symbol_expected, self.peak, Separators.rpa.value.value)
                    return None
                else:
                    expression_queue.append(operator)

            return expression_queue

        def make_ast() -> Optional[ExpressionNode]:
            """
            Parses a postfix expression (generated by ``shunting_yard``) and creates an ast of the respective nodes.

            :return: ExpressionNode or None if an error occurred
            """

            operand_stack: list[ExpressionNode] = []
            for tok in postfix_expression:
                if isinstance(tok, ExpressionNode) or isinstance(tok, list):  # Array/Dict/Det or other data structure
                    # noinspection PyTypeChecker
                    operand_stack.append(tok)

                elif tok.tt == TT.LITERAL:
                    operand_stack.append(ValueNode(tok))

                elif tok.tt == TT.IDENTIFIER:
                    operand_stack.append(VariableNode(tok))

                elif tok.tt == TT.OPERATOR:
                    if tok.value in UNARY_OPERATORS:
                        if len(operand_stack) >= 1:
                            node = operand_stack.pop()
                            if isinstance(node, ValueNode) and isinstance(node.repr_token.value, int):  # optimisation
                                node.repr_token.value = -node.repr_token.value
                                operand_stack.append(node)
                            else:
                                operand_stack.append(UnOpNode(tok, node))
                        else:
                            self.error(SyntaxError.expression_expected, tok)
                    else:
                        if len(operand_stack) >= 2:
                            node2 = operand_stack.pop()
                            node1 = operand_stack.pop()
                            if tok.value in ASSIGN_OPERATORS:
                                var_assign = self.var_assign_statement(node1, node2, tok)
                                if var_assign is not None:
                                    operand_stack.append(var_assign)

                            elif tok == Operators.index.value:
                                operand_stack.append(IndexOperatorNode(tok, node1, node2))

                            elif tok == Operators.dot.value:
                                # noinspection PyTypeChecker
                                operand_stack.append(DotOperatorNode(tok, node1, node2))

                            elif tok == Operators.func.value:
                                if not isinstance(node2, list):
                                    assert False

                                if isinstance(node1, VariableNode):
                                    operand_stack.append(FuncCallNode(node1.repr_token, node1, tuple(node2)))
                                else:
                                    self.error(SyntaxError.not_a_func, node1.repr_token, node1)

                            else:
                                operand_stack.append(BinOpNode(tok, node1, node2))
                        else:
                            self.error(SyntaxError.expression_expected, tok)
                else:
                    assert False

            if len(operand_stack) != 1:
                return None  # no expression found
            else:
                return operand_stack[0]

        if self.peak == Separators.rpa.value:
            self.error(SyntaxError.symbol_expected_before, self.peak, '(', ')')
            return None
        elif self.peak == Separators.rbr.value:
            self.error(SyntaxError.symbol_expected_before, self.peak, '[', ']')
            return None
        elif self.peak == Separators.rcb.value:
            self.error(SyntaxError.symbol_expected_before, self.peak, '{', '}')
            return None

        postfix_expression = shunting_yard()
        if postfix_expression is None:
            return None

        ast = make_ast()
        if ast is None:
            return None

        return ast

    def make_index(self, operator_stack: list[Token]) -> Optional[ExpressionNode]:
        """
        Parses the next expression until the end of indexing was found.

        If no expression is found, or more than 1 expression is found, then an error is detected.

        :param operator_stack: list from the calling function (``self.expression()``)
        :return: ExpressionNode or None if an error occurred
        """

        self.peak.tt = TT.OPERATOR  # changing its status to operator
        self.peak.value = '[]'
        operator_stack.append(self.peak)
        self.advance()

        index = self.repeat_until_symbol(Separators.rbr.value.value, Parser.expression, SyntaxError.expression_expected)

        if len(index) == 0:
            return None

        if len(index) > 1:
            self.error(SyntaxError.cannot_index_multiple, index[1].repr_token)  # error but accept the first expression

        return index[0]

    def while_statement(self) -> Optional[WhileNode]:
        """
        Parses the next while statement until it finds its end

        :return: WhileNode or None if an error was found
        """
        start = self.peak
        self.advance()

        condition = self.expression()
        if condition is None:
            self.error(SyntaxError.expression_expected, self.peak)

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
            self.error(SyntaxError.expression_expected, self.peak)

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

    def return_statement(self) -> ReturnNode:
        """
        Parses the return statement until it finds its end.

        return_statement : 'return' [expression]

        :return: ReturnNode
        """
        start = self.peak
        self.advance()

        return ReturnNode(start)

    def break_statement(self) -> BreakNode:
        """
        Parses the break statement until it finds its end.

        break_statement : 'break' [literal]

        :return: BreakNode
        """

        start = self.peak
        self.advance()
        value = None

        if self.peak.tt == TT.LITERAL and isinstance(self.peak.value, int):
            value = self.peak
            self.advance()

        return BreakNode(start, value)

    def skip_statement(self) -> SkipNode:
        """
        Parses the break statement until it finds its end.

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

    def make_type(self) -> Optional[Type]:
        """
        Parses the next type and returns it.

        Generics specified between [] will be added as part of the type.

        type : 'null'
            | identifier
            | '(' (type)* ')'
            | '[' type ']'
            | type -> type

        :return: Type or None if an error occurred
        """

        t = None
        if self.peak == Types.type.value.name:  # it's a typedef
            t = Type(self.peak)
            self.advance()

        elif self.peak.tt == TT.LITERAL and self.peak.value is None:  # dealing with the null special case
            t = Type(self.peak)
            self.advance()

        elif self.peak.tt == TT.IDENTIFIER:     # it's a type
            name = self.peak
            if self.preview() == Separators.lbr.value:
                self.advance(2)
                generics = []
                while self.peak != Separators.rpa.value:
                    t = self.make_type()
                    if t is None:
                        return None
                    generics.append(t)
                # t = Type(name, generics)  # TODO: remove commend when generics are added
                t = Type(name)
            else:
                t = Type(name)
            self.advance()

        elif self.peak == Separators.lpa.value:     # tuple
            self.advance()
            types: list[Type] = []
            while self.peak != Separators.rpa.value:
                t = self.make_type()
                if t is None:
                    return None
                types.append(t)
            self.advance()

            t = TupleType(types)

        elif self.peak == Separators.lbr.value:     # array also has alternative form
            self.advance()
            t = self.make_type()
            if t is None or self.peak != Separators.rbr.value:
                return None
            self.advance()
            t = ArrayType(t)

        else:
            return None     # no correct type notation found

        if self.peak == Operators.fn.value:     # function
            self.advance()
            ret_type = self.make_type()     # functions are right associative
            if ret_type is None:
                return None
            return FunctionType(t, ret_type)

        return t

    def make_name_define_statement(self) -> Optional[NameDefineNode]:
        """
        Parses the next statement (that does not start with a keyword) without knowing what kind of statement it is.
        The difference is spotted on how the statement goes.

        - It is a func_define if: IDENTIFIER [:] fn Type
        - It is a type_define if: IDENTIFIER [:] type Type
        - It is a macro_define if: IDENTIFIER [:] macro
        - It is a var_define if not the above: IDENTIFIER [:] Type

        :return: var_define_statement | func_define_statement | None if an expression/error occurred
        """

        name = self.peak
        if name.tt != TT.IDENTIFIER:
            return None

        self.advance()

        if self.peak == Separators.colon.value: # TODO semicolon is not needed. should I change that?
            self.advance()

        if self.peak == Keywords.fn.value:  # it's a function
            self.advance()
            return self.func_define_statement(name)

        elif self.peak == Keywords.type.value:  # it's a class
            self.advance()
            return self.type_define_statement(name)

        elif self.peak == Keywords.macro.value:  # it's a macro
            self.advance()
            return self.macro_define_statement(name)

        else:  # it's not using special syntax to define a new name, so it can be any variable type
            t = self.make_type()
            if t is None:
                return None

            return self.var_define_statement(name, t)

    def make_parameter_define(self) -> Optional[VarDefineNode]:
        name = self.peak
        if name.tt != TT.IDENTIFIER:
            return None

        self.advance()

        if self.peak == Separators.colon.value:  # TODO semicolon is not needed. should I change that?
            self.advance()

        t = self.make_type()
        if t is None:
            return None

        return self.var_define_statement(name, t)

    def var_define_statement(self, name: Token, var_type: Type) -> Optional[VarDefineNode]:
        """
        Parses the next variable declaration statement and returns its Node

        var_define_statement : 'var' type identifier
                             | 'var' type var_assign_statement

        :return: VarAssignNode or None if an error occurred
        """

        if self.peak != Operators.assign.value:
            return VarDefineNode(name, var_type, VariableNode(name))
        else:
            eq_symbol = self.peak
            self.advance()
            value = self.expression()

            if isinstance(value, ExpressionNode):
                return VarDefineNode(name, var_type, VariableNode(name), value)

            else:
                self.error(SyntaxError.symbol_expected, eq_symbol, '=')
        return None

    def var_assign_statement(self, node1, node2, tok) -> Optional[ExpressionNode]:
        """
        Parses the next variable assignment and returns its Node.
        It expands operate and assign operations such as ``+=``.

        var_assign_statement : identifier '=' expression

        :param node1: node containing the variable to be assigned the value to
        :param node2: node containing the value to be assigned to the variable
        :param tok: the assign token to be used to construct the resulting Node
        :return: AssignNode or None if an error occurred
        """
        if not isinstance(node1, VariableNode):
            self.error(SyntaxError.cannot_assign_to_non_var, node1.repr_token, node1.repr_token.value)
            return None

        if tok.value == '=':
            return AssignNode(node1, node2)
        else:
            tok.value = tok.value[:-1]
            return AssignNode(node1, BinOpNode(tok, node1, node2))

    def func_define_statement(self, name: Token) -> Optional[FuncDefineNode]:
        """
        Parses the next function declaration until its end is found, and returns its Node

        :return: FuncDefineNode or None if an error occurred
        """

        params: list[NameDefineNode] = []

        if self.peak == Separators.lpa.value:
            self.advance()
            while self.peak != Separators.rpa.value:
                param = self.make_parameter_define()
                if param is not None:
                    params.append(param)
            self.advance()

            if len(params) == 0:
                param_type = copy(Types.void.value)
            elif len(params) == 1:
                param_type = params[0].type
            else:
                param_type = TupleType([p.type for p in params])

        elif self.peak == Types.void.value.name:
            param_type = Type(self.peak) # void
            self.advance()
        else:
            self.error(SyntaxError.type_expected, self.peak)
            return None

        if self.peak == Operators.fn.value:
            self.advance()
        else:
            self.error(SyntaxError.symbol_expected, self.peak, Operators.fn.value)
            return None

        index = self.i

        if self.peak == Types.null.value.name:
            ret_param_type = Type(self.peak) # null
            self.advance()
        elif self.peak == Separators.lpa.value and self.preview() == Separators.rpa.value:  # void
            ret_param_type = copy(Types.void.value)
            self.advance(2)
        else:
            ret_param = self.make_parameter_define()

            if ret_param is None:
                self.i = index - 1  # rollback to before the ret_type
                self.advance()
                return None
            ret_param_type = ret_param.type

        func_type: FunctionType = FunctionType(param_type, ret_param_type)

        body = self.statement()
        if body is None:
            return None

        if isinstance(body, ScopeNode):
            body = IsolatedScopeNode.new_from_old(body)     # casting down scope node to func body node aka isolated

        return FuncDefineNode(VariableNode(name), func_type, tuple(params), ret_param_type, body)

    def type_define_statement(self, name: Token) -> Optional[TypeDefineNode]:
        """
        Parses the next type declaration until its end is found, and returns its Node
        :param name: name of the new type
        :return: TypeDefineNode or None if an error occurred
        """

        body = self.statement()
        if body is None:
            return None

        if isinstance(body, ScopeNode):
            nodes = body.child_nodes
        else:
            nodes = [body]

        fields = []
        for field in nodes:
            if isinstance(field, NameDefineNode):
                fields.append(field)
            else:
                self.error(SyntaxError.declaration_expected, field.repr_token)

        return TypeDefineNode(VariableNode(name), fields)

    def macro_define_statement(self, name: Token) -> Optional[MacroDefineNode]:
        """
        Parses the next macro definition until its end is found, and returns its Node
        :return: MacroDefineNode or None if an error occurred
        """

        if self.peak != Operators.assign.value:
            self.error(SyntaxError.symbol_expected, self.peak, Operators.assign.value)
            return None

        self.advance()

        value = self.expression()

        if value is None:
            self.error(SyntaxError.expression_expected, self.peak)
            return None

        return MacroDefineNode(name, VariableNode(name), value)

    def array_literal(self) -> ValueNode:
        """
        Parses and constructs an Array Literal from the next tokens and returns it.

        array : '[' {expression} ']'

        :return: ValueNode array literal
        """

        start_tok = self.peak
        self.advance()
        elements = self.repeat_until_symbol(']', Parser.expression, SyntaxError.expression_expected)

        return ValueNode(Literal(elements, copy(Types.array.value), start_tok.start, self.peak.end, start_tok.line))

    def dict_set_literal(self) -> ValueNode:
        """
        Parses and constructs either a Dict or Set Literal from the next tokens and returns it, correctly identifying
        which one being defined. Empty set will result in empty dict

        dict : '{' {expression ':' expression} '}'

        set : '{' {expression} '}'

        :return: ValueNode dict or set literal
        """
        start_tok = self.peak
        self.advance()
        is_dict = True

        rollback_index = self.i

        elements = self.repeat_until_symbol('}', Parser.expression, SyntaxError.expression_expected)

        if len(elements) > 0:   # detecting if its dict or set
            i = self.i
            self.i = rollback_index-1   # going 1 extra backwards to do advance afterwards
            self.advance()  # updates all variables needed

            error_number = len(self.errors)
            self.expression()       # this might create errors
            while len(self.errors) > error_number:
                self.errors.pop()   # this will remove duplicate errors generated
                self.expression()

            is_dict = self.peak == Separators.colon.value

            self.i = i

        if is_dict:
            dictionary = {}
            for key, value in zip(*[iter(elements)] * 2):   # did some magic here I can't explain, stackoverflow can xD
                dictionary[key] = value

            return ValueNode(Literal(dictionary, copy(Types.dict.value), start_tok.start, self.peak.end, start_tok.line))
        else:
            return ValueNode(Literal(set(elements), copy(Types.set.value), start_tok.start, self.peak.end, start_tok.line))

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
            element = method(self, *args)

            if element is None:
                if error is not None:
                    self.error(error, self.peak)
            else:
                output.append(element)

            while self.has_next() and self.peak.value in END_OF_EXPRESSION:
                self.advance()

        if not self.has_next():
            self.error(SyntaxError.symbol_expected, self.last, end_symbol)

        return output

    def error(self, error: SyntaxError, tok: Token, *args) -> None:
        """
        Creates a new Error and adds it to the error collection of tokens ``self.errors``

        :param error: the error found
        :param tok: the token where it occurred
        :param args: extra arguments for custom error message formatting
        """
        self.errors.append(Error(error, tok.start, tok.end, tok.line, global_vars.PROGRAM_LINES[tok.line - 1], *args))

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
