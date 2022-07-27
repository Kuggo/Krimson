import ast
from enum import Enum
from sys import argv, stdout, stderr
import os

from typing import List, Dict


def main():
    src_name = argv[1] if len(argv) >= 2 else None
    dest_name = argv[2] if len(argv) >= 3 else None

    if src_name == '--help':
        print('usage: krimson <source_file> <destination_file>')

    source = '''fixed<8> var = 1'''

    if src_name is not None:
        if os.path.isfile(src_name):
            with open(src_name, mode='r') as sf:
                source = sf.read().replace("\r", "")
        else:
            print(f'"{src_name}" is not a file', file=stderr)
            exit(1)
    else:
        src_name = 'terminal'
    if dest_name is not None:
        dest = open(dest_name, mode="w")
    else:
        dest = stdout

    lexer = Lexer(source, src_name)
    lexer.tokenize()
    print(lexer.tokens, file=dest)

    if len(lexer.errors) > 0:
        for err in lexer.errors:
            print(err, file=stderr)
        exit(1)

    if len(lexer.warnings) > 0:
        for warn in lexer.warnings:
            print(warn, file=stderr)

    parser = Parser(source, lexer.tokens, src_name)
    try:
        parser.parse()
        # print(parser.output, file=dest)   # im disabling this to see type checker output better
    except ErrorException as e:
        print(e.error, file=stderr)
        exit(1)

    make_default_types_helper()  # if no errors so far lets load default types to began type checking

    type_checker = TypeChecker(parser.output, source, src_name)
    type_checker.check()

    if len(type_checker.errors) > 0:
        for err in type_checker.errors:
            print(err, file=stderr)
        exit(1)

    print(type_checker.asts, file=dest)

    # TODO compile to urcl
    return


def make_default_types_helper():
    file_name = 'default_lib.txt'
    try:
        with open(file_name, 'r') as file:
            try:
                code = file.read()
                lexer = Lexer(code, file_name)
                lexer.tokenize()

                parser = Parser(code, lexer.tokens, file_name)
                parser.parse()
                for t in parser.output:
                    if isinstance(t, ObjectDeclarationNode):
                        default_types[t.name.value[2:-2]] = t
            except ErrorException as e:
                print(e.error, file=stderr)
                exit(1)
    except OSError:
        print('Unable to find built in type library', file=stderr)
        exit(1)


class TT(Enum):
    eof = 'eof'

    word = 'word'
    macro = 'macro'

    # pairs
    lpa = '('
    rpa = ')'
    lbr = '['
    rbr = ']'
    lcbr = '{'
    rcbr = '}'

    # comparison
    dif = '!='
    equ = '=='
    gt = '>'
    lt = '<'
    gte = '>='
    lte = '<='

    # logic
    and_ = '&&'
    or_ = '||'
    not_ = '!'

    # arithmetic
    add = '+'
    sub = '- '
    neg = '-'
    mlt = '*'
    div = '/'
    mod = '%'
    inc = '++'
    dec = '--'

    # bitwise
    shl = '<<'
    shr = '>>'
    b_and = '&'
    b_or = '|'
    b_xor = '^'
    b_not = '~'

    # variable assignment
    assign = '='

    assign_add = '+='
    assign_sub = '-='
    assign_mlt = '*='
    assign_div = '/='
    assign_mod = '%='

    assign_shl = '<<='
    assign_shr = '>>='
    assign_b_and = '&='
    assign_b_or = '|='
    assign_b_xor = '^='
    assign_b_not = '~='

    # primitive types
    bool_ = 'bool'
    int_ = 'int'
    uint = 'uint'
    fixed = 'fixed'
    float_ = 'float'
    char = 'char'

    func = 'func'
    func_call = 'func()'
    address = '[]'

    # primitive data structures
    string = 'string'
    array = 'array'
    object_ = 'object'

    # Default values
    true = 'true'
    false = 'false'
    null = 'null'

    # keywords
    if_ = 'if'
    elif_ = 'elif'
    else_ = 'else'
    switch = 'switch'
    case = 'case'
    default = 'default'
    exit_ = 'exit'
    skip = 'skip'
    for_ = 'for'
    foreach = 'foreach'
    while_ = 'while'
    do_ = 'do'
    return_ = 'return'
    goto = 'goto'

    urcl = 'urcl'
    label = 'label'

    # comment
    comment = '//'

    # separators
    comma = ','
    colon = ':'
    nln = r'\n'
    semi_col = ';'
    dot = '.'

    def __repr__(self) -> str:
        return self.value


keywords = {
    'bool': TT.bool_,
    'int': TT.int_,
    'uint': TT.uint,
    'fixed': TT.fixed,
    'float': TT.float_,
    'char': TT.char,

    'func': TT.func,

    'string': TT.string,
    'array': TT.array,
    'object': TT.object_,
    'true': TT.true,
    'false': TT.false,
    'null': TT.null,

    'if': TT.if_,
    'elif': TT.elif_,
    'else': TT.else_,
    'switch': TT.switch,
    'case': TT.case,
    'default': TT.default,
    'exit': TT.exit_,
    'skip': TT.skip,
    'for': TT.for_,
    'foreach': TT.foreach,  # not yet
    'while': TT.while_,
    'do': TT.do_,
    'return': TT.return_,
    'goto': TT.goto,

    'urcl': TT.urcl,
    'macro': TT.macro,
}

symbols = {
    '(': TT.lpa,
    ')': TT.rpa,
    '{': TT.lcbr,
    '}': TT.rcbr,
    '[': TT.lbr,
    ']': TT.rbr,
    '!': TT.not_,
    '=': TT.assign,
    '<': TT.lt,
    '>': TT.gt,
    '&': TT.b_and,
    '|': TT.b_or,
    '+': TT.add,
    '-': TT.sub,
    '*': TT.mlt,
    '/': TT.div,
    '%': TT.mod,
    '~': TT.b_not,
    ',': TT.comma,
    ';': TT.semi_col,
    ':': TT.colon,
    '.': TT.dot,
}

op_table = {
    # left to right
    TT.inc: (1, True),
    TT.dec: (1, True),
    TT.func_call: (20, True),
    TT.dot: (1, True),
    # right to left
    TT.not_: (2, False),
    TT.b_not: (2, False),
    TT.neg: (2, False),
    # unary plus and minus go here on 2
    # type cast should be here too but ill make it the same as function call above

    # left to right
    TT.mlt: (3, True),
    TT.div: (3, True),
    TT.mod: (3, True),
    TT.add: (4, True),
    TT.sub: (4, True),
    TT.shr: (5, True),
    TT.shl: (5, True),
    TT.gt: (6, True),
    TT.gte: (6, True),
    TT.lt: (6, True),
    TT.lte: (6, True),
    TT.dif: (7, True),
    TT.equ: (7, True),
    TT.b_and: (8, True),
    TT.b_xor: (9, True),
    TT.b_or: (10, True),
    TT.and_: (11, True),
    TT.or_: (12, True),
    # right to left
    TT.assign: (14, False),
    TT.assign_shl: (14, False),
    TT.assign_shr: (14, False),
    TT.assign_add: (14, False),
    TT.assign_sub: (14, False),
    TT.assign_div: (14, False),
    TT.assign_mlt: (14, False),
    TT.assign_b_and: (14, False),
    TT.assign_b_or: (14, False),
    TT.assign_b_not: (14, False),
    TT.assign_mod: (14, False),
    TT.lpa: (20, True),
    TT.address: (19, True),
    # TT.rbr: (19, True),   # not needed makes only trouble
}

assignment_toks = {TT.assign, TT.assign_shl, TT.assign_shr, TT.assign_add, TT.assign_sub, TT.assign_div, TT.assign_mlt,
                   TT.assign_b_and, TT.assign_b_or, TT.assign_b_not, TT.assign_mod}

unary_ops = {TT.inc, TT.dec, TT.not_, TT.b_not, TT.neg, TT.func_call}

default_ops = {
    TT.inc: '__inc__',
    TT.dec: '__dec__',
    TT.not_: '__not__',
    TT.b_not: '__bnot__',
    TT.neg: '__neg__',
    TT.mlt: '__mlt__',
    TT.div: '__div__',
    TT.mod: '__mod__',
    TT.add: '__add__',
    TT.sub: '__sub__',
    TT.shr: '__shr__',
    TT.shl: '__shl__',
    TT.gt: '__gt__',
    TT.gte: '__gte__',
    TT.lt: '__lt__',
    TT.lte: '__lte__',
    TT.dif: '__dif__',
    TT.equ: '__equ__',
    TT.b_and: '__band__',
    TT.b_xor: '__bxor__',
    TT.b_or: '__bor__',
    TT.and_: '__and__',
    TT.or_: '__or__',
    TT.address: '__get__',
}


class Token:
    def __init__(self, tt: TT, start_char, end_char, line, value=None, generics=None):
        self.type = tt
        self.start = start_char
        self.end = end_char
        self.line = line
        if value is None:
            self.value = ''
        else:
            self.value = value

        if generics is None:
            self.generics = []
        else:
            self.generics = generics
        return

    def __repr__(self):
        return f"<{self.line}:{self.start}:{self.end}:{self.type} {self.value}>"


class E(Enum):
    name_expected = 'Name expected'
    duplicate_object = 'Duplicate object name'
    duplicate_func = 'Duplicate function declaration'
    duplicate_var = 'Duplicate variable name'
    duplicate_macro = 'Duplicate macro'
    literal_expected = 'Literal expected'
    symbol_expected = '{} expected'
    expression_expected = 'Expression expected'
    statement_expected = 'Statement expected'
    string_expected = 'String expected'
    invalid_literal = 'Invalid literal'
    invalid_char = 'Invalid character'
    invalid_ret_type = 'Invalid return type'
    miss_close_sym = 'Missing single quote {}'
    if_expected = 'if keyword expected before else/elif'
    while_expected = 'while keyword expected'
    identifier_expected = 'Identifier expected'
    exit_not_in_loop = 'exit keyword found outside a loop/switch'
    skip_not_in_loop = 'skip keyword found outside a loop'
    foreach_not_applicable = 'foreach not applicable to variable'
    unknown_var_type = 'Unknown variable type'
    undefined_variable = 'Undefined variable'
    var_before_assign = 'Variable might been used before assignment'
    undefined_function = 'Undefined function'
    unknown_obj_type = 'Unknown object type'
    wrong_arg_type = 'Wrong function argument type'
    no_attribute = '{} has no attribute {}'

    def __repr__(self) -> str:
        return self.value


class Error:
    def __init__(self, error: E, start: int, end: int, line: int, file_name: str = None, code_line: str = None, *args):
        self.error = error
        self.start = start
        self.end = end
        self.line = line
        self.file_name = file_name
        self.code_line = code_line

        self.args = args
        return

    def __repr__(self):
        string = f'{self.file_name}:{self.start}:{self.line}: {self.error.value.format(*self.args)}\n'
        string += self.code_line + '\n'
        string += ' ' * (self.start - 1)
        string += '^' * (self.end - self.start + 1)
        return string


#########################################
# Exceptions
#########################################

class ErrorException(Exception):
    def __init__(self, error: Error):
        self.error = error
        return


#########################################
# Lexer
#########################################


class Lexer:
    def __init__(self, program: str, file_name: str):
        self.pr = program + '      \n'  # avoids checking self.has_next() tons of times
        self.lines = program.split('\n')
        self.file_name = file_name
        self.len = len(self.pr)
        self.i = 0
        self.n = 1
        self.end = 1
        self.tokens = []
        self.errors = []
        self.warnings = []

        self.peak = self.pr[self.i]
        self.last_tok = None
        self.start = self.end
        return

    def tokenize(self) -> None:
        while self.has_next():
            while self.has_next() and self.peak.isspace():
                self.advance()
            if not self.has_next():
                break
            self.reset_tok_pos()
            if self.peak == '/':
                self.advance()
                if self.has_next() and self.peak == '/':  # double slash means inline comment
                    self.advance()
                    self.inline_comment()
                    continue
                elif self.has_next() and self.peak == '*':
                    self.multi_line_comment()
                    continue
                elif self.has_next() and self.peak == '=':
                    self.token(TT.assign_div)
                    self.advance()
                else:
                    self.token(TT.div)

            self.make_token()
        return

    def make_token(self) -> None:
        if self.peak in symbols:
            self.make_symbol()

        elif self.peak.isnumeric():
            self.make_num()

        elif self.peak.isalpha() or self.peak == '_':
            self.make_word()

        elif self.peak == '"':
            self.make_string()

        elif self.peak == "'":
            self.make_char()

        else:
            self.error(E.invalid_char)
            self.advance()

        return

    def make_symbol(self) -> None:
        if self.peak == '<':
            self.advance()
            if self.peak == '<':
                self.advance()
                if self.peak == '=':
                    self.advance()
                    self.token(TT.assign_shl)
                else:
                    self.advance()
                    self.token(TT.shl)
            elif self.peak == '=':
                self.advance()
                self.token(TT.lte)
            else:
                self.token(TT.lt)

        elif self.peak == '>':
            self.advance()
            if self.peak == '>':
                self.advance()
                if self.peak == '=':
                    self.advance()
                    self.token(TT.assign_shl)
                else:
                    self.token(TT.shl)
            elif self.peak == '=':
                self.advance()
                self.token(TT.gte)
            else:
                self.token(TT.gt)

        elif self.peak == '=':
            self.advance()
            if self.peak == '=':
                self.advance()
                self.token(TT.equ)
            else:
                self.token(TT.assign)

        elif self.peak == '!':
            self.advance()
            if self.peak == '=':
                self.advance()
                self.token(TT.dif)
            else:
                self.token(TT.not_)

        elif self.peak == '&':
            self.advance()
            if self.peak == '&':
                self.advance()
                self.token(TT.and_)
            elif self.peak == '=':
                self.advance()
                self.token(TT.assign_b_and)
            else:
                self.token(TT.b_and)

        elif self.peak == '|':
            self.advance()
            if self.peak == '|':
                self.advance()
                self.token(TT.or_)
            elif self.peak == '=':
                self.advance()
                self.token(TT.assign_b_or)
            else:
                self.token(TT.b_or)

        elif self.peak == '^':
            self.advance()
            if self.peak == '=':
                self.advance()
                self.token(TT.assign_b_xor)
            else:
                self.token(TT.b_xor)

        elif self.peak == '+':
            self.advance()
            if self.peak == '+':
                self.advance()
                self.token(TT.inc)
            elif self.peak == '=':
                self.advance()
                self.token(TT.assign_add)
            else:
                if self.last_tok is None or self.last_tok.type in op_table:
                    # its unary plus, thus can be ignored
                    pass
                else:
                    self.token(TT.add)

        elif self.peak == '-':
            self.advance()
            if self.peak == '-':
                self.advance()
                self.token(TT.dec)
            elif self.peak == '=':
                self.advance()
                self.token(TT.assign_sub)
            else:
                if self.last_tok is None or self.last_tok.type in op_table:
                    # its unary minus
                    self.token(TT.neg)
                else:
                    self.token(TT.sub)

        elif self.peak == '*':
            self.advance()
            if self.peak == '=':
                self.advance()
                self.token(TT.assign_mlt)
            else:
                self.token(TT.mlt)

        elif self.peak == '/':
            self.advance()
            if self.peak == '=':
                self.advance()
                self.token(TT.assign_div)
            else:
                self.token(TT.div)

        elif self.peak == '%':
            self.advance()
            if self.peak == '=':
                self.advance()
                self.token(TT.assign_mod)
            else:
                self.token(TT.mod)

        else:
            symbol = self.peak
            self.advance()
            self.token(symbols[symbol])
        return

    def make_num(self) -> None:
        if not self.peak.isnumeric():
            self.error(E.literal_expected)
            return
        num = ''
        dot_count = 0
        while self.has_next() and (self.peak.isalnum() or self.peak == '.'):
            if self.peak == '.':
                if dot_count > 0:
                    self.token(TT.float_, float(num))
                    self.advance()
                    self.token(TT.dot)
                    self.make_word()
                    return
                dot_count += 1
            num += self.peak
            self.advance()

        if dot_count > 0:
            try:
                self.token(TT.float_, float(num))
            except ValueError:
                self.error(E.invalid_literal)
        else:
            try:
                self.token(TT.int_, int(num, 0))
            except ValueError:
                self.error(E.invalid_literal)
        return

    def make_word(self) -> None:
        if not (self.peak.isalpha() or self.peak == '_'):
            self.error(E.name_expected)
            return
        word = ''
        while self.has_next() and (self.peak.isalnum() or self.peak == '_'):
            word += self.peak
            self.advance()
            if self.peak == '.':
                if word in keywords:
                    self.token(keywords[word], keywords[word].value)
                else:
                    self.token(TT.word, word)
                self.token(TT.dot)
                self.advance()
                self.make_word()
                return

        if word in keywords:
            self.token(keywords[word], keywords[word].value)
        else:
            self.token(TT.word, word)
        return

    def make_string(self) -> None:
        string = self.peak
        self.advance()
        escape = False
        escape_char = {  # putting more later
            'n': '\n',
        }
        while self.has_next() and ((self.peak != '"' and self.peak != '\n') or escape):
            if escape:
                string += escape_char.get(self.peak, self.peak)
                escape = False
            else:
                if self.peak == '\\':
                    escape = True
                else:
                    string += self.peak
            self.advance()

        if self.peak == '\n':
            self.error(E.miss_close_sym, '"')
        string += '"'
        self.token(TT.string, string)
        self.advance()
        return

    def make_char(self) -> None:
        char = self.peak
        self.advance()
        while self.has_next() and self.peak != "'" and self.peak != '\n':
            char += self.peak
            self.advance()

        if self.peak == '\n':
            self.error(E.miss_close_sym, "'")
        char += "'"
        char = ast.literal_eval(char)
        if len(char) != 1:
            self.error(E.invalid_literal)
        else:
            self.token(TT.char, char)
        self.advance()
        return

    def inline_comment(self) -> None:
        comment = ''
        while self.has_next() and self.peak != '\n':
            comment += self.peak
            self.advance()
        self.token(TT.comment, comment)
        return

    def multi_line_comment(self) -> None:
        while self.has_next(1):
            if self.peak == '*':
                self.advance()
                if self.peak == '/':
                    self.advance()
                    return
            self.advance()
        return

    def token(self, tt: TT, value=None) -> Token:
        tok = Token(tt, self.start, self.end - 1, self.n, value)
        self.tokens.append(tok)
        self.reset_tok_pos()
        self.last_tok = tok
        return tok

    def error(self, error: E, *args) -> None:
        self.errors.append(Error(error, self.start, self.end, self.n, self.file_name, self.lines[self.n - 1], args))
        self.reset_tok_pos()
        return

    def warning(self, error: E, *args) -> None:
        self.errors.append(Error(error, self.start, self.end, self.n, self.file_name, self.lines[self.n - 1], args))
        self.reset_tok_pos()
        return

    def reset_tok_pos(self) -> None:
        self.start = self.end
        return

    def advance(self, i=1) -> None:
        if self.peak == '\n':
            self.end = 0
            self.n += 1
        self.i += i
        self.end += i
        if self.has_next():
            self.peak = self.pr[self.i]

    def has_next(self, i=0) -> bool:
        return self.i + i < self.len


#########################################
# AST
#########################################


class Node:
    def __init__(self, start, end, line, t=TT.null.value):
        self.parent = None
        self.type: str = t
        self.start = start
        self.end = end
        self.line = line
        return

    def assign_parent(self, parent) -> None:
        self.parent = parent

    def is_valid_expression(self, variables, defined_vars, funcs, types) -> bool:
        pass

    def is_valid_statement(self, variables, defined_vars, funcs, types) -> bool:
        pass


class ValueNode(Node):
    def __init__(self, value: Token):
        super().__init__(value.start, value.end, value.line, t=value.value)
        self.value = value

    def is_valid_expression(self, variables, defined_vars, funcs, types) -> bool:
        if self.value.type == TT.word:
            if self.value.value not in variables:
                raise ErrorException(Error(E.undefined_variable, self.value.start, self.value.end, self.value.line))
            if isinstance(variables[self.value.value], MacroDefNode):
                return variables[self.value.value].is_valid_statement(variables, defined_vars, funcs, types)
            if self.value.value not in defined_vars:
                raise ErrorException(Error(E.var_before_assign, self.value.start, self.value.end, self.value.line))
        return True

    def is_valid_statement(self, variables, defined_vars, funcs, types) -> bool:
        return False

    def __repr__(self):
        return f'{self.value}'


class UnOpNode(Node):
    def __init__(self, op: Token, child: Node):
        super().__init__(None, None, op.line)
        if op.type in op_table and op_table[op.type][1]:
            self.start = child.start
            self.end = op.end
        else:
            self.start = op.start
            self.end = child.end
        self.op = op
        self.child = child
        self.child.assign_parent(self.parent)
        return

    def is_valid_expression(self, variables, defined_vars, funcs, types) -> bool:
        if isinstance(self.child, ValueNode) and self.child.value.type == TT.word and \
                self.child.value.value in variables and isinstance(variables[self.child.value.value], MacroDefNode):
            self.child = variables[self.child.value.value].expression

        return self.child.is_valid_expression(variables, defined_vars, funcs, types)

    def replace_with_dunder_func(self, types):  # TODO
        if self.child.type in types:
            pass
        return

    def is_valid_statement(self, variables, defined_vars, funcs, types) -> bool:
        if self.op.type == TT.inc or self.op.type == TT.dec:
            return self.child.is_valid_expression(variables, defined_vars, funcs, types)
        else:
            return False

    def __repr__(self):
        if self.op.type in op_table and op_table[self.op.type][1]:
            return f'<{self.child} {self.op.type}>'
        else:
            return f'<{self.op.type} {self.child}>'


class BinOpNode(Node):
    def __init__(self, op: Token, left: Node, right: Node):
        super().__init__(left.start, right.end, op.line)
        self.left_child = left
        self.right_child = right
        self.op = op

        self.left_child.assign_parent(self.parent)
        self.right_child.assign_parent(self.parent)
        return

    def is_valid_expression(self, variables, defined_vars, funcs, types) -> bool:
        if isinstance(self.left_child, ValueNode) and self.left_child.value.type == TT.word and \
                self.left_child.value.value in variables and \
                isinstance(variables[self.left_child.value.value], MacroDefNode):

            self.left_child = variables[self.left_child.value.value].expression

        if isinstance(self.right_child, ValueNode) and self.right_child.value.type == TT.word and \
                self.right_child.value.value in variables and \
                isinstance(variables[self.right_child.value.value], MacroDefNode):

            self.right_child = variables[self.right_child.value.value].expression

        return self.left_child.is_valid_expression(variables, defined_vars, funcs, types) and \
               self.right_child.is_valid_expression(variables, defined_vars, funcs, types)

    def is_valid_statement(self, variables, defined_vars, funcs, types) -> bool:
        if self.op.type in assignment_toks:
            return self.right_child.is_valid_expression(variables, defined_vars, funcs, types)
        else:
            return False

    def __repr__(self):
        return f'<{self.left_child} {self.op.type} {self.right_child}>'


class DotOpNode(Node):
    def __init__(self, op: Token, obj: Node, prop: Node):
        super().__init__(obj.start, prop.end, op.line)
        self.object = obj
        self.property = prop
        self.op = op

        self.object.assign_parent(self.parent)
        self.property.assign_parent(self.parent)
        return

    def is_valid_expression(self, variables, defined_vars, funcs, types) -> bool:
        if self.object.type in types and types[self.object.type].contains_attribute(self.property):
            return self.object.is_valid_expression(variables, defined_vars, funcs, types)
        else:
            raise ErrorException(Error(E.no_attribute, self.property.start, self.property.end, self.property.line,
                                       None, None, self.object, self.property))

    def is_valid_statement(self, variables, defined_vars, funcs, types) -> bool:
        return False

    def __repr__(self):
        return f'<{self.object}.{self.property}>'


class VarDeclarationNode(Node):
    def __init__(self, var_type: Token, name: Token, value: BinOpNode = None):
        super().__init__(var_type.start, name.end, name.line)
        self.var_type = var_type
        self.name = name
        self.value = value
        if value is not None:
            value.assign_parent(self.parent)
        return

    def is_valid_expression(self, variables, defined_vars, funcs, types) -> bool:
        return False

    def is_valid_statement(self, variables, defined_vars, funcs, types) -> bool:
        if self.var_type.value in types and types[self.var_type.value].check_generics(self.var_type.generics):
            return self.value.right_child.is_valid_expression(variables, defined_vars, funcs, types)
        else:
            raise ErrorException(
                Error(E.identifier_expected, self.var_type.start, self.var_type.end, self.var_type.line))

    def __repr__(self):
        if self.value is None:
            return f'<{self.var_type.value} {self.name} = {self.value}>'
        else:
            return f'<{self.var_type.value} {self.name} = {self.value.right_child}>'


class ScopedNode(Node):
    def __init__(self, start=None, end=None, line=None):
        super().__init__(start, end, line)
        self.body = None
        self.vars = {}
        self.funcs = {}
        self.classes = {}
        return

    def assign_body(self, body: List[Node] = None):
        if body is None:
            self.body = []
        else:
            self.body = body
            for b in body:
                b.assign_parent(self)
                if isinstance(b, VarDeclarationNode):
                    self.vars[b.name.value] = b
                elif isinstance(b, FuncDeclarationNode):
                    self.funcs[b.name.value] = b
                elif isinstance(b, ObjectDeclarationNode):
                    self.classes[b.name.value] = b
        if len(body) > 0:
            self.end = self.body[-1].end
        return

    def is_valid_expression(self, variables, defined_vars, funcs, types) -> bool:
        return False

    def is_valid_statement(self, variables, defined_vars, funcs, types) -> bool:
        return True


class FuncDeclarationNode(ScopedNode):
    def __init__(self, ret_type: Token, name: Token, args: List[VarDeclarationNode] = None):
        super().__init__(start=name.start, end=name.end, line=name.line)
        self.type = ret_type.value
        self.name = name
        if args is None:
            self.args = []
        else:
            self.args = args
            for arg in args:
                arg.assign_parent(self)
        self.body = None
        return

    def is_valid_statement(self, variables, defined_vars, funcs, types) -> bool:
        for arg in self.args:
            if arg.type not in types:
                raise ErrorException(Error(E.unknown_var_type, arg.start, arg.end, arg.line))
        return True

    def get_arg_types(self) -> List[str]:
        args = []
        for arg in self.args:
            args.append(arg.value.type)
        return args

    def assign_parent(self, parent) -> None:
        self.parent = parent

    def __repr__(self):
        args = str(self.args)[1:-1]
        string = f'<{self.type} {self.name} ({args}) ' + '{'
        for node in self.body:
            string += str(node)
        string += '}>'
        return string


class ObjectDeclarationNode(ScopedNode):
    def __init__(self, name: Token, generics: List[str], parent_classes: List[Token]):
        super().__init__(start=name.start, end=name.end, line=name.line)
        self.name = name
        self.generics = generics
        self.parent_classes = parent_classes
        return

    def is_valid_statement(self, variables, defined_vars, funcs, types) -> bool:
        for c in self.parent_classes:
            if c not in types:
                raise ErrorException(Error(E.unknown_obj_type, c.start, c.end, c.line))
        return True

    def contains_attribute(self, name: str, types) -> bool:
        output = name in self.vars or name in self.funcs or name in self.classes
        if output:
            return True
        for parent in self.parent_classes:
            if parent.value in types and types[parent.value].contains_attribute(name, types):
                return True
        return False

    def check_generics(self, generics) -> bool:
        return len(generics) == len(self.generics)

    def __repr__(self):
        parents = str(self.parent_classes)[1:-1]
        if len(parents) == 0:
            string = f'{self.name} ' + '{'
        else:
            string = f'{self.name}({parents}) ' + '{'
        for node in self.body:
            string += str(node)
        string += '}>'
        return string


class MacroDefNode(Node):
    def __init__(self, name: Token, expression: Node):
        super().__init__(name.start, expression.end, name.line)
        self.name = name.value
        self.expression = expression

    def is_valid_expression(self, variables, defined_vars, funcs, types) -> bool:
        return False

    def is_valid_statement(self, variables, defined_vars, funcs, types) -> bool:
        return self.expression.is_valid_expression(variables, defined_vars, funcs, types)

    def __repr__(self):
        return f'{self.name} = {self.expression}'


class FuncCallNode(Node):
    def __init__(self, token: Token, args: List):
        super().__init__(token.start, token.end, token.line)
        self.func = token
        self.func_name = token.value
        self.args = args
        return

    def is_valid_expression(self, variables, defined_vars, funcs, types) -> bool:
        return self.is_valid_statement(variables, defined_vars, funcs, types)

    def is_valid_statement(self, variables, defined_vars, funcs, types) -> bool:
        if self.func_name not in funcs:
            raise ErrorException(Error(E.undefined_function, self.func.start, self.func.end, self.func.line))
        func_args = funcs[self.func_name].get_arg_types()
        for i, arg in enumerate(self.args):
            if arg.type == TT.word:
                if arg.value not in variables:
                    raise ErrorException(Error(E.undefined_variable, arg.start, arg.end, arg.line))
                else:
                    arg = variables[arg.value]
            if arg.type != func_args[i]:
                raise ErrorException(Error(E.wrong_arg_type, self.args[i].start, self.args[i].end, self.args[i].line))
        return True

    def get_arg_types(self) -> List[str]:
        args = []
        for arg in self.args:
            args.append(arg.type.value)
        return args

    def __repr__(self):
        return f'{self.func_name}({str(self.args)[1:-1]})'


class IfNode(Node):
    def __init__(self, condition: Node, body: List[Node] = None):
        super().__init__(condition.start, condition.end, condition.line)
        self.condition = condition
        if body is None:
            self.body = []
        else:
            self.body = body
            for b in body:
                b.assign_parent(self.parent)
        return

    def is_valid_statement(self, variables, defined_vars, funcs, types) -> bool:
        if not self.condition.is_valid_expression(variables, defined_vars, funcs, types):
            raise ErrorException(
                Error(E.expression_expected, self.condition.start, self.condition.end, self.condition.line))
        return True

    def __repr__(self):
        string = f'<if ({self.condition}) ' + '{'
        for node in self.body:
            string += str(node) + ';'
        string += '}>'
        return string


class ElseNode(Node):
    def __init__(self, start, end, line, body: List[Node] = None):
        super().__init__(start, end, line)
        if body is None:
            self.body = []
        else:
            self.body = body
            for b in body:
                b.parent.assign_parent(self.parent)
        return

    def __repr__(self):
        string = '<else {'
        for node in self.body:
            string += str(node) + ';'
        string += '}>'
        return string


class ForNode(ScopedNode):
    def __init__(self, var: Node, condition: Node, step: Node):
        super().__init__(start=var.start, end=var.end, line=var.line)
        self.var = var
        self.var.parent = self
        self.condition = condition
        self.condition.parent = self
        self.step = step
        self.step.parent = self
        return

    def is_valid_statement(self, variables, defined_vars, funcs, types) -> bool:
        if not self.var.is_valid_statement(variables, defined_vars, funcs, types):
            raise ErrorException(Error(E.statement_expected, self.var.start, self.var.end, self.var.line))

        if not self.condition.is_valid_expression(variables, defined_vars, funcs, types):
            raise ErrorException(
                Error(E.expression_expected, self.condition.start, self.condition.end, self.condition.line))

        if not self.step.is_valid_statement(variables, defined_vars, funcs, types):
            raise ErrorException(
                Error(E.statement_expected, self.step.start, self.step.end, self.step.line))
        return True

    def __repr__(self):
        string = f'<for ({self.var};{self.condition};{self.step}) ' + '{'
        for node in self.body:
            string += str(node) + ';'
        string += '}>'
        return string


class ForeachNode(ScopedNode):
    def __init__(self, var: Node, collection: Node):
        super().__init__(start=var.start, end=var.end, line=var.line)
        self.var = var
        self.var.parent = self
        self.collection = collection
        self.collection.parent = self
        return

    def is_valid_statement(self, variables, defined_vars, funcs, types) -> bool:
        if not self.var.is_valid_expression(variables, defined_vars, funcs, types):
            raise ErrorException(Error(E.expression_expected, self.var.start, self.var.end, self.var.line))
        if self.collection not in variables:
            raise ErrorException(
                Error(E.foreach_not_applicable, self.collection.start, self.collection.end, self.collection.line))
        return True

    def __repr__(self):
        string = f'<for ({self.var}:{self.collection}) ' + '{'
        for node in self.body:
            string += str(node) + ';'
        string += '}>'
        return string


class WhileNode(ScopedNode):
    def __init__(self, condition: Node):
        super().__init__(start=condition.start, end=condition.end, line=condition.line)
        self.condition = condition
        self.condition.parent = self
        return

    def is_valid_statement(self, variables, defined_vars, funcs, types) -> bool:
        if not self.condition.is_valid_expression(variables, defined_vars, funcs, types):
            raise ErrorException(
                Error(E.expression_expected, self.condition.start, self.condition.end, self.condition.line))
        return True

    def __repr__(self):
        string = f'<while ({self.condition}) ' + '{'
        for node in self.body:
            string += str(node) + ';'
        string += '}>'
        return string


class DoWhileNode(ScopedNode):
    def __init__(self):
        super().__init__()
        self.condition = None
        return

    def assign_cnd(self, condition: Node):
        self.condition = condition
        self.condition.parent = self
        self.start = condition.start
        self.line = condition.line
        return

    def is_valid_statement(self, variables, defined_vars, funcs, types) -> bool:
        if not self.condition.is_valid_expression(variables, defined_vars, funcs, types):
            raise ErrorException(
                Error(E.expression_expected, self.condition.start, self.condition.end, self.condition.line))
        return True

    def __repr__(self):
        string = '<do {'
        for node in self.body:
            string += str(node) + ';'
        string += '}' + f' while ({self.condition})>'
        return string


class SwitchNode(ScopedNode):
    def __init__(self, switch_val: Node):
        super().__init__(start=switch_val.start, end=switch_val.end, line=switch_val.line)
        self.switch_val = switch_val
        return

    def is_valid_statement(self, variables, defined_vars, funcs, types) -> bool:
        if not self.switch_val.is_valid_expression(variables, defined_vars, funcs, types):
            raise ErrorException(
                Error(E.expression_expected, self.switch_val.start, self.switch_val.end, self.switch_val.line))
        return True

    def __repr__(self):
        string = f'<switch ({self.switch_val}) ' + '{'
        for node in self.body:
            string += str(node) + ';'
        string += '}>'
        return string


class KeywordNode(Node):
    def __init__(self, keyword: Token):
        super().__init__(keyword.start, keyword.end, keyword.line)
        self.keyword = keyword

    def is_valid_expression(self, variables, defined_vars, funcs, types) -> bool:
        return False

    def is_valid_statement(self, variables, defined_vars, funcs, types) -> bool:
        return True

    def __repr__(self):
        return f'{self.keyword}'


class SingleExpressionNode(Node):
    def __init__(self, value, word: TT):
        super().__init__(value.start, value.end, value.line)
        self.val = value
        self.word = word

    def is_valid_expression(self, variables, defined_vars, funcs, types) -> bool:
        return False

    def is_valid_statement(self, variables, defined_vars, funcs, types) -> bool:
        if not self.val.is_valid_expression(variables, defined_vars, funcs, types):
            raise ErrorException(Error(E.expression_expected, self.val.start, self.val.end, self.val.line))
        return True

    def __repr__(self):
        return f'{self.word.value} {self.val}'


class CaseNode(SingleExpressionNode):
    def __init__(self, value):
        SingleExpressionNode.__init__(self, value, TT.case)


class ReturnNode(SingleExpressionNode):
    def __init__(self, value: Node):
        SingleExpressionNode.__init__(self, value, TT.return_)


class GotoNode(SingleExpressionNode):
    def __init__(self, label: Token):
        SingleExpressionNode.__init__(self, label, TT.goto)


class UrclNode(SingleExpressionNode):
    def __init__(self, string: Token):
        SingleExpressionNode.__init__(self, string, TT.urcl)


#########################################
# Value Objects
#########################################

true = ValueNode(Token(TT.true, None, None, None))
false = ValueNode(Token(TT.false, None, None, None))
null = ValueNode(Token(TT.null, None, None, None))

default_types: Dict[str, ObjectDeclarationNode] = {
    TT.bool_.value: ObjectDeclarationNode(Token(TT.word, None, None, None, 'bool'), [], []),
    TT.int_.value: ObjectDeclarationNode(Token(TT.word, None, None, None, 'int'), [], []),
    TT.uint.value: ObjectDeclarationNode(Token(TT.word, None, None, None, 'uint'), [], []),
    TT.fixed.value: ObjectDeclarationNode(Token(TT.word, None, None, None, 'fixed'), [], []),
    TT.float_.value: ObjectDeclarationNode(Token(TT.word, None, None, None, 'float'), [], []),
    TT.char.value: ObjectDeclarationNode(Token(TT.word, None, None, None, 'char'), [], []),
    TT.func.value: None,  # function can't have operations
    TT.string.value: ObjectDeclarationNode(Token(TT.word, None, None, None, 'string'), [], []),
    TT.array.value: ObjectDeclarationNode(Token(TT.word, None, None, None, 'array'), [], []),
    TT.object_.value: None  # object type doesn't have operations
}


#########################################
# Parser
#########################################


class Parser:
    def __init__(self, program: str, toks: List[Token], file_name: str):
        self.toks = toks
        self.len = len(self.toks)
        self.i = 0
        self.peak = self.toks[self.i]
        self.last = None
        self.lines = program.split('\n')
        self.toks.append(Token(TT.eof, len(program), len(program), len(self.lines)))
        self.file_name = file_name
        self.output: List[Node] = []

        self.scope: List[Node] = []
        return

    def parse(self):
        if self.peak.type == TT.comment:
            self.advance()
        self.output += self.parse_body(TT.eof)

    def parse_body(self, end_tt: TT) -> List[Node]:
        body: List[Node] = []
        while self.has_next() and self.peak.type != end_tt:
            body += self.next_statement()
        return body

    def next_statement(self) -> List[Node]:
        tt = self.peak.type
        if tt == TT.func:
            return [self.func_def()]

        elif tt == TT.object_:
            return [self.make_object()]

        elif tt == TT.macro:
            return [self.make_macro()]

        elif tt.value in default_types:
            return [self.assign_var()]

        elif tt == TT.word:  # can be a custom type, a function call, or a label
            word = self.peak
            self.advance()
            if self.peak.type == TT.colon:
                word.type = TT.label
                self.advance()
                return [KeywordNode(word)]

            elif self.peak.type == TT.word:
                self.advance(-1)
                return [self.assign_var()]

            else:
                self.advance(-1)
                return [self.make_expression()]

        elif tt == TT.if_:
            return self.make_if()

        elif tt == TT.elif_:
            self.error(E.if_expected, self.peak)

        elif tt == TT.else_:
            self.error(E.if_expected, self.peak)

        elif tt == TT.switch:
            return [self.make_switch()]

        elif tt == TT.case:
            return [self.make_case()]

        elif tt == TT.default:
            return [self.make_default()]

        elif tt == TT.exit_:
            return [self.make_exit()]

        elif tt == TT.skip:
            return [self.make_skip()]

        elif tt == TT.for_:
            return [self.make_for()]

        elif tt == TT.foreach:
            return [self.make_foreach()]

        elif tt == TT.while_:
            return [self.make_while()]

        elif tt == TT.do_:
            return [self.make_do_while()]

        elif tt == TT.return_:
            return [self.make_return()]

        elif tt == TT.goto:
            return [self.make_goto()]

        elif tt == TT.urcl:
            return [self.make_urcl()]

        else:
            return [self.make_expression()]

    def make_expression(self) -> Node:
        toks = self.shunting_yard()
        node = self.make_ast(toks)
        if self.peak.type in {TT.comma, TT.colon, TT.semi_col}:  # TT.rpa is not there because it should not consume it
            self.advance()
        return node

    def next_statements(self) -> List[Node]:
        if self.peak.type == TT.lcbr:
            self.advance()
            statement = self.parse_body(TT.rcbr)
            self.advance()
            return statement
        else:
            return self.next_statement()

    def shunting_yard(self) -> List:
        queue: List = []
        stack = []
        start_index = self.i
        while self.has_next() and self.peak.type not in {TT.comma, TT.colon, TT.semi_col, TT.rcbr, TT.lcbr, TT.eof}:
            t = self.peak.type
            if t == TT.lpa:
                if self.last.type == TT.word:
                    queue.append(self.make_func_call(queue.pop()))
                else:
                    stack.append(self.peak)

            elif t == TT.lbr:
                if self.last.type == TT.word:
                    self.peak.type = TT.address
                    stack.append(self.peak)
                else:  # it's an array then
                    queue.append(self.make_array())

            elif t == TT.rbr:
                while len(stack) > 0 and stack[-1].type != TT.address:
                    queue.append(stack.pop())
                if len(stack) > 0:
                    queue.append(stack.pop())
                else:
                    break

            elif t in op_table:
                while len(stack) > 0 and (op_table[t][0] > op_table[stack[-1].type][0] or
                                          (op_table[t][0] == op_table[stack[-1].type][0] and op_table[t][1])):
                    queue.append(stack.pop())
                stack.append(self.peak)

            elif t == TT.rpa:
                stack_elements = []
                while len(stack) > 0 and stack[-1].type != TT.lpa and stack[-1].type != TT.func_call:
                    stack_elements.append(stack.pop())
                if len(stack) > 0:
                    if stack[-1].type == TT.func_call:
                        func_call = FuncCallNode(stack.pop(), stack_elements)
                        queue.append(func_call)
                    else:
                        stack.pop()
                        self.advance()
                        queue += stack_elements
                        if len(stack) == 0 and self.peak.type not in op_table:
                            break
                        continue
                else:
                    break  # we found the matching close parenthesis
            else:
                if self.check_if_expression_over(start_index):
                    break
                queue.append(self.peak)
            self.advance()

        while len(stack) > 0:
            queue.append(stack.pop())
        return queue

    def check_if_expression_over(self, start_index):
        if self.peak.value == self.peak.type.value:
            return True
        i = self.i - 1
        while i >= start_index and self.toks[i].type in {TT.rpa, TT.rbr}:
            i -= 1

        if i >= start_index and self.toks[i].type not in op_table:
            return True
        else:
            return False

    def make_ast(self, queue) -> Node:
        stack: List[Node] = []
        for tok in queue:
            if isinstance(tok, Node):
                stack.append(tok)
            elif tok.type in op_table:
                if tok.type in unary_ops:
                    if len(stack) >= 1:
                        node_a = stack.pop()
                        stack.append(UnOpNode(tok, node_a))
                    else:
                        self.error(E.expression_expected, tok)
                else:
                    if len(stack) >= 2:
                        node_b = stack.pop()
                        node_a = stack.pop()
                        stack.append(BinOpNode(tok, node_a, node_b))
                    else:
                        self.error(E.expression_expected, tok)
            else:
                stack.append(ValueNode(tok))
        if len(stack) > 0:
            return stack.pop()
        else:
            self.error(E.expression_expected, self.peak)

    # processing keywords

    def assign_var(self) -> VarDeclarationNode:
        var_type = self.peak
        self.advance()
        if self.peak.type == TT.lt:
            var_type.generics = self.make_generics(fixed_point=var_type.type == TT.fixed)

        if self.peak.type != TT.word:
            self.error(E.identifier_expected, self.peak)

        name = self.peak
        expression = self.make_expression()
        if isinstance(expression, BinOpNode):
            declair_node = VarDeclarationNode(var_type, name, expression)
        else:
            declair_node = VarDeclarationNode(var_type, name)
        return declair_node

    def func_def(self) -> Node:
        self.advance()
        ret_type = self.peak
        if ret_type.type != TT.word and ret_type.value not in default_types and ret_type.type != TT.null:
            self.error(E.invalid_ret_type, self.peak)

        self.advance()
        func_name = self.peak
        if func_name.type != TT.word:
            self.error(E.identifier_expected, self.peak)

        self.advance()
        if self.peak.type != TT.lpa:
            self.error(E.symbol_expected, self.peak, '(')

        self.advance()
        args = []
        while self.peak.type != TT.rpa:
            args.append(self.assign_var())

        self.advance()
        node = FuncDeclarationNode(ret_type, func_name, args)
        self.scope.append(node)
        body = self.next_statements()
        self.scope.pop()
        node.assign_body(body)
        return node

    def make_object(self) -> Node:
        self.advance()
        if self.peak.type != TT.word:
            self.error(E.identifier_expected, self.peak)
        name = self.peak
        self.advance()
        generics = []
        if self.peak.type == TT.lt:
            generics = self.make_generics(defining=True)

        parent_classes = []
        if self.peak.type == TT.lpa:
            while self.peak.type != TT.rpa:
                if self.peak.type != TT.word:
                    self.error(E.identifier_expected, self.peak)
                parent_classes.append(self.peak)
                self.advance()
                while self.peak in {TT.comma, TT.colon, TT.semi_col}:
                    self.advance()
            self.advance()

        node = ObjectDeclarationNode(name, generics, parent_classes)
        self.scope.append(node)
        body = self.next_statements()
        self.scope.pop()
        node.assign_body(body)
        return node

    def make_if(self) -> List[Node]:
        self.advance()
        condition = self.make_expression()
        body = self.next_statements()
        if self.peak.type == TT.elif_:
            return [IfNode(condition, body), self.make_elif()]
        elif self.peak.type == TT.else_:
            return [IfNode(condition, body), self.make_else()]
        return [IfNode(condition, body)]

    def make_elif(self) -> List[Node]:
        self.advance()
        cnd = self.make_expression()
        body = self.next_statements()
        if self.peak.type == TT.elif_:
            return [ElseNode(cnd.start, cnd.end, cnd.line, [IfNode(cnd, body)]), self.make_elif()]
        elif self.peak.type == TT.else_:
            return [ElseNode(cnd.start, cnd.end, cnd.line, [IfNode(cnd, body)]), self.make_else()]
        return [ElseNode(cnd.start, cnd.end, cnd.line, [IfNode(cnd, body)])]

    def make_else(self) -> ElseNode:
        start = self.peak.start
        self.advance()
        body = self.next_statements()
        return ElseNode(start, self.peak.end, self.peak.line, body)

    def make_switch(self) -> SwitchNode:
        self.advance()
        switch_val = self.make_expression()
        node = SwitchNode(switch_val)

        self.scope.append(node)
        body = self.next_statements()
        self.scope.pop()

        node.assign_body(body)
        return node

    def make_case(self) -> CaseNode:
        self.advance()
        case = self.make_expression()
        return CaseNode(case)

    def make_default(self) -> KeywordNode:
        tok = self.peak
        self.advance()
        return KeywordNode(tok)

    def outside_loop(self, switch_count=False) -> bool:
        for node in reversed(self.scope):
            t = type(node)
            if switch_count and t in {ForNode, WhileNode, DoWhileNode, SwitchNode}:
                return False
            if t in {ForNode, WhileNode, DoWhileNode}:
                return False
            if t in {FuncDeclarationNode, ObjectDeclarationNode}:
                return True

        return True

    def make_exit(self) -> KeywordNode:
        if self.outside_loop(True):
            self.error(E.exit_not_in_loop, self.peak)

        tok = self.peak
        self.advance()
        return KeywordNode(tok)

    def make_skip(self) -> KeywordNode:
        if self.outside_loop(False):
            self.error(E.skip_not_in_loop, self.peak)

        tok = self.peak
        self.advance()
        return KeywordNode(tok)

    def make_for(self) -> ForNode:
        self.advance()
        has_parenthesis = False
        if self.peak.type == TT.lpa:  # i need to patch this.
            self.advance()
            has_parenthesis = True

        setup = self.make_expression()
        end_cnd = self.make_expression()
        step = self.make_expression()
        node = ForNode(setup, end_cnd, step)

        if has_parenthesis:
            self.advance()
        self.scope.append(node)
        body = self.next_statements()
        self.scope.pop()

        node.assign_body(body)
        return node

    def make_foreach(self) -> ForeachNode:
        self.advance()
        has_parenthesis = False
        if self.peak.type == TT.lpa:  # i need to patch this.
            self.advance()
            has_parenthesis = True

        var = self.assign_var()
        collection = self.make_expression()
        node = ForeachNode(var, collection)

        if has_parenthesis:
            self.advance()
        self.scope.append(node)
        body = self.next_statements()
        self.scope.pop()

        node.assign_body(body)
        return node

    def make_while(self) -> WhileNode:
        self.advance()
        condition = self.make_expression()
        node = WhileNode(condition)

        self.scope.append(node)
        body = self.next_statements()
        self.scope.pop()

        node.assign_body(body)
        return node

    def make_do_while(self) -> Node:
        self.advance()
        node = DoWhileNode()

        self.scope.append(node)
        body = self.next_statements()
        self.scope.pop()

        node.assign_body(body)

        if self.peak.type != TT.while_:
            self.error(E.while_expected, self.peak)

        condition = self.make_expression()
        node.assign_cnd(condition)
        return node

    def make_return(self) -> ReturnNode:
        self.advance()
        if self.peak.type in {TT.comma, TT.colon, TT.semi_col}:
            self.advance()
            return ReturnNode(null)
        else:
            expression = self.make_expression()
            return ReturnNode(expression)

    def make_goto(self) -> GotoNode:
        self.advance()
        label = self.peak
        if label.type != TT.word:
            self.error(E.identifier_expected, self.peak)
        self.advance()
        return GotoNode(label)

    def make_urcl(self) -> UrclNode:
        self.advance()
        body = self.peak
        if body != TT.string:
            self.error(E.string_expected, self.peak)
        self.advance()
        return UrclNode(body)

    def make_macro(self) -> MacroDefNode:
        self.advance()
        if self.peak.type != TT.word:
            self.error(E.identifier_expected, self.peak)

        name = self.peak
        expression = self.make_expression()
        if isinstance(expression, BinOpNode):
            node = MacroDefNode(name, expression.right_child)
            return node
        else:
            self.error(E.expression_expected, self.peak)

    # utils

    def make_array(self) -> Node:
        start_tok = self.peak
        self.advance()
        elements = []
        while self.has_next() and self.peak.type != TT.rbr:
            elements.append(self.make_expression())

        return ValueNode(Token(TT.array, start_tok.start, self.peak.end, start_tok.line, elements))

    def make_func_call(self, func_name) -> FuncCallNode:
        self.advance()
        args = []
        while self.peak.type != TT.rpa:
            args.append(self.make_expression())

        return FuncCallNode(func_name, args)

    def make_generics(self, defining=False, fixed_point=False):
        self.advance()
        generics = []
        while self.has_next() and self.peak.type != TT.gt:
            if defining:
                if self.peak.type == TT.word:
                    generics.append(self.peak)
                else:
                    self.error(E.identifier_expected, self.peak)
            else:
                if self.peak.type == TT.word or (fixed_point and self.peak.type == TT.int_):
                    generics.append(self.peak)
                else:
                    self.error(E.identifier_expected, self.peak)
            self.advance()
            if self.peak.type == TT.comma:
                self.advance()
        self.advance()
        return generics

    def error(self, error: E, tok: Token, *args) -> None:
        raise ErrorException(Error(error, tok.start, tok.end, tok.line, self.file_name, self.lines[tok.line - 1], args))

    def advance(self, i=1) -> None:
        self.i += i
        self.last = self.peak
        while self.has_next() and self.toks[self.i].type == TT.comment:
            self.i += 1
        if self.has_next():
            self.peak = self.toks[self.i]

    def has_next(self, i=0) -> bool:
        return self.i + i < self.len


#########################################
# Type Checker
#########################################

class TypeChecker:
    def __init__(self, trees: List[Node], program: str, file_name):
        self.lines = program.split('\n')
        self.file_name = file_name
        self.asts = trees
        self.funcs: Dict[(str, List[str]), FuncDeclarationNode] = {}
        self.vars: Dict[str] = {}
        self.types: Dict[str, ObjectDeclarationNode] = default_types.copy()

        self.errors = []
        return

    def check(self):
        self.check_scope(self.asts)
        return

    def check_scope(self, trees):
        funcs = self.funcs.copy()
        variables = self.vars.copy()
        types = self.types.copy()

        for tree in trees:
            if isinstance(tree, VarDeclarationNode):
                if tree.name.value in self.vars:
                    self.error(E.duplicate_var, tree)
                else:
                    self.vars[tree.name.value] = tree

            elif isinstance(tree, MacroDefNode):
                if tree.name in self.vars:
                    self.error(E.duplicate_macro, tree)
                else:
                    self.vars[tree.name] = tree

            elif isinstance(tree, ScopedNode):
                if isinstance(tree, FuncDeclarationNode):
                    self.funcs[(tree.name.value, tree.get_arg_types())] = tree

                elif isinstance(tree, ObjectDeclarationNode):
                    if tree.name.value in self.types:
                        self.error(E.duplicate_object, tree)
                    else:
                        self.types[tree.name.value] = tree

                self.vars += tree.vars
                self.funcs += tree.funcs
                self.types += tree.classes

        defined_vars = set()
        for tree in trees:
            self.visit_node(tree, defined_vars)

        self.funcs = funcs
        self.vars = variables
        self.types = types
        return

    def visit_node(self, node: Node, defined_vars):
        try:
            if not node.is_valid_statement(self.vars, defined_vars, self.funcs, self.types):
                self.error(E.statement_expected, node)
                return

            if isinstance(node, VarDeclarationNode):
                if node.var_type.value not in self.types:
                    self.error(E.unknown_var_type, node)
                if node.value is not None:
                    defined_vars.add(node.name.value)

            elif isinstance(node, ObjectDeclarationNode):
                for parent in node.parent_classes:
                    if parent not in self.types:
                        self.error(E.unknown_var_type, node)

            if isinstance(node, ScopedNode):
                self.check_scope(node.body)

        except ErrorException as e:
            e.error.file_name = self.file_name
            e.error.code_line = self.lines[e.error.line - 1]
            self.errors.append(e.error)

        return

    def error(self, e: E, node: Node):
        self.errors.append(
            Error(e, node.start, node.end, node.line, self.file_name, self.lines[node.line - 1]))


if __name__ == "__main__":
    main()
