import ast
from enum import Enum
from sys import argv, stdout, stderr
import os

from typing import List


def main():
    src_name = argv[1] if len(argv) >= 2 else None
    dest_name = argv[2] if len(argv) >= 3 else None

    if src_name == '--help':
        print('usage: krimson <source_file> <destination_file>')

    source = '''for ()'''

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
    print(lexer.tokens)

    if len(lexer.errors) > 0:
        for err in lexer.errors:
            print(err, file=stderr)
        exit(1)

    parser = Parser(source, lexer.tokens, src_name)
    try:
        parser.parse()
        print(parser.output)
    except ErrorException as e:
        print(e.error, file=stderr)
        exit(1)
    return


class TT(Enum):
    eof = 'eof'

    f_declair = 'f_declair'
    declair = 'declair'

    word = 'word'

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
    ushr = '>>>'
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
    assign_ushr = '>>>='
    assign_b_and = '&='
    assign_b_or = '|='
    assign_b_xor = '^='
    assign_b_not = '~='

    # primitive types
    bool_ = 'bool'
    int_ = 'int'
    uint = 'uint'
    fixed = 'fixed'
    ufixed = 'ufixed'
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
    # foreach = 'foreach' # not going to implement yet
    while_ = 'while'
    do_ = 'do'
    return_ = 'return'
    goto = 'goto'

    urcl = 'urcl'

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
    'ufixed': TT.ufixed,
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
    # 'foreach': TT.foreach,    # not yet
    'while': TT.while_,
    'do': TT.do_,
    'return': TT.return_,
    'goto': TT.goto,

    'urcl': TT.urcl,
}

types = {TT.bool_, TT.int_, TT.uint, TT.fixed, TT.ufixed, TT.float_, TT.char, TT.func, TT.string, TT.array, TT.object_}

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
    TT.ushr: (5, True),
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
    TT.assign_ushr: (14, False),
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
    TT.rbr: (19, True),
}

unary_ops = {TT.inc, TT.dec, TT.not_, TT.b_not, TT.neg}


class Token:
    def __init__(self, type: TT, start_index, start_char, end_char, line, value=None):
        self.type = type
        self.start = start_char
        self.end = end_char
        self.line = line
        self.start_index = start_index
        if value is None:
            self.value = ''
        else:
            self.value = value
        return

    def __repr__(self):
        return f"<{self.line}:{self.start}:{self.end}:{self.type} {self.value}>"


class E(Enum):

    name_expected = 'Name expected'
    literal_expected = 'Literal expected'
    symbol_expected = '{} expected'
    expression_expected = 'Expression expected'
    invalid_literal = 'Invalid literal'
    invalid_char = 'Invalid character'
    invalid_ret_type = 'Invalid return type'
    miss_close_sym = 'Missing single quote {}'
    if_expected = 'if keyword expected before else/elif'
    while_expected = 'while keyword expected'
    identifier_expected = 'Identifier expected'
    exit_not_in_loop = 'exit keyword found outside a loop/switch'
    skip_not_in_loop = 'skip keyword found outside a loop'

    def __repr__(self) -> str:
        return self.value


class Error:
    def __init__(self, error: E, start: int, end: int, line: int, file_name: str, code_line: str, *args):
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
# AST
#########################################


class Node:
    def __init__(self, value):
        self.value = value
        self.parent = None
        return

    def __repr__(self):
        return self.value.__repr__()


class ValueNode(Node):
    def __init__(self, value):
        super().__init__(value)


class UnOpNode(Node):
    def __init__(self, op: Token, child: Node):
        super().__init__(self)
        self.op = op
        self.child = child
        self.child.parent = self.parent
        return

    def __repr__(self):
        if self.op.type in op_table and op_table[self.op.type][1]:
            return f'<{self.child} {self.op.type}>'
        else:
            return f'<{self.op.type} {self.child}>'


class BinOpNode(Node):
    def __init__(self, op: Token, left: Node, right: Node):
        super().__init__(self)
        self.left_child = left
        self.right_child = right
        self.op = op

        self.left_child.parent = self.parent
        self.right_child.parent = self.parent
        return

    def __repr__(self):
        return f'<{self.left_child} {self.op.type} {self.right_child}>'


class VarDeclarationNode(Node):
    def __init__(self, type: Token, name: Token, value: Node = None):
        super().__init__(self)
        self.type = type
        self.name = name
        self.value = value
        if value is not None:
            value.parent = self.parent
        return

    def __repr__(self):
        if self.value is None:
            return f'<{self.type} {self.name} = {self.value}>'
        else:
            return f'<{self.type} {self.name} = {self.value.value.right_child.value}>'


class ScopedNode(Node):
    def __init__(self, value):
        super().__init__(value)
        self.body = None
        return

    def assign_body(self, body: List[Node] = None):
        if body is None:
            self.body = []
        else:
            self.body = body
            for b in body:
                b.parent = self
        return


class FuncDeclarationNode(ScopedNode):
    def __init__(self, type: Token, name: Token, args: List[VarDeclarationNode] = None):
        super().__init__(self)
        self.type = type
        self.name = name
        if args is None:
            self.args = []
        else:
            self.args = args
            for arg in args:
                arg.parent = self
        self.body = None
        return

    def __repr__(self):
        args = str(self.args).replace('[', '(')
        args = args.replace(']', ')')
        string = f'<{self.type} {self.name} {args} ' + '{'
        for node in self.body:
            string += str(node) + ';'
        string += '}>'
        return string


class IfNode(Node):
    def __init__(self, condition: Node, body: List[Node] = None):
        super().__init__(self)
        self.condition = condition
        if body is None:
            self.body = []
        else:
            self.body = body
            for b in body:
                b.parent = self.parent
        return

    def __repr__(self):
        string = f'<if ({self.condition}) ' + '{'
        for node in self.body:
            string += str(node) + ';'
        string += '}>'
        return string


class ElseNode(Node):
    def __init__(self, body: List[Node] = None):
        super().__init__(self)
        if body is None:
            self.body = []
        else:
            self.body = body
            for b in body:
                b.parent = self.parent
        return

    def __repr__(self):
        string = '<else {'
        for node in self.body:
            string += str(node) + ';'
        string += '}>'
        return string


class ForNode(ScopedNode):
    def __init__(self, var: Node, condition: Node, step: Node):
        super().__init__(self)
        self.var = var
        self.var.parent = self
        self.condition = condition
        self.condition.parent = self
        self.step = step
        self.step.parent = self
        return

    def __repr__(self):
        string = f'<for ({self.var};{self.condition};{self.step}) ' + '{'
        for node in self.body:
            string += str(node) + ';'
        string += '}>'
        return string


class WhileNode(ScopedNode):
    def __init__(self, condition: Node):
        super().__init__(self)
        self.condition = condition
        self.condition.parent = self
        return

    def __repr__(self):
        string = f'<while ({self.condition}) ' + '{'
        for node in self.body:
            string += str(node) + ';'
        string += '}>'
        return string


class DoWhileNode(ScopedNode):
    def __init__(self):
        super().__init__(self)
        self.condition = None
        return

    def assign_cnd(self, condition: Node):
        self.condition = condition
        self.condition.parent = self
        return

    def __repr__(self):
        string = '<do {'
        for node in self.body:
            string += str(node) + ';'
        string += '}' + f' while ({self.condition})>'
        return string


class SwitchNode(ScopedNode):
    def __init__(self, switch_val: Node):
        super().__init__(self)
        self.switch_val = switch_val
        return

    def __repr__(self):
        string = f'<switch ({self.switch_val}) ' + '{'
        for node in self.body:
            string += str(node) + ';'
        string += '}>'
        return string


class KeywordNode(Node):
    def __init__(self, value):
        super().__init__(value)


class ReturnNode(Node):
    def __init__(self, value: Node):
        super().__init__(self)
        self.ret_value = value

    def __repr__(self):
        return f'return {self.value}'


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
        self.start_index = self.i
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
                if self.has_next() and self.peak == '/':    # double slash means inline comment
                    self.advance()
                    self.inline_comment()
                elif self.has_next() and self.peak == '*':
                    self.multi_line_comment()

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

        elif self.peak.isalpha():
            self.make_word()

        elif self.peak == '"':
            self.make_string()

        elif self.peak == "'":
            self.make_char()

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
                if self.peak == '>':
                    self.advance()
                    if self.peak == '=':
                        self.advance()
                        self.token(TT.assign_ushr)
                    else:
                        self.token(TT.ushr)
                elif self.peak == '=':
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
                if self.last_tok.type in op_table:  # its unary plus, thus can be ignored
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
                if self.last_tok is None or self.last_tok.type in op_table:  # its unary minus
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
                self.make_word()

        if word in keywords:
            self.token(keywords[word], keywords[word].value)
        else:
            self.token(TT.word, word)
        return

    def make_string(self) -> None:
        string = self.peak
        self.advance()
        while self.has_next() and self.peak != '"' and self.peak != '\n':
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
            self.token(TT.string, char)
        self.advance()
        return

    def inline_comment(self) -> None:
        comment = ''
        while self.has_next() and self.peak != '\n':
            comment += self.peak
            self.advance()
        self.token(TT.comment, comment)
        self.advance()
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
        tok = Token(tt, self.start_index, self.start, self.end-1, self.n, value)
        self.tokens.append(tok)
        self.reset_tok_pos()
        self.last_tok = tok
        return tok

    def error(self, error: E, *args) -> None:
        self.errors.append(Error(error, self.start, self.end, self.n, self.file_name, self.lines[self.n-1], args))
        self.reset_tok_pos()
        return

    def warning(self, error: E, *args) -> None:
        self.errors.append(Error(error, self.start, self.end, self.n, self.file_name, self.lines[self.n - 1], args))
        self.reset_tok_pos()
        return

    def reset_tok_pos(self) -> None:
        self.start = self.end
        self.start_index = self.i + 1
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
# Parser
#########################################


class Parser:
    def __init__(self, program: str, toks: List[Token], file_name: str):
        self.toks = toks
        self.toks.append(Token(TT.eof, None, None, None, None))
        self.len = len(self.toks)
        self.i = 0
        self.peak = self.toks[self.i]
        self.last = None
        self.lines = program.split('\n')
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
            t = self.peak.type

            if t in types and self.peak.value == t.value and not (t in {TT.func, TT.object_}):
                body.append(self.assign_var())

            elif t == TT.func:
                body.append(self.func_def())

            elif t == TT.object_:
                body.append(self.make_object())

            elif t == TT.if_:
                body += self.make_if()

            elif t == TT.elif_:
                self.error(E.if_expected, self.peak)

            elif t == TT.else_:
                self.error(E.if_expected, self.peak)

            elif t == TT.switch:
                body.append(self.make_switch())

            elif t == TT.case:
                body.append(self.make_case())

            elif t == TT.default:
                body.append(self.make_default())

            elif t == TT.exit_:
                body.append(self.make_exit())

            elif t == TT.skip:
                body.append(self.make_skip())

            elif t == TT.for_:
                body.append(self.make_for())

            elif t == TT.while_:
                body.append(self.make_while())

            elif t == TT.do_:
                body.append(self.make_do_while())

            elif t == TT.return_:
                body.append(self.make_return())

            elif t == TT.goto:
                body.append(self.make_goto())

            elif t == TT.urcl:
                body.append(self.make_urcl())

            else:
                self.make_expression()
        return body

    def make_expression(self) -> Node:
        toks = self.shunting_yard()
        node = self.make_ast(toks)
        if self.peak.type in {TT.comma, TT.colon, TT.semi_col}:     # TT.rpa is not there cause it should not consume it
            self.advance()
        return node

    def next_expressions(self) -> List[Node]:
        if self.peak.type == TT.lcbr:
            self.advance()
            expression = self.parse_body(TT.rcbr)
            self.advance()
            return expression
        else:
            return [self.make_expression()]

    def shunting_yard(self) -> List[Token]:
        queue: List[Token] = []
        stack = []
        while self.has_next() and self.peak.type not in {TT.comma, TT.colon, TT.semi_col, TT.rcbr, TT.lcbr, TT.eof}:
            type = self.peak.type
            if type == TT.lpa:
                if self.last.type == TT.word:
                    self.peak.type = TT.func_call
                stack.append(self.peak)

            elif type == TT.lbr:
                if self.last.type == TT.word:
                    self.peak.type = TT.address
                    stack.append(self.peak)
                else:   # it's an array then
                    # TODO
                    pass

            elif type == TT.rbr:
                while len(stack) > 0 and stack[-1].type != TT.address:
                    queue.append(stack.pop())
                if len(stack) > 0:
                    queue.append(stack.pop())

            elif type in op_table:
                while len(stack) > 0 and (op_table[type][0] > op_table[stack[-1].type][0] or
                                          (op_table[type][0] == op_table[stack[-1].type][0] and op_table[type][1])):
                    queue.append(stack.pop())
                stack.append(self.peak)

            elif type == TT.rpa:
                while len(stack) > 0 and stack[-1].type != TT.lpa and stack[-1].type != TT.func_call:
                    queue.append(stack.pop())
                if len(stack) > 0:
                    if stack[-1].type == TT.func_call:
                        queue.append(stack.pop())
                    else:
                        stack.pop()
                        self.advance()
                        if len(stack) == 0 and self.peak.type not in op_table:
                            break
                        continue
                else:
                    # self.advance()
                    break   # we found the matching close parenthesis
            else:
                queue.append(self.peak)
            self.advance()

        while len(stack) > 0:
            queue.append(stack.pop())
        return queue

    def make_ast(self, queue) -> Node:
        stack: List[Node] = []
        for tok in queue:
            if tok.type in op_table:
                if tok.type in unary_ops:
                    node_a = stack.pop()
                    stack.append(UnOpNode(tok, node_a))
                else:
                    node_b = stack.pop()
                    node_a = stack.pop()
                    stack.append(BinOpNode(tok, node_a, node_b))
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
        if self.peak.type != TT.word:
            self.error(E.identifier_expected, self.peak)

        name = self.peak
        expression = self.make_expression()
        if type(expression) != ValueNode:
            declair_node = VarDeclarationNode(var_type, name, expression)
        else:
            declair_node = VarDeclarationNode(var_type, name)
        return declair_node

    def func_def(self) -> Node:
        self.advance()
        ret_type = self.peak
        if ret_type.type != TT.null and ret_type.type not in types:
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
        while self.peak.type not in {TT.comma, TT.colon, TT.lcbr, TT.rpa}:
            args.append(self.assign_var())

        self.advance()
        node = FuncDeclarationNode(ret_type, func_name, args)
        self.scope.append(node)
        body = self.next_expressions()
        self.scope.pop()
        node.assign_body(body)
        return node

    def make_object(self) -> Node:
        # TODO
        return

    def make_if(self) -> List[Node]:
        self.advance()
        condition = self.make_expression()
        body = self.next_expressions()
        if self.peak.type == TT.elif_:
            return [IfNode(condition, body), self.make_elif()]
        elif self.peak.type == TT.else_:
            return [IfNode(condition, body), self.make_else()]
        return [IfNode(condition, body)]

    def make_elif(self) -> List[Node]:
        self.advance()
        condition = self.make_expression()
        body = self.next_expressions()
        if self.peak.type == TT.elif_:
            return [ElseNode([IfNode(condition, body)]), self.make_elif()]
        elif self.peak.type == TT.else_:
            return [ElseNode([IfNode(condition, body)]), self.make_else()]
        return [ElseNode([IfNode(condition, body)])]

    def make_else(self):
        self.advance()
        body = self.next_expressions()
        return ElseNode(body)

    def make_switch(self) -> SwitchNode:
        self.advance()
        switch_val = self.make_expression()
        node = SwitchNode(switch_val)

        self.scope.append(node)
        body = self.next_expressions()
        self.scope.pop()

        node.assign_body(body)
        return node

    def make_case(self) -> Node:
        # TODO
        return

    def make_default(self) -> KeywordNode:
        self.advance()
        # TODO
        return

    def outside_loop(self, switch_count=False) -> bool:
        for node in reversed(self.scope):
            t = type(node)
            if switch_count and t in {ForNode, WhileNode, DoWhileNode, SwitchNode}:
                return False
            if t in {ForNode, WhileNode, DoWhileNode}:
                return False
            if t == FuncDeclarationNode:
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
        if self.peak.type == TT.lpa:    # i need to patch this.
            self.advance()
        # for cannot handle the parenthesis because its 3 expressions. i can make a method to grab all 3 expressions
        # if it has a parenthesis, so it will reach the end and it will

        setup = self.make_expression()
        end_cnd = self.make_expression()
        step = self.make_expression()
        node = ForNode(setup, end_cnd, step)

        self.scope.append(node)
        body = self.next_expressions()
        self.scope.pop()

        node.assign_body(body)
        return node

    def make_while(self) -> WhileNode:
        self.advance()
        condition = self.make_expression()
        node = WhileNode(condition)

        self.scope.append(node)
        body = self.next_expressions()
        self.scope.pop()

        node.assign_body(body)
        return node

    def make_do_while(self) -> Node:
        self.advance()
        node = DoWhileNode()

        self.scope.append(node)
        body = self.next_expressions()
        self.scope.pop()

        node.assign_body(body)

        if self.peak.type != TT.while_:
            self.error(E.while_expected, self.peak)

        condition = self.make_expression()
        node.assign_cnd(condition)
        return node

    def make_return(self) -> ReturnNode:    # TODO bare return not supported yet
        self.advance()
        expression = self.make_expression()
        return ReturnNode(expression)

    def make_goto(self) -> Node:
        # TODO
        return

    def make_urcl(self) -> Node:
        # TODO
        return

    # utils

    def error(self, error: E, tok: Token, *args) -> None:
        raise ErrorException(Error(error, tok.start, tok.end, tok.line, self.file_name, self.lines[tok.line-1], args))

    def advance(self, i=1) -> None:
        self.i += i
        self.last = self.peak
        while self.has_next() and self.toks[self.i].type == TT.comment:
            self.i += 1
        if self.has_next():
            self.peak = self.toks[self.i]

    def has_next(self, i=0) -> bool:
        return self.i + i < self.len


if __name__ == "__main__":
    main()
