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

    source = '''1+1*1'''

    if src_name is not None:
        if os.path.isfile(src_name):
            with open(src_name, mode='r') as sf:
                source = sf.read().replace("\r", "")
        else:
            print(f'"{src_name}" is not a file', file=stderr)
            exit(1)
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

    parser = Parser(source, lexer.tokens, dest_name)
    parser.parse()
    print(parser.output)
    if len(parser.errors) > 0:
        for err in lexer.errors:
            print(err, file=stderr)
        exit(1)

    return


#########################################
# Global Variables
#########################################

bases = 'oOxXbBdD'


class TT(Enum):
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
    sub = '-'
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
    'while': TT.while_,
    'do': TT.do_,
    'return': TT.return_,
    'goto': TT.goto,

    'urcl': TT.urcl,
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
}

op_precedence = {
    # left to right
    TT.inc: 1,
    TT.dec: 1,
    TT.lpa: 1,
    TT.lbr: 1,
    TT.dot: 1,
    # right to left
    TT.not_: 2,
    TT.b_not: 2,
    # unary plus and minus go here on 2
    # type cast should be here too but ill make it the same as function call above

    # left to right
    TT.mlt: 3,
    TT.div: 3,
    TT.mod: 3,
    TT.add: 4,
    TT.sub: 4,
    TT.ushr: 5,
    TT.shr: 5,
    TT.shl: 5,
    TT.gt: 6,
    TT.gte: 6,
    TT.lt: 6,
    TT.lte: 6,
    TT.dif: 7,
    TT.equ: 7,
    TT.b_and: 8,
    TT.b_xor: 9,
    TT.b_or: 10,
    TT.and_: 11,
    TT.or_: 12,
    # right to left
    TT.assign: 14,
    TT.assign_shl: 14,
    TT.assign_ushr: 14,
    TT.assign_shr: 14,
    TT.assign_add: 14,
    TT.assign_sub: 14,
    TT.assign_div: 14,
    TT.assign_mlt: 14,
    TT.assign_b_and: 14,
    TT.assign_b_or: 14,
    TT.assign_b_not: 14,
    TT.assign_mod: 14,
}


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
    invalid_literal = 'Invalid literal'
    invalid_char = 'Invalid character'
    miss_close_sym = 'Missing single quote {}'

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
# AST
#########################################


class Node:
    def __init__(self, value):
        self.value = value
        return

    def __repr__(self):
        return self.value.__repr__()


class BinOpNode(Node):
    def __init__(self, op: Token, left: Node, right: Node):
        super().__init__(self)
        self.left_child = left
        self.right_child = right
        self.op = op
        return

    def __repr__(self):
        return f'{self.left_child} {self.op.type} {self.right_child}'


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
        self.len = len(toks)
        self.i = 0
        self.peak = self.toks[self.i]
        self.lines = program.split('\n')
        self.file_name = file_name
        self.output: List[Node] = []
        self.errors = []
        return

    def parse(self):
        while self.has_next():
            self.make_expression()
        return

    def make_expression(self):
        toks = self.shunting_yard()
        astree = self.make_ast(toks)
        self.output.append(astree)
        return

    def shunting_yard(self) -> List[Token]:
        queue: List[Token] = []
        stack = []
        while self.has_next():
            type = self.peak.type
            if type == TT.lpa:
                stack.append(self.peak)

            elif type in op_precedence:
                while len(stack) > 0 and op_precedence[type] > op_precedence[stack[-1].type]:
                    queue.append(stack.pop())
                stack.append(self.peak)

            elif type == TT.rpa:
                while len(stack) > 0 and stack[-1].type != TT.lpa:
                    queue.append(stack.pop())
                if len(stack) > 0:
                    stack.pop()
                else:
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
            if tok.type in op_precedence:
                node_b = stack.pop()
                node_a = stack.pop()
                stack.append(BinOpNode(tok, node_a, node_b))
            else:
                stack.append(Node(tok))
        return stack.pop()
    
    def error(self, error: E, tok: Token, *args):
        self.errors.append(Error(error, tok.start, tok.end, tok.line, self.file_name, self.lines[tok.line], args))
        return

    def advance(self, i=1):
        self.i += i
        if self.has_next():
            self.peak = self.toks[self.i]

    def has_next(self, i=0):
        return self.i + i < self.len


if __name__ == "__main__":
    main()
