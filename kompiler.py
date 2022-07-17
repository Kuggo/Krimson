from enum import Enum
from sys import argv, stdout, stderr
import os

from typing import List


def main():
    src_name = argv[1] if len(argv) >= 2 else None
    dest_name = argv[2] if len(argv) >= 3 else None

    if src_name == '--help':
        print('usage: krimson <source_file> <destination_file>')

    source = '''int var = 69'''

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
    return


#########################################
# Global Variables
#########################################

bases = 'oOxXbBdD'

op_precedence = {}

#########################################
# Classes
#########################################


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
    ushl = '>>>'
    b_and = '&'
    b_or = '|'
    b_not = '~'

    # variable assignment
    assign = '='

    assign_add = '+='
    assign_sub = '-='
    assign_mlt = '*='
    assign_div = '/='

    assign_shl = '<<='
    assign_shr = '>>='
    assign_ushl = '>>>='
    assign_b_and = '&='
    assign_b_or = '|='
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

        self.peak = self.pr[self.i]
        self.start_index = self.i
        self.start = self.end
        return

    def tokenize(self):
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

    def make_token(self):
        if self.peak in symbols:
            self.make_symbol()

        elif self.peak.isnumeric():
            self.make_num()

        elif self.peak.isalpha():
            self.make_word()
        return

    def make_symbol(self):
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
                        self.token(TT.assign_ushl)
                    else:
                        self.token(TT.ushl)
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

    def make_word(self):
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

    def reset_tok_pos(self):
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


class Parser:
    def __init__(self, program: str, toks: List[Token], file_name: str):
        self.toks = toks
        self.len = len(toks)
        self.i = 0
        self.lines = program.split('\n')
        self.file_name = file_name
        self.errors = []
        return
    
    def error(self, error: E, tok: Token, *args):
        self.errors.append(Error(error, tok.start, tok.end, tok.line, self.file_name, self.lines[tok.line], args))
        return

    def advance(self, i=1):
        self.i += i

    def has_next(self, i=0):
        return self.i + i < self.len

    def peak(self):
        return self.toks[self.i]


if __name__ == "__main__":
    main()
