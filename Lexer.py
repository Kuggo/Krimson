from compiler import FILE_NAME, PROGRAM_LINES
from enum import Enum


class TT(Enum):
    SEPARATOR = 0
    OPERATOR = 1
    KEYWORD = 2
    NAME = 3
    LITERAL = 4


class Token:
    def __init__(self, token_type, value, start=-1, end=-1, line=-1):
        self.tt: TT = token_type
        self.value = value
        self.start = start
        self.end = end
        self.line = line

    def __eq__(self, other):
        return other is not None and self.tt == other.tt and self.value == other.value

    def __repr__(self):
        return f'<{self.line}:{self.start}:{self.end}: {self.tt.name}, {self.value}>'


class Operators(Enum):
    dif = Token(TT.OPERATOR, '!=')
    equ = Token(TT.OPERATOR, '==')
    gt = Token(TT.OPERATOR, '>')
    lt = Token(TT.OPERATOR, '<')
    gte = Token(TT.OPERATOR, '>=')
    lte = Token(TT.OPERATOR, '<=')

    shl = Token(TT.OPERATOR, '<<')
    shr = Token(TT.OPERATOR, '>>')
    b_and = Token(TT.OPERATOR, '&')
    b_or = Token(TT.OPERATOR, '|')
    b_xor = Token(TT.OPERATOR, '^')
    b_not = Token(TT.OPERATOR, '~')

    add = Token(TT.OPERATOR, '+')
    sub = Token(TT.OPERATOR, '- ')
    neg = Token(TT.OPERATOR, '-')
    mlt = Token(TT.OPERATOR, '*')
    div = Token(TT.OPERATOR, '/')
    mod = Token(TT.OPERATOR, '%')

    index = Token(TT.OPERATOR, '[]')

    assign = Token(TT.OPERATOR, '=')

    assign_shl = Token(TT.OPERATOR, '<<=')
    assign_shr = Token(TT.OPERATOR, '>>=')
    assign_b_and = Token(TT.OPERATOR, '&=')
    assign_b_or = Token(TT.OPERATOR, '|=')
    assign_b_xor = Token(TT.OPERATOR, '^=')

    assign_add = Token(TT.OPERATOR, '+=')
    assign_sub = Token(TT.OPERATOR, '-=')
    assign_mlt = Token(TT.OPERATOR, '*=')
    assign_div = Token(TT.OPERATOR, '/=')
    assign_mod = Token(TT.OPERATOR, '%=')


class Separators(Enum):
    comma = Token(TT.SEPARATOR, ',')
    colon = Token(TT.SEPARATOR, ':')
    # nln = Token(TT.SEPARATOR, '\n')
    semi_col = Token(TT.SEPARATOR, ';')
    lpa = Token(TT.SEPARATOR, '(')
    rpa = Token(TT.SEPARATOR, ')')
    lbr = Token(TT.SEPARATOR, '[')
    rbr = Token(TT.SEPARATOR, ']')
    lcb = Token(TT.SEPARATOR, '{')
    rcb = Token(TT.SEPARATOR, '}')
    eof = Token(TT.SEPARATOR, 'eof')


class E(Enum):
    name_expected = 'Name expected'
    duplicate_class = 'Duplicate class name'
    duplicate_func = 'Duplicate function declaration'
    duplicate_var = 'Duplicate variable name'
    duplicate_macro = 'Duplicate macro'
    literal_expected = 'Literal expected'
    symbol_expected = '{} expected'
    symbol_expected_before = "'{}' expected before '{}'"
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
    generic_expected = 'Generic type/value expected'
    unexpected_argument = 'Unexpected argument'
    break_not_in_loop = 'break keyword found outside a loop body'
    skip_not_in_loop = 'skip keyword found outside a loop body'
    return_not_in_func = 'return keyword found outside a function body'
    unknown_var_type = 'Unknown variable type'
    undefined_variable = 'Undefined variable'
    cannot_assign_to_non_var = '{} is not a variable that can be assigned a value to'
    var_before_assign = 'Variable might be used before assignment'
    undefined_function = 'Undefined function for the given args'
    unknown_obj_type = 'Unknown object type'
    unknown_name = "No variable, class or function named '{}' is visible in scope"
    no_attribute = "'{}' has no attribute '{}'"
    type_missmatch = "expected '{}' and got '{}'"
    type_incompatible = "Expected subtype of '{}' and got '{}'"
    bin_dunder_not_found = 'Cannot {} for {} and {}. No suitable declaration of {} exists anywhere'
    unary_dunder_not_found = 'Cannot {} for {}. No suitable declaration of {} exists anywhere'
    constructor_outside_class = "Constructor for class '{}' found outside its class"
    this_outside_class = 'this keyword found outside an class definition'
    this_on_static = 'this keyword cannot be used in a static context'
    super_outside_class = 'super keyword found outside an class definition'
    class_is_not_subtype = "class '{}' does not inherit from another class"
    instance_needed = "Cannot access fields of object '{}' without an instance of it"
    static_class_no_constructor = "Static class cannot have a constructor"
    static_not_in_class_scope = 'Static modifier cannot be applied outside a Class definition scope'
    cannot_default_arg = "Cannot assign a default value to function argument '{}'"

    def __repr__(self) -> str:
        return self.value


class Error(Exception):
    def __init__(self, error: E, start, end, line, code_line, *args):
        self.e: E = error
        self.start = start
        self.end = end
        self.line = line
        self.code_line = code_line
        self.file_name = FILE_NAME

        self.args = args
        return

    def __repr__(self):
        string = f'{self.file_name}:{self.start}:{self.line}: {self.e.value.format(*self.args)}\n'
        string += self.code_line + '\n'
        string += ' ' * (self.start - 1)
        string += '^' * (self.end - self.start + 1)
        return string


# constants
SEPARATORS = {'.', ',', ':', ';', '{', '}', '[', ']', '(', ')'}

KEYWORDS = {'if', 'else', 'break', 'skip', 'while', 'return', 'class', 'func', 'var'}

SYMBOLS = {'&', '|', '+', '-', '*', '/', '=', '<', '>', '!', '(', ')', '{', '}', '[', ']'}

UNARY_OPERATORS = {'!', '-', '~'}

ASSIGN_OPERATORS = {'=', '+=', '-=', '*=', '/=', '%=', '<<=', '>>=', '|=', '&=', '^='}


class Lexer:
    def __init__(self, input_string):
        self.input_string: str = input_string + '    '
        self.i: int = 0
        self.len: int = len(self.input_string)
        self.line: int = 1
        self.line_offset: int = -1
        self.peak: str = self.input_string[0]
        self.tokens: list[Token] = []
        self.errors: list[Error] = []

    def tokenize(self):
        while self.i < len(self.input_string):
            if self.peak.isspace():
                self.advance()
                continue

            elif self.peak == '/':
                if self.preview() == '/':
                    self.inline_comment()

                elif self.preview() == '*':
                    self.multi_line_comment()

                else:
                    self.token(TT.OPERATOR, self.peak, self.i, self.i)
                    self.advance()

            elif self.peak in SYMBOLS:
                self.make_symbol()

            elif self.peak.isalpha() or self.peak == '_':
                self.make_word()

            elif self.peak.isdigit():
                self.make_number()

            elif self.peak == '"':
                self.make_string()

            elif self.peak == "'":
                self.make_char()

            else:
                raise self.error(E.invalid_char, self.i, self.i)

        return

    def make_word(self):
        start = self.i
        while self.has_next() and (self.peak.isalnum() or self.peak == '_'):
            self.advance()

        name = self.input_string[start:self.i]
        if name in KEYWORDS:
            token_type = TT.KEYWORD
        else:
            token_type = TT.NAME

        self.token(token_type, name, start, self.i - 1)
        return

    def make_number(self):
        start = self.i
        while self.has_next() and self.peak.isdigit():
            self.advance()
        value = int(self.input_string[start:self.i], 0)
        self.token(TT.LITERAL, value, start, self.i - 1)
        return

    def make_symbol(self) -> None:
        start = self.i
        if self.peak == '<':
            if self.preview() == '<':
                self.advance()
                if self.preview() == '=':
                    self.advance()
                    self.token(TT.OPERATOR, '<<=', start, self.i)
                else:
                    self.token(TT.OPERATOR, '<<', start, self.i)
            elif self.preview() == '=':
                self.advance()
                self.token(TT.OPERATOR, '<=', start, self.i)

            else:
                self.token(TT.OPERATOR, '<', start, self.i)

        elif self.peak == '>':
            if self.preview() == '>':
                self.advance()
                if self.preview() == '=':
                    self.advance()
                    self.token(TT.OPERATOR, '>>=', start, self.i)
                else:
                    self.token(TT.OPERATOR, '>>', start, self.i)
            elif self.preview() == '=':
                self.advance()
                self.token(TT.OPERATOR, '>=', start, self.i)

            else:
                self.token(TT.OPERATOR, '>', start, self.i)

        elif self.peak == '=':
            if self.preview() == '=':
                self.advance()
                self.token(TT.OPERATOR, '==', start, self.i)
            else:
                self.token(TT.OPERATOR, '=', start, self.i)

        elif self.peak == '!':
            if self.preview() == '=':
                self.advance()
                self.token(TT.OPERATOR, '!=', start, self.i)
            else:
                self.token(TT.OPERATOR, '!', start, self.i)

        elif self.peak == '&':
            if self.preview() == '&':
                self.advance()
                self.token(TT.OPERATOR, '&&', start, self.i)

            elif self.preview() == '=':
                self.advance()
                self.token(TT.OPERATOR, '&=', start, self.i)
            else:
                self.token(TT.OPERATOR, '&', start, self.i)

        elif self.peak == '|':
            if self.preview() == '|':
                self.advance()
                self.token(TT.OPERATOR, '||', start, self.i)

            elif self.preview() == '=':
                self.advance()
                self.token(TT.OPERATOR, '|=', start, self.i)
            else:
                self.token(TT.OPERATOR, '|', start, self.i)

        elif self.peak == '^':
            if self.preview() == '=':
                self.advance()
                self.token(TT.OPERATOR, '^=', start, self.i)
            else:
                self.token(TT.OPERATOR, '^', start, self.i)

        elif self.peak == '+':
            if self.preview() == '=':
                self.advance()
                self.token(TT.OPERATOR, '+=', start, self.i)
            else:
                self.token(TT.OPERATOR, '+', start, self.i)

        elif self.peak == '-':
            if self.preview() == '=':
                self.advance()
                self.token(TT.OPERATOR, '-=', start, self.i)
            else:
                if self.tokens[-1].tt == TT.OPERATOR or self.tokens[-1].value == ')':
                    self.token(TT.OPERATOR, '-', start, self.i)
                else:
                    self.token(TT.OPERATOR, '- ', start, self.i)

        elif self.peak == '*':
            if self.preview() == '=':
                self.advance()
                self.token(TT.OPERATOR, '*=', start, self.i)
            else:
                self.token(TT.OPERATOR, '*', start, self.i)

        elif self.peak == '/':
            if self.preview() == '=':
                self.advance()
                self.token(TT.OPERATOR, '/=', start, self.i)
            else:
                self.token(TT.OPERATOR, '/', start, self.i)

        elif self.peak == '%':
            if self.preview() == '=':
                self.advance()
                self.token(TT.OPERATOR, '%=', start, self.i)
            else:
                self.token(TT.OPERATOR, '%', start, self.i)

        else:
            self.token(TT.SEPARATOR, self.peak, self.i, self.i)

        self.advance()
        return

    def make_string(self):
        self.advance()
        start = self.i

        while self.has_next() and self.peak != '"':
            self.advance()

        string = self.input_string[start:self.i].encode('raw_unicode_escape').decode('unicode_escape')
        self.token(TT.LITERAL, string, start - 1, self.i)
        self.advance()
        return

    def make_char(self):
        self.advance()
        start = self.i

        while self.has_next() and self.peak != "'":
            if self.peak == '\n':
                self.error(E.miss_close_sym, start, self.i, "'")
                return
            self.advance()

        char = self.input_string[start:self.i].encode('raw_unicode_escape').decode('unicode_escape')
        if len(char) != 1:
            self.error(E.invalid_char, start - 1, self.i)
        else:
            self.token(TT.LITERAL, char, start - 1, self.i)
            self.advance()
        return

    def inline_comment(self) -> None:
        while self.has_next() and self.peak != '\n':
            self.advance()
        return

    def multi_line_comment(self) -> None:
        self.advance(2)
        while self.has_next(1):
            if self.peak == '*':
                self.advance()
                if self.peak == '/':
                    self.advance()
                    return
            self.advance()

    def token(self, tt: TT, value, start, end) -> Token:
        tok = Token(tt, value, start - self.line_offset, end - self.line_offset, self.line)
        self.tokens.append(tok)
        return tok

    def error(self, error: E, start, end, *args) -> None:
        self.errors.append(
            Error(error, start - self.line_offset, end - self.line_offset, self.line, PROGRAM_LINES[self.line - 1],
                  args))
        self.advance()
        return

    def advance(self, i=1) -> None:
        if self.peak == '\n':
            self.line_offset = self.i
            self.line += 1
        self.i += i
        if self.has_next():
            self.peak = self.input_string[self.i]
        else:
            pass  # raise EOFError

    def preview(self, i=1) -> str:
        return self.input_string[self.i + i]

    def has_next(self, i=0) -> bool:
        return self.i + i < self.len
