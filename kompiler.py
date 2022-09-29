import ast
from enum import Enum
from sys import argv, stdout, stderr
import os

from typing import List, Dict, Tuple, Set, Union


def main():
    src_name = argv[1] if len(argv) >= 2 else None
    dest_name = argv[2] if len(argv) >= 3 else None

    if src_name == '--help':
        print('usage: krimson <source_file> <destination_file>')

    source = ''''''
    global file_name

    if src_name is not None:
        file_name = src_name

    if os.path.isfile(file_name):
        with open(file_name, mode='r') as sf:
            source = sf.read().replace("\r", "")
    else:
        print(f'"{src_name}" is not a file', file=stderr)
        exit(1)

    if src_name is None:
        file_name = 'terminal'

    if dest_name is not None:
        dest = open(dest_name, mode="w")
    else:
        dest = stdout

    lexer = Lexer(source)
    lexer.tokenize()
    print(lexer.tokens, file=dest)

    if len(lexer.errors) > 0:
        for err in lexer.errors:
            print(err, file=stderr)
        exit(1)

    parser = Parser(lexer.tokens)
    try:
        parser.parse()
        # print(parser.output, file=dest)   # im disabling this to see type checker output better
    except ErrorException as e:
        print(e.error, file=stderr)
        exit(1)

    read_lib(f_name='builtins')  # if no errors so far lets load default types

    global top_level
    top_level.assign_body(parser.output)
    top_level = top_level.lower()  # process all the nodes recursively

    type_checker = TypeChecker(parser.output)

    if len(type_checker.errors) > 0:
        for err in type_checker.errors:
            print(err, file=stderr)
        exit(1)

    for node in type_checker.asts:
        print(node, file=dest)

    # TODO compile to urcl
    return


imports = set()
file_extension = '.krim'
file_name = 'input.krim'
errors = []


def read_lib(f_name):
    if os.path.isdir(f_name):
        for subdir, dirs, files in os.walk(f_name):
            for file in files:
                read_file(file, f_name)

    elif os.path.isfile(f_name):
        read_file(f_name)

    else:
        print(f'Unable to find "{f_name}" library', file=stderr)


def read_file(f_name, up_dir=''):
    f_name = up_dir + '/' + f_name
    if not f_name.endswith(file_extension) or f_name in imports:
        return
    try:
        with open(f_name, 'r') as file:
            imports.add(f_name)
            try:
                code = file.read()
                lexer = Lexer(code, f_name)
                lexer.tokenize()
                parser = Parser(lexer.tokens, f_name)
                parser.parse()
                built_in = ScopedNode(0, len(code) - 1, '', 0)
                for t in parser.output:
                    if isinstance(t, ObjectDeclarationNode):
                        built_in.types[t.name.value] = t

                    elif isinstance(t, FuncDeclarationNode):
                        built_in.funcs[(t.name.value, t.get_arg_types())] = t

                    elif isinstance(t, VarDeclarationNode) or isinstance(t, MacroDefNode):
                        built_in.vars[t.name.value] = t

                return built_in

            except ErrorException as e:
                print(e.error, file=stderr)
                exit(1)
    except OSError:
        print(f'Unable to find "{f_name}" file', file=stderr)


class TT(Enum):
    eof = 'eof'

    word = 'word'
    num = 'num'
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

    weak = 'weak'
    static = 'static'
    func = 'fn'
    class_ = 'class'

    string = 'string'
    array = 'array'
    char = ' char'

    func_call = 'func()'
    address = '[]'

    # Default values
    true = 'true'
    false = 'false'
    null = 'null'
    super = 'super'
    this = 'this'

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

    urcl = 'urcl'
    generic = 'generic'

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


class Type:
    def __init__(self, t: str, weak: bool = False, value: bool = False, generics=tuple(), size=1):
        self.type = t
        self.weak = weak
        self.value_type = value
        self.generics: Tuple[Type] = generics
        self.size = size
        return

    def __eq__(self, other: 'Type'):
        return self.type == other.type and self.weak == other.weak and self.generics == other.generics

    def compatible(self, other) -> bool:
        if self == other:
            return True
        else:
            # TODO add all the rules for type conversion
            return False


keywords = {
    'func': TT.func,
    'class': TT.class_,
    'true': TT.true,
    'false': TT.false,
    'null': TT.null,
    'weak': TT.weak,
    'static': TT.static,
    'super': TT.super,
    'this': TT.this,

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
    TT.func_call: (1, True),
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
    TT.assign_mod: (14, False),
    TT.lpa: (20, True),
    TT.address: (19, True),
}

unary_ops = {TT.inc, TT.dec, TT.not_, TT.b_not, TT.neg, TT.func_call}

assign_ops: Dict[TT, TT] = {
    TT.assign: None,
    TT.assign_shl: TT.shl,
    TT.assign_shr: TT.shr,
    TT.assign_add: TT.add,
    TT.assign_sub: TT.sub,
    TT.assign_div: TT.div,
    TT.assign_mlt: TT.mlt,
    TT.assign_b_and: TT.b_and,
    TT.assign_b_or: TT.b_or,
    TT.assign_mod: TT.mod,
}

default_ops = {
    # operations
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

types: Dict[str, Type] = {
    'null': Type('null'),
    'bool': Type('bool'),
    'uint': Type('uint'),
}


class Token:
    def __init__(self, tt: TT, start_char, end_char, line: str, line_num: int, value=None):
        self.type = tt
        self.start = start_char
        self.end = end_char
        self.line = line
        self.line_num = line_num
        if value is None:
            self.value = ''
        else:
            self.value = value
        return

    def __repr__(self):
        return f"<{self.line}:{self.start}:{self.end}:{self.type} {self.value}>"


class E(Enum):
    name_expected = 'Name expected'
    duplicate_class = 'Duplicate class name'
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
    generic_expected = 'Generic type/value expected'
    unexpected_argument = 'Unexpected argument'
    exit_not_in_loop = 'exit keyword found outside a loop/switch body'
    skip_not_in_loop = 'skip keyword found outside a loop body'
    return_not_in_func = 'return keyword found outside a function body'
    unknown_var_type = 'Unknown variable type'
    undefined_variable = 'Undefined variable'
    var_before_assign = 'Variable might be used before assignment'
    undefined_function = 'Undefined function for the given args'
    unknown_obj_type = 'Unknown object type'
    unknown_name = "No variable, class or function named '{}' is visible in scope"
    no_attribute = "'{}' has no attribute '{}'"
    type_missmatch = "expected '{}' and got '{}'"
    type_incompatible = "Expected subtype of '{}' and got '{}'"
    bin_dunder_not_found = 'Cannot {} for {} and {}. No suitable declaration of {} exists anywhere'
    un_dunder_not_found = 'Cannot {} for {}. No suitable declaration of {} exists anywhere'
    weak_cant_be_on_value = "Cannot apply 'weak' modifier to value types"
    weak_cant_assign_constructor = "Cannot assign a new constructed object to a weak reference holder"
    constructor_outside_class = "Constructor for class '{}' found outside its class"
    this_outside_class = 'this keyword found outside an class definition'
    this_on_static = 'this keyword cannot be used in a static context'
    super_outside_class = 'super keyword found outside an class definition'
    class_not_found = "class '{}' not defined or not in scope"
    error_not_found = "Cannot yeet '{}' as its not a defined error code or its not visible in scope"
    class_is_not_subtype = "class '{}' does not inherit from another class"
    instance_needed = "Cannot access fields of object '{}' without an instance of it"
    static_class_no_constructor = "Static class cannot have a constructor"
    static_not_in_class_scope = 'Static modifier cannot be applied outside a Class definition scope'
    cannot_default_arg = "Cannot assign a default value to function argument '{}'"

    def __repr__(self) -> str:
        return self.value


class Error:
    def __init__(self, err: E, start: int, end: int, line: int, code_line: str, f_name=file_name, *args):
        self.error = err
        self.start = start
        self.end = end
        self.line = line
        self.file_name = f_name
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

def error(e: E, node, *args) -> None:
    errors.append(Error(e, node.start, node.end, node.line_num, node.line, *args))


def token(tt: TT, value=None):
    return Token(tt, None, None, '', 0, value)


class ErrorException(Exception):
    def __init__(self, e: E, node: ('Node', Token), f_name=file_name, *args):
        self.error = Error(e, node.start, node.end, node.line_num, node.line, f_name, *args)
        errors.append(self.error)
        return


#########################################
# Lexer
#########################################


class Lexer:
    def __init__(self, program: str, f_name=file_name):
        self.pr = program + '      \n'  # avoids checking self.has_next() tons of times
        self.file_name = f_name
        self.lines = program.split('\n')
        self.len = len(self.pr)
        self.i = 0
        self.n = 1
        self.end = 1
        self.tokens = []
        self.errors = []

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
                if self.last_tok is None or self.last_tok.type in op_table or self.last_tok.value in keywords:
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
        if self.last_tok.type == TT.neg:
            self.tokens.pop()
            num = '-'
        else:
            num = ''
        dot_count = 0
        while self.has_next() and (self.peak.isalnum() or self.peak == '.'):
            if self.peak == '.':
                if dot_count > 0:
                    self.token(TT.num, float(num))
                    self.advance()
                    self.token(TT.dot)
                    self.make_word()
                    return
                dot_count += 1
            num += self.peak
            self.advance()

        if dot_count > 0:
            try:
                self.token(TT.num, float(num))
            except ValueError:
                self.error(E.invalid_literal)
        else:
            try:
                self.token(TT.num, int(num, 0))
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
        tok = Token(tt, self.start, self.end - 1, self.lines[self.n - 1], self.n, value)
        self.tokens.append(tok)
        self.reset_tok_pos()
        self.last_tok = tok
        return tok

    def error(self, err: E, *args) -> None:
        self.errors.append(Error(err, self.start, self.end, self.n, self.lines[self.n - 1], self.file_name, args))
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

def op_to_dunder(op_type: TT, start, end, line, left_op, right_op=None):
    if op_type in default_ops:
        if right_op is None:
            dunder_func = top_level.types[left_op.type].get_func(default_ops[op_type], (left_op.type,))
        else:
            dunder_func = top_level.types[left_op.type].get_func(default_ops[op_type], (left_op.type, right_op.type))

        if dunder_func is not None:
            tok = Token(TT.word, start, end, line, default_ops[op_type])
            if right_op is None:
                func = FuncCallNode(ValueNode(tok), [left_op])
            else:
                func = FuncCallNode(ValueNode(tok), [left_op, right_op])
            func.type = dunder_func.type
            return func
    return


def implicit_type_convert(target_type: Type, node: 'ValueNode') -> None:
    type_set = (types['int'], types['uint'], types['fixed'], types['float'])

    if node.type == target_type or not (node.type in type_set and target_type in type_set):
        return

    # value conversion
    if node.value.type != TT.num:  # if the token type is not a number so no cast can be done
        return
    if target_type == types['float'] or target_type == types['fixed']:
        node.value.value = float(node.value.value)
    elif target_type == types['int']:
        node.value.value = int(node.value.value)
    elif target_type == types['uint']:
        node.value.value = abs(int(node.value.value))

    node.type = target_type  # type cast happened here
    return


class Node:
    def __init__(self, start, end, line, line_num, t=TT.null.value):
        self.parent: (ScopedNode, None) = None
        self.type: Type = types[t]
        self.start = start
        self.end = end
        self.line = line
        self.line_num = line_num

        self.lowered = False

        self.extra_nodes_before = None
        self.extra_nodes_after = None
        return

    def assign_parent(self, parent) -> None:
        self.parent = parent

    def lower(self) -> 'Node':
        return self

    def check(self) -> None:
        return

    def add_extra_nodes(self, nodes: List['Node'], before=True) -> None:
        # self.parent must not be None at this point
        if isinstance(self.parent, ScopedNode) or self.parent is None:
            if before:
                if self.extra_nodes_before is None:
                    self.extra_nodes_before = nodes
                else:
                    self.extra_nodes_before += nodes
            else:
                if self.extra_nodes_after is None:
                    self.extra_nodes_after = nodes
                else:
                    self.extra_nodes_after += nodes
            return  # it was found
        else:
            self.parent.add_extra_nodes(self, nodes, before=before)
        return  # node was not found so ignore it ig?

    def find_parent_class(self) -> ('ObjectDeclarationNode', None):
        if isinstance(self.parent, ObjectDeclarationNode):
            return self.parent
        elif self.parent is None:
            return None
        else:
            return self.parent.find_parent_class()

    def find_parent_function(self) -> ('FuncDeclarationNode', None):
        if isinstance(self.parent, FuncDeclarationNode):
            return self.parent
        elif self.parent is None:
            return None
        else:
            return self.parent.find_parent_function()

    def update_tc_dicts(self) -> None:
        return

    def is_weak(self) -> bool:
        return False

    def error(self, err: E, tok: (Token, 'Node'), *args) -> None:
        raise ErrorException(err, tok, file_name, args)


class ValueNode(Node):
    def __init__(self, value: Token):
        super().__init__(value.start, value.end, value.line, value.line_num)
        self.value = value
        return

    def find_type(self, func_call=False, expression=False) -> None:
        if self.value.type.value in default_values:
            self.type = default_values[self.value.type.value].type
        elif self.value.value in top_level.types:
            self.type = top_level.types[self.value.value].type
        elif self.value.value in top_level.vars:
            self.type = top_level.vars[self.value.value].type
        elif top_level.contains_func_name(self.value.value):
            self.type = TT.func
        else:
            parent = self.find_parent_class()
            if parent is not None and self.value.value in parent.static_vars:
                self.type = parent.static_vars[self.value.value].type
                return
            elif self.value.type == TT.word and parent is not None:
                generic_def = parent.get_generic_definition(self.value.value)
                if isinstance(generic_def, VarDeclarationNode):
                    self.type = generic_def.type
                else:
                    self.type = self.value.value
                return  # not changing the type

            parent = self.find_parent_function()
            if self.value.type == TT.word and parent is not None and self.value.value in parent.vars:
                self.type = parent.vars[self.value.value].type
                return
            elif func_call:
                self.error(E.undefined_function, self.value)
            elif expression:
                self.error(E.undefined_variable, self.value)
            else:
                self.error(E.unknown_name, self, self.value.value)

    def lower(self, func_call=False, expression=False, declaring=None) -> Node:
        if self.lowered:  # we did this already so shortcut
            return self
        self.find_type(func_call, expression)
        if self.value.type == TT.word:
            if self.value.value == declaring:
                return self
            if self.value.value in top_level.vars and isinstance(top_level.vars[self.value.value], MacroDefNode):
                top_level.vars[self.value.value].assign_parent(self.parent)
                return top_level.vars[self.value.value].expression.lower()
            if self.value.value in top_level.vars and self.value.value not in top_level.defined_vars:
                self.error(E.var_before_assign, self.value)

            if self.value.value in top_level.vars and isinstance(top_level.vars[self.value.value], VarDeclarationNode):
                if top_level.vars[self.value.value].type.value_type:
                    # it's a ref value. gotta check for null and for strong/weak refs
                    self.add_extra_nodes([null_ref_check(self), ref_check(self)], before=True)

        elif self.value.type == TT.this:
            current_class = self.find_parent_class()
            if current_class is None:
                self.error(E.this_outside_class, self)

            current_func = self.find_parent_function()
            if current_class.static or \
                    (current_func is not None and current_func.static and not current_func.constructor):
                self.error(E.this_on_static, self.value)

            self.type = current_class.name.value
            return self

        elif self.value.type == TT.super:
            current_class = self.find_parent_class()
            if current_class is None:
                self.error(E.super_outside_class, self)

            super_class = current_class.parent_class
            if super_class is None:
                self.error(E.class_is_not_subtype, self, current_class.name.value)

            self.type = super_class.value
            return self
        self.lowered = True
        return self

    def check(self) -> None:
        return

    def is_weak(self) -> bool:
        return self.value.value in top_level.vars and top_level.vars[self.value.value].is_weak(top_level)

    def __repr__(self):
        return f'{self.value}'


class UnOpNode(Node):
    def __init__(self, op: Token, child: (ValueNode, 'UnOpNode', 'BinOpNode', 'FuncCallNode')):
        super().__init__(None, None, op.line, op.line_num)
        if op.type in op_table and op_table[op.type][1]:
            self.start = child.start
            self.end = op.end
        else:
            self.start = op.start
            self.end = child.end
        self.op = op
        self.child = child
        return

    def lower(self, expression=True) -> Node:
        if self.lowered:  # we did this already so shortcut
            return self
        self.child.assign_parent(self.parent)
        self.child = self.child.lower(expression)
        if self.op.type in default_ops:
            dunder_func = op_to_dunder(self.op.type, self.start, self.end, self.line, self.child)
            if dunder_func is None:
                self.error(E.un_dunder_not_found, self.op, self.op.type.value, self.child.type,
                           default_ops[self.op.type])
            else:
                return dunder_func
        self.lowered = True
        return self

    def __repr__(self):
        if self.op.type in op_table and op_table[self.op.type][1]:
            return f'<{self.child} {self.op.type}>'
        else:
            return f'<{self.op.type} {self.child}>'


class BinOpNode(Node):
    def __init__(self, op: Token, left: (ValueNode, 'UnOpNode', 'BinOpNode', 'FuncCallNode'),
                 right: (ValueNode, 'UnOpNode', 'BinOpNode', 'FuncCallNode')):

        super().__init__(left.start, right.end, op.line, op.line_num)
        self.left_child = left
        self.right_child = right
        self.op = op
        return

    def lower(self, expression=True) -> ('BinOpNode', 'FuncCallNode'):
        if self.lowered:  # we did this already so shortcut
            return self
        self.left_child.assign_parent(self.parent)
        self.right_child.assign_parent(self.parent)
        self.left_child = self.left_child.lower(expression=expression)
        self.right_child = self.right_child.lower(expression=expression)
        if self.op.type not in default_ops:
            return self
        dunder_func = op_to_dunder(self.op.type, self.start, self.end, self.line, self.left_child, self.right_child)
        if dunder_func is None:
            self.error(E.bin_dunder_not_found, self.op, self.op.type.value, self.left_child.type,
                       self.right_child.type, default_ops[self.op.type])
        else:
            self.type = dunder_func.type
            self.lowered = True
            return dunder_func

    def __repr__(self):
        return f'<{self.left_child} {self.op.type} {self.right_child}>'


class DotOpNode(Node):
    def __init__(self, op: Token, obj: (ValueNode, 'DotOpNode'), prop: ValueNode):
        super().__init__(obj.start, prop.end, op.line, op.line_num)
        self.object = obj
        self.property: ValueNode = prop
        self.op = op
        return

    def lower(self, func_call=False, expression=False) -> Node:
        if self.lowered:  # we did this already so shortcut
            return self
        self.object.assign_parent(self.parent)
        self.property.assign_parent(self.parent)
        self.object = self.object.lower(expression=expression, func_call=False)

        if self.object.type in top_level.types and \
                top_level.types[self.object.type].has_attribute(self.property.value.value):
            object_type = top_level.types[self.object.type]
            self.type = object_type.get_attribute_type(self.property.value.value)
            if isinstance(self.object, ValueNode) and self.object.value.value == self.object.type:
                # it's the class name or super/this keywords
                if self.find_parent_class() != object_type:
                    if self.property.value.value in object_type.vars:
                        self.error(E.instance_needed, self.object, self.object.value.value)

            self.lowered = True
            return self
        else:
            self.error(E.no_attribute, self.property, self.object.value.value,
                       self.property.value.value)

    def is_weak(self) -> bool:
        if self.object.type in top_level.types and self.property.value.value in top_level.types[self.object.type].vars:
            var = top_level.types[self.object.type].vars[self.property.value.value]
            return var.is_weak(top_level)
        elif self.object.type in top_level.types and \
                self.property.value.value in top_level.types[self.object.type].static_vars:
            var = top_level.types[self.object.type].static_vars[self.property.value.value]
            return var.is_weak(top_level)
        else:
            return False

    def __repr__(self):
        return f'<{self.object}.{self.property}>'


class VarDeclarationNode(Node):
    def __init__(self, var_type: Token, name: Token, value: 'AssignNode' = None, weak=None, static=False):
        super().__init__(var_type.start, name.end, name.line, name.line_num)
        self.var_type = var_type
        self.type: Type = var_type.value
        self.weak = weak
        self.static = static
        self.name = name
        self.value = value
        return

    def lower(self, check_generics=True) -> 'VarDeclarationNode':
        if self.lowered:  # we did this already so shortcut
            return self
        if self.value is not None:
            self.value.assign_parent(self.parent)
            self.value = self.value.lower()
        if self.var_type.value not in top_level.types:
            self.error(E.identifier_expected, self.var_type)
        if self.weak is not None and self.var_type.value.value_type:
            self.error(E.weak_cant_be_on_value, self.weak)

        if self.weak is not None and isinstance(self.value.value, FuncCallNode) and \
                self.value.value.type == self.value.value.func_name:  # detecting a constructor here
            self.error(E.weak_cant_assign_constructor, self.weak)

        if self.static and self.parent != self.find_parent_class():
            # was not declared in the object's scope and its static
            self.error(E.static_not_in_class_scope, self)

        if check_generics:
            top_level.types[self.var_type.value].check_generics(self.var_type.generics, self.var_type.end + 1,
                                                                self.var_type.generic_end, self.line)
        self.lowered = True
        return self

    def update_tc_dicts(self) -> None:
        top_level.vars[self.name.value] = self
        return

    def __repr__(self):
        if self.value is None:
            return f'<{self.var_type.value} {self.name} = {self.value}>'
        else:
            return f'<{self.var_type.value} {self.value}>'


class AssignNode(Node):  # TODO ARC inc and dec refs
    def __init__(self, var: (ValueNode, DotOpNode), value: Node):
        super().__init__(var.start, value.end, value.line, value.line_num)
        self.var: (ValueNode, DotOpNode) = var
        self.value: Node = value
        return

    def lower(self) -> 'AssignNode':
        if self.lowered:  # we did this already so shortcut
            return self
        self.var.assign_parent(self.parent)
        self.value.assign_parent(self.parent)
        if isinstance(self.var, ValueNode):
            self.var = self.var.lower(declaring=self.var.value.value)
        else:
            self.var = self.var.lower()
        self.value = self.value.lower()

        if not self.var.type.value_type:  # reference type
            self.add_extra_nodes([], before=True)  # TODO add the code
            self.add_extra_nodes([], before=False)

        if isinstance(self.var, ValueNode) and isinstance(self.value, ValueNode):
            implicit_type_convert(self.var.value.value, self.value)

        if self.var.type != self.value.type:
            self.error(E.type_missmatch, self.value, self.var.type, self.value.type)

        if isinstance(self.var, ValueNode):
            top_level.defined_vars.add(self.var.value.value)

        self.lowered = True
        return self

    def __repr__(self):
        return f'<{self.var} = {self.value}>'


class ScopedNode(Node):
    def __init__(self, start=None, end=None, line=None, line_num=None):
        super().__init__(start, end, line, line_num)
        self.body = None
        self.vars: Dict[str, (VarDeclarationNode, MacroDefNode)] = {}
        self.funcs: Dict[Tuple[str, Tuple[Type]], FuncDeclarationNode] = {}
        self.types: Dict[str, ObjectDeclarationNode] = {}
        self.defined_vars: Set[str] = set()
        return

    def lower(self) -> Node:
        if self.lowered:  # we did this already so shortcut
            return self
        self.body = top_level.check_scope(self.body, context=(self.vars, self.funcs, self.types))
        self.lowered = True
        return self

    def check_scope(self, nodes, context: Tuple = None, class_scope=False) -> List[Node]:
        funcs = self.funcs.copy()
        variables = self.vars.copy()
        type_defs = self.types.copy()
        defined_vars = self.defined_vars.copy()

        if context is None:
            for node in nodes:
                node.update_tc_dicts(self)
        else:
            self.vars.update(context[0])
            self.funcs.update(context[1])
            self.types.update(context[2])
            for key in context[0]:
                self.defined_vars.add(key)

        body_lowered = []
        for node in nodes:
            if class_scope and not (isinstance(node, VarDeclarationNode) or isinstance(node, MacroDefNode) or
                                    isinstance(node, ScopedNode)):
                error(E.identifier_expected, node, file_name)
                continue
            try:
                node_lowered = node.lower()
                if node_lowered is not None:
                    if node_lowered.extra_nodes_before is not None:
                        try:
                            for nl in node_lowered.extra_nodes_before:
                                body_lowered.append(nl.lower(self))
                        except ErrorException:
                            continue
                        node_lowered.extra_nodes_before = None
                    body_lowered.append(node_lowered)

                    if node_lowered.extra_nodes_after is not None:
                        try:
                            for nl in node_lowered.extra_nodes_after:
                                body_lowered.append(nl.lower(self))
                        except ErrorException:
                            continue
                        node_lowered.extra_nodes_after = None
            except ErrorException:
                continue

        for i, node in enumerate(body_lowered):
            if node.extra_nodes_before is not None:
                try:
                    for j, n in enumerate(node.extra_nodes_before):
                        node.extra_nodes_before[j] = node.lower(self)
                except ErrorException:
                    continue
                body_lowered = body_lowered[:i] + node.extra_nodes_before + body_lowered[i:]
            if node.extra_nodes_after is not None:
                try:
                    for j, n in enumerate(node.extra_nodes_after):
                        node.extra_nodes_after[j] = node.lower(self)
                except ErrorException:
                    continue
                body_lowered = body_lowered[:i + 1] + node.extra_nodes_after + body_lowered[i + 1:]

        self.funcs = funcs
        self.vars = variables
        self.defined_vars = defined_vars
        self.types = type_defs
        return body_lowered

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
                    self.funcs[(b.name.value, b.get_arg_types())] = b
                elif isinstance(b, ObjectDeclarationNode):
                    self.types[b.name.value] = b
        if len(body) > 0:
            self.end = self.body[-1].end
        return

    def contains_func_name(self, name) -> bool:
        for key in self.funcs:
            if key[0] == name:
                return True
        return False

    def get_func(self, func_name, args, static_func_args=None) -> ('FuncDeclarationNode', None):
        if (func_name, args) in self.funcs:
            return self.funcs[(func_name, args)]
        elif static_func_args is not None and (func_name, static_func_args) in self.funcs:
            func = self.funcs[(func_name, static_func_args)]
            if func.static or func.constructor:
                return func

        for obj in self.types.values():
            if obj is None:
                continue
            func = obj.get_func(func_name, args, static_func_args)
            if func is not None:
                return func
        return None

    def contains_name(self, name) -> bool:
        return name in self.vars or name in self.types or self.contains_func_name(name)

    def types_compatible(self, target_type: Type, node_type: ValueNode) -> True:
        if target_type not in self.types or node_type.type not in self.types:
            return False
        implicit_type_convert(target_type, node_type)

        if node_type.value.value == target_type:
            return True

        for t in self.types.values():
            if t is not None and t.parent_class is not None:
                if node_type.type == t.parent_class.value:
                    return True
        return False

    def __repr__(self):
        string = ''
        for node in self.body:
            string += str(node) + '\n'
        return string


class FuncDeclarationNode(ScopedNode):
    def __init__(self, ret_type: Token, name: Token, args: List[VarDeclarationNode] = None, static=False):
        super().__init__(start=name.start, end=name.end, line=name.line, line_num=name.line_num)
        self.parent_obj = None
        self.type_tok = ret_type
        self.type = ret_type.value
        self.name = name
        self.static = static
        if args is None:
            self.args = []
        else:
            self.args = args
            for arg in args:
                arg.assign_parent(self)
        self.body = None
        self.constructor = False
        return

    def lower(self) -> Node:
        if self.lowered:  # we did this already so shortcut
            return self
        if self.type not in top_level.types and self.type != null.value.type.value:
            self.error(E.invalid_ret_type, self.type_tok)

        self.parent_obj = self.find_parent_class()
        if self.type == self.name.value:  # it's a constructor
            self.constructor = True
            if self.parent_obj is None or self.type != self.parent_obj.type:  # constructor outside its class definition
                self.error(E.constructor_outside_class, self, self.type)
            elif self.parent_obj.static:
                self.error(E.static_class_no_constructor, self)
            else:
                self.parent_obj.constructors[(self.type, self.get_arg_types(parent_obj=self.parent_obj))] = self

        if self.static:
            if self.parent != self.parent_obj:  # was not declared in the object's scope and its static
                self.error(E.static_not_in_class_scope, self)
        elif not self.constructor and self.parent_obj is not None:
            var_type = token(TT.word, self.parent_obj.type)
            self.args.insert(0, VarDeclarationNode(var_type, this.value))

        for i, arg in enumerate(self.args):
            self.vars[arg.name.value] = arg
            if arg.value is not None:
                error(E.cannot_default_arg, arg, file_name, arg.name.value)
            else:
                self.args[i] = arg.lower(check_generics=False)

        self.body = top_level.check_scope(self.body, context=(self.vars, self.funcs, self.types))

        if self.constructor:
            instance_vars: Set[str] = set()  # instance variable not assigned
            for node in self.parent_obj.vars.values():
                if node.value is None:
                    instance_vars.add(node.name.value)
                else:
                    self.body.insert(0, node.value)

            for node in self.body:
                if isinstance(node, AssignNode) and isinstance(node.var, ValueNode) and \
                        node.var.value.value in self.parent_obj.vars and node.var.value.value in instance_vars:
                    instance_vars.remove(node.var.value.value)

            for var_name in instance_vars:  # iterating the vars that weren't assigned
                var = self.parent_obj.vars[var_name]
                self.body.append(
                    AssignNode(ValueNode(Token(TT.word, var.start, var.end, var.line, var.line_num, var_name)), null))

        if self.constructor:  # we must return the new object
            node = ReturnNode(this, Token(TT.return_, self.end, self.end, self.line, self.line_num))
            node.assign_parent(self)
            self.body.append(node.lower())
        elif len(self.body) == 0 or not (isinstance(self.body[-1], ReturnNode) or isinstance(self.body[-1], UrclNode)):
            # body doesn't end with return, we will autogenerate it
            node = ReturnNode(null, Token(TT.return_, self.end, self.end, self.line, self.line_num))
            node.assign_parent(self)
            self.body.append(node.lower())
        self.lowered = True
        return self

    def update_tc_dicts(self) -> None:
        top_level.funcs[(self.name.value, self.get_arg_types())] = self
        return

    def get_arg_types(self, parent_obj: 'ObjectDeclarationNode' = None) -> Tuple[Type]:
        if parent_obj is None or parent_obj.static or self.static:
            args = []
        else:
            args = [parent_obj.type]
        for arg in self.args:
            args.append(arg.type)
        return tuple(args)

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
    def __init__(self, name: Token, generics: List[Union[ValueNode, VarDeclarationNode]], parent_class: Token = None,
                 static=False):
        super().__init__(start=name.start, end=name.end, line=name.line, line_num=name.line_num)
        self.name = name
        if name.value.startswith('__') and name.value.endswith('__'):
            self.type = name.value[2:-2]  # removing __type__
        else:
            self.type = name.value
        self.generics = generics
        self.static = static
        self.static_vars: Dict[Tuple[str, Tuple[Type]]] = {}
        self.constructors: Dict[Tuple[str, Tuple[Type]], FuncDeclarationNode] = {}
        self.destructor: (FuncDeclarationNode, None) = None
        self.parent_class = parent_class
        return

    def assign_body(self, body: List[Node] = None):
        if body is None:
            self.body = []
        else:
            self.body = body
            for b in body:
                b.assign_parent(self)
                if isinstance(b, VarDeclarationNode):
                    if self.static:
                        b.static = True
                    if b.static:
                        self.static_vars[b.name.value] = b
                    else:
                        self.vars[b.name.value] = b
                elif isinstance(b, FuncDeclarationNode):
                    if self.static:
                        b.static = True
                    self.funcs[(b.name.value, b.get_arg_types(parent_obj=self))] = b
                elif isinstance(b, ObjectDeclarationNode):
                    self.types[b.name.value] = b
        if len(body) > 0:
            self.end = self.body[-1].end
        return

    def lower(self) -> 'ObjectDeclarationNode':
        if self.lowered:  # we did this already so shortcut
            return self
        if self.parent_class is not None and self.parent_class.value not in top_level.types:
            self.error(E.unknown_obj_type, self.parent_class)

        self.inherit()
        self.body = top_level.check_scope(self.body, context=(self.vars, self.funcs, self.types))

        self.create_destructor()
        self.lowered = True
        return self

    def update_tc_dicts(self) -> None:
        top_level.types[self.name.value] = self
        return

    def inherit(self) -> None:
        if self.parent_class is None:
            return
        parent_class_def = top_level.types[self.parent_class.value]
        vars_copy = parent_class_def.vars.copy()
        static_vars_copy = parent_class_def.static_vars.copy()
        funcs_copy = parent_class_def.funcs.copy()
        types_copy = parent_class_def.types.copy()

        vars_copy.update(self.vars)
        static_vars_copy.update(self.static_vars)
        funcs_copy.update(self.funcs)
        types_copy.update(self.types)

        self.vars = vars_copy
        self.static_vars = static_vars_copy
        self.funcs = funcs_copy
        self.types = types_copy
        return

    def has_attribute(self, name: str, static=False) -> bool:
        if static or self.static:
            return name in self.static_vars or name in self.types or self.contains_func_name(name)
        else:
            return name in self.static_vars or name in self.vars or name in self.types or self.contains_func_name(name)

    def get_attribute_type(self, name, static=False) -> Type:
        if self.contains_func_name(name):
            # return self.funcs[name].type
            return types['func']
        if static and name in self.static_vars:
            return self.static_vars[name].type
        if name in self.vars:
            return self.vars[name].type
        if name in self.types:
            return self.types[name].type

        assert True, False

    def check_generics(self, generics, generic_start, generic_end, generic_line):
        return

    def get_generic_definition(self, name) -> (None, VarDeclarationNode, ValueNode):
        for generic in self.generics:
            if generic.name.value == name:
                return generic
        return None

    def contains_func_name(self, name) -> bool:
        for key in self.funcs:
            if key[0] == name:
                return True
        return False

    def get_func(self, func_name, args, static_func_args=None) -> (FuncDeclarationNode, None):
        if (func_name, args) in self.funcs:
            return self.funcs[(func_name, args)]

        elif static_func_args is not None and (func_name, static_func_args) in self.funcs:
            func = self.funcs[(func_name, static_func_args)]
            if func.static:
                return func
        else:
            return None

    def create_destructor(self):
        func = FuncDeclarationNode(null.value, tok_destructor)
        body = []
        for var in self.vars.values():
            if isinstance(var, VarDeclarationNode) and not var.type.value_type:
                code = dealloc_var(var, self.name)
                if code is not None:
                    body += code
        func.assign_body(body)
        self.destructor = func
        self.funcs[(func.name.value, func.get_arg_types(parent_obj=self))] = func
        return

    def __repr__(self):
        if self.parent_class is None:
            string = f'{self.name} ' + '{'
        else:
            string = f'{self.name}({str(self.parent_class)}) ' + '{'

        for node in self.body:
            string += str(node)
        string += '}>'
        return string


class MacroDefNode(Node):
    def __init__(self, name: Token, expression: Node):
        super().__init__(name.start, expression.end, name.line, name.line_num)
        self.name = name
        self.expression = expression
        return

    def lower(self) -> Node:
        if self.lowered:  # we did this already so shortcut
            return self
        self.expression.assign_parent(self.parent)
        self.expression = self.expression.lower()
        self.type = self.expression.type
        self.lowered = True
        return self

    def update_tc_dicts(self) -> None:
        top_level.vars[self.name.value] = self
        return

    def is_weak(self) -> bool:
        return self.expression.is_weak()

    def __repr__(self):
        return f'{self.name.value} = {self.expression}'


class FuncCallNode(Node):
    def __init__(self, name: (ValueNode, DotOpNode), args: List[Node]):
        super().__init__(name.start, name.end, name.line, name.line_num)
        self.func = name
        self.func_name = None
        self.func_declaration = None
        self.args = args
        return

    def lower(self, func_call=True, expression=False) -> Node:
        if self.lowered:  # we did this already so shortcut
            return self
        self.func.assign_parent(self.parent)
        self.func = self.func.lower(func_call, expression=False)

        for i, arg in enumerate(self.args):
            arg.assign_parent(self.parent)
            self.args[i] = arg.lower()

        static_func_args = None
        if isinstance(self.func, ValueNode):
            if self.func.value.type in default_ops:  # the function name is either a word or the name of the type
                self.func_name = default_ops[self.func.value.type]
            else:
                self.func_name = self.func.value.value
        elif isinstance(self.func, DotOpNode):
            self.func_name = self.func.property.value.value
            if not (isinstance(self.func.object, ValueNode) and self.func.object.value.value in top_level.types):
                static_func_args = tuple(self.args)  # saving a copy of the function's original args in case it's static
                self.args.insert(0, self.func.object)
        else:
            assert True, False

        args = self.get_arg_types()
        self.func_declaration = top_level.get_func(self.func_name, args, static_func_args)

        if self.func_declaration is None:
            self.error(E.undefined_function, self.func)
        self.type = self.func_declaration.type
        self.lowered = True
        return self

    def get_arg_types(self) -> Tuple[Type]:
        args = []
        for arg in self.args:
            args.append(arg.type)
        return tuple(args)

    def __repr__(self):
        return f'{self.func_name}({str(self.args)[1:-1]})'


class IfNode(Node):
    def __init__(self, condition: Node, body: List[Node] = None):
        super().__init__(condition.start, condition.end, condition.line, condition.line_num)
        self.condition = condition
        if body is None:
            self.body = []
        else:
            self.body = body
        return

    def lower(self) -> Node:
        if self.lowered:  # we did this already so shortcut
            return self
        self.condition.assign_parent(self.parent)
        self.condition = self.condition.lower()
        for i, node in enumerate(self.body):
            node.assign_parent(self.parent)
            self.body[i] = node.lower()
        self.lowered = True
        return self

    def update_tc_dicts(self) -> None:
        for node in self.body:
            node.update_tc_dicts()
        return

    def __repr__(self):
        string = f'<if ({self.condition}) ' + '{'
        for node in self.body:
            string += str(node) + ';'
        string += '}>'
        return string


class ElseNode(Node):
    def __init__(self, start, end, line, line_num, body: List[Node] = None):
        super().__init__(start, end, line, line_num)
        if body is None:
            self.body = []
        else:
            self.body = body
        return

    def lower(self) -> Node:
        if self.lowered:  # we did this already so shortcut
            return self
        for i, node in enumerate(self.body):
            node.assign_parent(self.parent)
            self.body[i] = node.lower()
        self.lowered = True
        return self

    def update_tc_dicts(self) -> None:
        for node in self.body:
            node.update_tc_dicts()
        return

    def __repr__(self):
        string = '<else {'
        for node in self.body:
            string += str(node) + ';'
        string += '}>'
        return string


class ForNode(ScopedNode):
    def __init__(self, var: Node, condition: Node, step: Node):
        super().__init__(start=var.start, end=var.end, line=var.line, line_num=var.line_num)
        self.var = var
        self.var.parent = self
        self.condition = condition
        self.condition.parent = self
        self.step = step
        self.step.parent = self
        return

    def lower(self) -> Node:
        if self.lowered:  # we did this already so shortcut
            return self
        self.var = self.var.lower()
        self.condition = self.condition.lower()
        self.step = self.step.lower()
        self.body = top_level.check_scope(self.body, context=(self.vars, self.funcs, self.types))
        self.lowered = True
        return self

    def __repr__(self):
        string = f'<for ({self.var};{self.condition};{self.step}) ' + '{'
        for node in self.body:
            string += str(node) + ';'
        string += '}>'
        return string


class WhileNode(ScopedNode):
    def __init__(self, condition: Node):
        super().__init__(start=condition.start, end=condition.end, line=condition.line, line_num=condition.line_num)
        self.condition = condition
        self.condition.assign_parent(self)
        return

    def lower(self) -> Node:
        if self.lowered:  # we did this already so shortcut
            return self
        self.condition = self.condition.lower()
        self.body = top_level.check_scope(self.body, context=(self.vars, self.funcs, self.types))
        self.lowered = True
        return self

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

    def lower(self) -> Node:
        if self.lowered:  # we did this already so shortcut
            return self
        self.condition.assign_parent(self)
        self.condition = self.condition.lower()
        self.body = top_level.check_scope(self.body, context=(self.vars, self.funcs, self.types))
        self.lowered = True
        return self

    def assign_cnd(self, condition: Node):
        self.condition = condition
        self.condition.parent = self
        self.start = condition.start
        self.line = condition.line
        return

    def __repr__(self):
        string = '<do {'
        for node in self.body:
            string += str(node) + ';'
        string += '}' + f' while ({self.condition})>'
        return string


class SwitchNode(ScopedNode):
    def __init__(self, switch_val: Node):
        super().__init__(start=switch_val.start, end=switch_val.end, line=switch_val.line, line_num=switch_val.line_num)
        self.switch_val = switch_val
        self.switch_val.assign_parent(self)
        return

    def lower(self) -> Node:
        if self.lowered:  # we did this already so shortcut
            return self
        self.switch_val = self.switch_val.lower()
        self.body = top_level.check_scope(self.body, context=(self.vars, self.funcs, self.types))
        self.lowered = True
        return self

    def __repr__(self):
        string = f'<switch ({self.switch_val}) ' + '{'
        for node in self.body:
            string += str(node) + ';'
        string += '}>'
        return string


class KeywordNode(Node):
    def __init__(self, keyword: Token):
        super().__init__(keyword.start, keyword.end, keyword.line, keyword.line_num)
        self.keyword = keyword
        return

    def lower(self) -> Node:
        if self.keyword.type == TT.exit_:
            self.deallocate_vars()
            self.lowered = True
        return self

    def deallocate_vars(self):
        extra_code = []
        if isinstance(self.parent, ScopedNode):
            for var in self.parent.vars.values():
                code = dealloc_var(top_level, var)
                if code is not None:
                    extra_code += code
            self.add_extra_nodes(extra_code, before=False)
        return

    def __repr__(self):
        return f'{self.keyword}'


class SingleExpressionNode(Node):
    def __init__(self, value, word: Token):
        super().__init__(word.start, value.end, value.line, value.line_num)
        self.val = value
        self.word = word.type
        return

    def __repr__(self):
        return f'<{self.word.value} {self.val}>'


class CaseNode(SingleExpressionNode):
    def __init__(self, value: Node, word):
        SingleExpressionNode.__init__(self, value, word)
        return

    def lower(self) -> Node:
        if self.lowered:  # we did this already so shortcut
            return self
        self.val = self.val.lower()
        return self


class ReturnNode(SingleExpressionNode):
    def __init__(self, value: Node, word: Token):
        SingleExpressionNode.__init__(self, value, word)
        return

    def lower(self) -> Node:
        if self.lowered:  # we did this already so shortcut
            return self
        self.val.assign_parent(self.parent)
        self.val = self.val.lower()
        ret_type = self.get_current_func().type
        if ret_type != self.val.type and self.val.type != null.type:  # im accepting null as any type
            self.error(E.type_missmatch, self.val, ret_type.type, self.val.type)

        self.deallocate_vars()
        self.add_extra_nodes([UrclNode(tok_urcl, [token(TT.string, 'IMM R2 0')])], before=False)
        return self

    def get_current_func(self) -> FuncDeclarationNode:
        parent = self.parent
        while parent is not None:
            if isinstance(parent, FuncDeclarationNode):
                return parent
            parent = parent.parent
        self.error(E.return_not_in_func, self)

    def deallocate_vars(self):
        extra_code = []
        parent = self.parent
        while parent is not None:
            if isinstance(parent, ScopedNode):
                for var in parent.vars.values():
                    code = dealloc_var(top_level, var)
                    if code is not None:
                        extra_code += code
                if isinstance(parent, FuncDeclarationNode):
                    self.add_extra_nodes(extra_code, before=False)
                    return
            parent = parent.parent
        return


class YeetNode(Node):
    def __init__(self, word: Token, class_name: Token):
        super().__init__(word.start, class_name.end, word.line, word.line_num)
        self.word = word
        self.error_name = class_name
        return

    def lower(self) -> 'Node':
        if self.lowered:  # we did this already so shortcut
            return self
        if self.error_name.value in top_level.vars:
            self.deallocate_vars()
            self.lowered = True
            return self
        else:
            self.error(E.error_not_found, self.error_name, self.error_name.value)

    def deallocate_vars(self):
        extra_code = []
        parent = self.parent
        while parent is not None:
            if isinstance(parent, ScopedNode):
                for var in parent.vars.values():
                    code = dealloc_var(top_level, var)
                    if code is not None:
                        extra_code += code
                if isinstance(parent, FuncDeclarationNode):
                    self.add_extra_nodes(extra_code, before=False)
                    return
            parent = parent.parent
        return

    def __repr__(self):
        return f'yeet {self.error_name.value}'


class UrclNode(Node):
    def __init__(self, word: Token, block: List[Token]):
        super().__init__(word.start, word.end, word.line, word.line_num)
        self.type = TT.urcl
        self.body = block

    def __repr__(self):
        string = ''
        for tok in self.body:
            string += tok.value + '\n'
        return string


#########################################
# Value Objects
#########################################

top_level = ScopedNode()

true = ValueNode(token(TT.true))
true.type = Type
false = ValueNode(token(TT.false))
false.type = types['bool']
null = ValueNode(token(TT.null))

zero = ValueNode(token(TT.num))
zero.type = types['uint']

this = ValueNode(token(TT.this))
super_ = ValueNode(token(TT.super))

default_values = {
    true.value.type.value: true,
    false.value.type.value: false,
    null.value.type.value: null,
    this.value.type.value: this,
    super_.value.type.value: super_,
}

#########################################
# ARC
#########################################

tok_equ = token(TT.equ)
tok_ret = token(TT.return_)
tok_urcl = token(TT.urcl)
tok_destructor = token(TT.word, '__del__')
tok_dealloc_func = token(TT.word, '__free__')
tok_dot = token(TT.dot)
tok_inc = token(TT.inc)
tok_dec = token(TT.dec)

null_pointer = token(TT.word, 'NullPointer')


def null_ref_check(ref: ValueNode) -> Node:
    body = YeetNode(tok_ret, null_pointer)
    body.parent = ref.parent
    null_check = IfNode(BinOpNode(tok_equ, ref, null), [body])
    return null_check


def ref_check(ref: (ValueNode, DotOpNode)) -> List[Node]:
    output = []

    str_ref_count = BinOpNode(token(TT.address), ref, ValueNode(token(TT.num, -2)))
    str_ref_count.lowered = True  # we don't want top_level to check these hacks
    str_ref_count.type = types['int']

    weak_count = BinOpNode(token(TT.address), ref, ValueNode(token(TT.num, -1)))
    weak_count.lowered = True  # we don't want top_level to check these hacks
    weak_count.type = types['int']

    destructor_call = FuncCallNode(DotOpNode(tok_dot, ref, ValueNode(tok_destructor)), [])
    # self will be filled by top_level

    if ref.is_weak():
        output.append(UnOpNode(tok_dec, weak_count))
        # if both weak and strong ref counts are 0 then object can be safely deallocated
        output.append(IfNode(BinOpNode(tok_equ, str_ref_count, zero),
                             [IfNode(BinOpNode(tok_equ, weak_count, zero),
                                     [FuncCallNode(ValueNode(tok_dealloc_func), [ref])])]))
    else:
        output.append(UnOpNode(tok_dec, str_ref_count))
        output.append(IfNode(BinOpNode(tok_equ, str_ref_count, zero),
                             [destructor_call, IfNode(BinOpNode(tok_equ, weak_count, zero),
                                                      [FuncCallNode(ValueNode(tok_dealloc_func), [ref])])]))
    return output


def alloc_var():
    return


def dealloc_var(var: (VarDeclarationNode, ValueNode), obj_name=None):
    if var.type in top_level.types and not var.type.value_type:
        if obj_name is None:
            if isinstance(var, VarDeclarationNode):  # we must transform this node into a ValueNode first
                return ref_check(ValueNode(var.name))
            else:
                return ref_check(var)
        else:
            # we know that var is VarDeclarationNode at this point
            node = DotOpNode(tok_dot, ValueNode(obj_name), ValueNode(var.name))  # must change the var to object.field
            return ref_check(node)
    else:
        return  # we simply ignore. if it is in a func scope we also increase the SP later


#########################################
# Parser
#########################################


class Parser:
    def __init__(self, toks: List[Token], f_name=file_name):
        self.toks = toks
        self.len = len(self.toks)
        self.i = 0
        self.last = None
        self.toks.append(token(TT.eof))
        self.peak = self.toks[self.i]
        self.file_name = f_name
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
        if tt == TT.static:
            self.advance()
            tt = self.peak.type
            if tt == TT.static:
                self.error(E.identifier_expected, self.peak)

        if self.peak.type == TT.lcbr:
            self.advance()
            statement = self.parse_body(TT.rcbr)
            self.advance()
            return statement

        if tt == TT.func:
            return [self.func_def()]

        elif tt == TT.class_:
            return [self.make_class()]

        elif tt == TT.macro:
            return [self.make_macro()]

        elif tt == TT.weak:
            return [self.assign_var()]

        elif tt == TT.word:  # can be a custom type, a function call
            self.advance()
            if self.peak.type == TT.word or self.peak.type == TT.lt:
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

        elif tt == TT.while_:
            return [self.make_while()]

        elif tt == TT.do_:
            return [self.make_do_while()]

        elif tt == TT.return_:
            return [self.make_return()]

        elif tt == TT.urcl:
            return [self.make_urcl()]

        elif tt == TT.lcbr:
            return [self.make_scope()]
        else:
            return [self.make_expression()]

    def make_expression(self) -> Node:
        toks = self.shunting_yard()
        node = self.make_ast(toks)
        if self.peak.type in {TT.comma, TT.colon, TT.semi_col}:  # TT.rpa is not there because it should not consume it
            self.advance()
        return node

    def shunting_yard(self) -> List:
        queue: List = []
        stack = []
        start_index = self.i
        while self.has_next() and self.peak.type not in {TT.comma, TT.colon, TT.semi_col, TT.rcbr, TT.lcbr, TT.eof}:
            t = self.peak.type
            if t == TT.lpa:
                if self.last.type == TT.word:
                    while len(stack) > 0 and (op_table[TT.func_call][0] > op_table[stack[-1].type][0] or
                                              (op_table[TT.func_call][0] == op_table[stack[-1].type][0] and
                                               op_table[TT.func_call][1])):
                        queue.append(stack.pop())
                    queue.append(self.make_func_call())
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
                while len(stack) > 0 and stack[-1].type != TT.lpa:
                    queue.append(stack.pop())
                if len(stack) > 0:
                    stack.pop()
                    self.advance()
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
            continue

        while len(stack) > 0:
            queue.append(stack.pop())
        return queue

    def check_if_expression_over(self, start_index):
        if self.peak.value in keywords and self.peak.value not in default_values:
            return True

        i = self.i - 1
        while i >= start_index and self.toks[i].type in {TT.lpa, TT.rpa, TT.rbr, TT.lbr}:
            i -= 1

        if i >= start_index:
            if self.peak.type == TT.word:  # current rules
                if self.toks[i].type in op_table and self.toks[i].type in unary_ops and op_table[self.toks[i].type][1]:
                    return True
                if self.toks[i].type not in op_table:
                    return True
                if self.toks[i].type == TT.word:
                    return True
        return False

    def make_ast(self, queue: List[Token]) -> Node:
        stack: List[Node] = []
        for tok in queue:
            if isinstance(tok, Node):
                stack.append(tok)
            elif isinstance(tok, list) and len(stack) >= 1:
                stack.append(FuncCallNode(stack.pop(), tok))

            elif tok.type in op_table:
                if tok.type in unary_ops:
                    if len(stack) >= 1:
                        node_a = stack.pop()
                        stack.append(UnOpNode(tok, node_a))
                    else:
                        self.error(E.expression_expected, tok)
                    continue

                if len(stack) >= 2:
                    node_b = stack.pop()
                    node_a = stack.pop()
                else:
                    self.error(E.expression_expected, tok)
                    raise  # this is so python type checker chills a bit

                if tok.type in assign_ops:
                    if tok.type == TT.assign:
                        stack.append(AssignNode(node_a, node_b))
                    else:
                        op_tok = Token(assign_ops[tok.type], tok.start, tok.end, tok.line, tok.line_num)
                        stack.append(AssignNode(node_a, BinOpNode(op_tok, node_a, node_b)))
                else:
                    if tok.type == TT.dot and isinstance(node_b, ValueNode):
                        stack.append(DotOpNode(tok, node_a, node_b))
                    else:
                        stack.append(BinOpNode(tok, node_a, node_b))
            else:
                stack.append(ValueNode(tok))
            continue
        if len(stack) > 0:
            return stack.pop()
        else:
            self.error(E.expression_expected, self.peak)

    # processing keywords

    def assign_var(self, func_def=False) -> (VarDeclarationNode, Node):
        static = False
        if self.last is not None and self.last.type == TT.static:
            static = True
        is_weak = None
        if self.peak.type == TT.weak:
            is_weak = self.peak
            self.advance()
        var_type = self.peak
        self.advance()
        if self.peak.type == TT.lt:
            pass  # generics later

        if self.peak.type != TT.word:
            if not func_def and self.peak.type == TT.dot:
                self.advance(-1)
                return self.make_expression()
            else:
                self.error(E.identifier_expected, self.peak)

        name = self.peak
        expression = self.make_expression()
        if isinstance(expression, AssignNode):
            declair_node = VarDeclarationNode(var_type, name, expression, weak=is_weak, static=static)
        else:
            declair_node = VarDeclarationNode(var_type, name, weak=is_weak, static=static)
        return declair_node

    def func_def(self) -> Node:
        static = False
        if self.last is not None and self.last.type == TT.static:
            static = True
        self.advance()
        ret_type = self.peak
        if ret_type.type != TT.word and ret_type.type != TT.null:
            self.error(E.invalid_ret_type, self.peak)

        self.advance()
        if self.peak.type == TT.assign:  # it was a variable of type function instead
            self.advance(-2)
            return self.assign_var()
        if self.peak.type == TT.lt:
            pass  # generics later

        if self.peak.type == TT.lpa:  # then it's a constructor
            static = True  # constructors should be static even tho the user doesn't need to say it
            func_name = ret_type
        elif self.peak.type != TT.word:
            self.error(E.identifier_expected, self.peak)
            raise  # just so python typechecker chills a bit
        else:
            func_name = self.peak
            self.advance()

        if self.peak.type != TT.lpa:
            self.error(E.symbol_expected, self.peak, '(')

        self.advance()
        args = []
        while self.has_next() and self.peak.type != TT.rpa:
            args.append(self.assign_var(func_def=True))

        self.advance()
        node = FuncDeclarationNode(ret_type, func_name, args, static)
        self.scope.append(node)
        body = self.next_statement()
        self.scope.pop()
        node.assign_body(body)
        return node

    def make_class(self) -> Node:
        static = False
        if self.last is not None and self.last.type == TT.static:
            static = True
        self.advance()
        if self.peak.type != TT.word:
            self.error(E.identifier_expected, self.peak)
        name = self.peak
        self.advance()
        generics = []
        if self.peak.type == TT.lt:
            pass  # generics later

        parent_class = None
        if self.peak.type == TT.lpa:
            self.advance()
            if self.peak.type != TT.word:
                self.error(E.identifier_expected, self.peak)
            parent_class = self.peak
            self.advance()
            if self.peak.type != TT.rpa:
                self.error(E.symbol_expected, self.peak, TT.rpa.value)
            else:
                self.advance()

        node = ObjectDeclarationNode(name, generics, parent_class, static=static)
        self.scope.append(node)
        body = self.next_statement()
        self.scope.pop()
        node.assign_body(body)
        return node

    def make_if(self) -> List[Node]:
        self.advance()
        condition = self.make_expression()
        body = self.next_statement()
        if self.peak.type == TT.elif_:
            return [IfNode(condition, body), self.make_elif()]
        elif self.peak.type == TT.else_:
            return [IfNode(condition, body), self.make_else()]
        return [IfNode(condition, body)]

    def make_elif(self) -> List[Node]:
        self.advance()
        cnd = self.make_expression()
        body = self.next_statement()
        if self.peak.type == TT.elif_:
            return [ElseNode(cnd.start, cnd.end, cnd.line, cnd.line_num, [IfNode(cnd, body)]), self.make_elif()]
        elif self.peak.type == TT.else_:
            return [ElseNode(cnd.start, cnd.end, cnd.line, cnd.line_num, [IfNode(cnd, body)]), self.make_else()]
        return [ElseNode(cnd.start, cnd.end, cnd.line, cnd.line_num, [IfNode(cnd, body)])]

    def make_else(self) -> ElseNode:
        start = self.peak.start
        self.advance()
        body = self.next_statement()
        return ElseNode(start, self.peak.end, self.peak.line, self.peak.line_num, body)

    def make_switch(self) -> SwitchNode:
        self.advance()
        switch_val = self.make_expression()
        node = SwitchNode(switch_val)

        self.scope.append(node)
        body = self.next_statement()
        self.scope.pop()

        node.assign_body(body)
        return node

    def make_case(self) -> CaseNode:
        word = self.peak
        self.advance()
        case = self.make_expression()
        return CaseNode(case, word)

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
        body = self.next_statement()
        self.scope.pop()

        node.assign_body(body)
        return node

    def make_while(self) -> WhileNode:
        self.advance()
        condition = self.make_expression()
        node = WhileNode(condition)

        self.scope.append(node)
        body = self.next_statement()
        self.scope.pop()

        node.assign_body(body)
        return node

    def make_do_while(self) -> Node:
        self.advance()
        node = DoWhileNode()

        self.scope.append(node)
        body = self.next_statement()
        self.scope.pop()

        node.assign_body(body)

        if self.has_next() and self.peak.type != TT.while_:
            self.error(E.while_expected, self.peak)

        condition = self.make_expression()
        node.assign_cnd(condition)
        return node

    def make_return(self) -> ReturnNode:
        word = self.peak
        self.advance()
        if self.peak.type in {TT.comma, TT.colon, TT.semi_col}:
            self.advance()
            return ReturnNode(null, word)
        else:
            expression = self.make_expression()
            return ReturnNode(expression, word)

    def make_urcl(self) -> UrclNode:
        word = self.peak
        self.advance()
        if self.peak.type != TT.lcbr:
            self.error(E.symbol_expected, self.peak, '{')

        self.advance()
        body = []
        while self.has_next() and self.peak.type != TT.rcbr:
            if self.peak.type != TT.string:
                self.error(E.string_expected, self.peak)
            else:
                body.append(self.peak)
            self.advance()

        self.advance()
        return UrclNode(word, body)

    def make_macro(self) -> MacroDefNode:
        self.advance()
        if self.peak.type != TT.word:
            self.error(E.identifier_expected, self.peak)

        name = self.peak
        expression = self.make_expression()
        if isinstance(expression, AssignNode):
            node = MacroDefNode(name, expression.value)
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

        return ValueNode(Token(TT.array, start_tok.start, self.peak.end, start_tok.line, start_tok.line_num, elements))

    def make_func_call(self) -> List[Node]:
        self.advance()
        args = []
        while self.has_next() and self.peak.type != TT.rpa:
            args.append(self.make_expression())

        return args

    def make_scope(self):
        start = self.peak
        nodes = self.next_statement()
        node = ScopedNode(start.start, self.last.end, start.line, start.line_num)
        node.assign_body(nodes)
        return node

    def error(self, err: E, tok: Token, *args) -> None:
        raise ErrorException(err, tok, self.file_name, args)

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
    def __init__(self, trees: List[Node]):
        self.file_name = file_name
        self.asts = trees
        self.funcs: Dict[Tuple[str, Tuple[Type]], FuncDeclarationNode] = {}
        self.vars: Dict[str, (VarDeclarationNode, MacroDefNode)] = {}
        self.defined_vars: Set[VarDeclarationNode] = set()
        self.types: Dict[str, ObjectDeclarationNode] = {}

        self.errors = []
        return

    def error(self, e: E, node, *args) -> None:
        self.errors.append(
            Error(e, node.start, node.end, node.line_num, node.line, file_name, *args))


if __name__ == "__main__":
    main()
