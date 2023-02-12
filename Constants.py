from enum import Enum


# global variables

class Globals:
    """Class used to be able to set dependable variables from external modules"""
    def __init__(self):
        self.FILE_NAME = 'IDE'
        self.PROGRAM_LINES = []


global_vars = Globals()


# constants
SEPARATORS = {',', ':', ';', '{', '}', '[', ']', '(', ')'}
"""Set of characters accepted as separators"""

END_OF_EXPRESSION = {',', ';', ':'}
"""Set of valid characters that end an expression"""

KEYWORDS = {'if', 'else', 'break', 'skip', 'while', 'return', 'class', 'fn', 'var', 'macro'}
"""Set containing all the language's keywords"""

BOOLEANS = ['false', 'true']
"""List of the 2 boolean values"""

SYMBOLS = {'&', '|', '+', '-', '*', '/', '=', '<', '>', '!', '(', ')', '{', '}', '[', ']', '.', ',', ':', ';'}
"""Set of characters accepted as valid symbols"""

UNARY_OPERATORS = {'!', '-', '~'}
"""Set of Characters accepted as valid values for unary operators"""

ASSIGN_OPERATORS = {'=', '+=', '-=', '*=', '/=', '%=', '<<=', '>>=', '|=', '&=', '^='}
"""Set of strings containing all the valid assign&operate operators"""

operator_precedence = {
    '(': 0,
    '[]': 0,
    '=': 1,
    '||': 2,
    '&&': 3,
    '!': 4,
    '==': 5,
    '!=': 5,
    '<=': 5,
    '>=': 5,
    '>': 5,
    '<': 5,
    '|': 6,
    '^': 7,
    '&': 8,
    '>>': 9,
    '<<': 9,
    '+': 10,
    '-': 10,
    '*': 11,
    '/': 11,
    '%': 11,
}
"""Dict mapping the operators and their precedence (all operators are left associative)"""


class TT(Enum):
    """Basic Token types"""
    SEPARATOR = 0
    OPERATOR = 1
    KEYWORD = 2
    IDENTIFIER = 3
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

    and_ = Token(TT.OPERATOR, '&&')
    or_ = Token(TT.OPERATOR, '||')
    not_ = Token(TT.OPERATOR, '!')

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
    dot = Token(TT.OPERATOR, '.')

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


class Keywords(Enum):
    if_ = Token(TT.KEYWORD, 'if')
    else_ = Token(TT.KEYWORD, 'else')
    break_ = Token(TT.KEYWORD, 'break')
    skip = Token(TT.KEYWORD, 'skip')
    while_ = Token(TT.KEYWORD, 'while')
    return_ = Token(TT.KEYWORD, 'return')
    class_ = Token(TT.KEYWORD, 'class')
    func = Token(TT.KEYWORD, 'func')
    var = Token(TT.KEYWORD, 'var')
    macro = Token(TT.KEYWORD, 'macro')


class E(Enum):
    duplicate_class = 'Duplicate class name'
    duplicate_func = 'Duplicate function declaration'
    duplicate_var = 'Duplicate variable name'
    duplicate_macro = 'Duplicate macro'
    literal_expected = 'Literal expected'
    string_expected = 'String expected'
    invalid_literal = 'Invalid literal'
    invalid_ret_type = 'Invalid return type'
    while_expected = 'while keyword expected'
    unexpected_argument = 'Unexpected argument'
    break_not_in_loop = 'break keyword found outside a loop body'
    skip_not_in_loop = 'skip keyword found outside a loop body'
    return_not_in_func = 'return keyword found outside a function body'
    unknown_var_type = 'Unknown variable type'
    undefined_variable = 'Undefined variable'
    var_before_assign = 'Variable might be used before assignment'
    unknown_obj_type = 'Unknown object type'
    unknown_identifier = "No variable, class or function named '{}' is visible in scope"
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
    def __init__(self, error: Enum, start, end, line, code_line, *args):
        self.e: Enum = error
        self.start = start
        self.end = end
        self.line = line
        self.code_line = code_line
        self.file_name = global_vars.FILE_NAME

        self.args = args
        return

    def __repr__(self):
        string = f'{self.file_name}:{self.start}:{self.line}: {self.e.value.format(*self.args)}\n'
        string += self.code_line + '\n'
        string += ' ' * (self.start - 1)
        string += '^' * (self.end - self.start + 1)
        return string
