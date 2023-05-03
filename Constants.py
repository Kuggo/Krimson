from enum import Enum
from typing import Optional


# global variables

class Globals:
    """Class used to be able to set dependable variables from external modules"""
    def __init__(self):
        self.FILE_NAME: str = 'IDE'
        """Name of the input file containing krimson code"""

        self.PROGRAM_LINES: list[str] = []
        """list containing the input program split between new line separators ``\\n``"""
        return


global_vars = Globals()
"""Object that holds the variables related to the input code (to be configurable from other modules when compiling)"""


# constants
SEPARATORS = {',', ':', ';', '{', '}', '[', ']', '(', ')'}
"""Set of characters accepted as separators"""

END_OF_EXPRESSION = {',', ';', ':'}
"""Set of valid characters that end an expression"""

KEYWORDS = {'if', 'else', 'break', 'skip', 'while', 'return', 'type', 'macro', 'fn'}
"""Set containing all the language's keywords"""

BOOLEANS = ['false', 'true']
"""List of the 2 boolean values"""

NULL = 'null'
"""Text representation of Null value"""

SYMBOLS = {'&', '|', '+', '-', '*', '/', '=', '<', '>', '!', '(', ')', '{', '}', '[', ']', '.', ',', ':', ';'}
"""Set of characters accepted as valid symbols"""

UNARY_OPERATORS = {'!', '-', '~'}
"""Set of Characters accepted as valid values for unary operators"""

ASSIGN_OPERATORS = {'=', '+=', '-=', '*=', '/=', '%=', '<<=', '>>=', '|=', '&=', '^='}
"""Set of strings containing all the valid assign&operate operators"""

OPERATOR_PRECENDENCE = {
    '=': -1,
    '(': 0,
    '[]': 0,
    '()': 0,
    '.': 0,
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
    '- ': 10,
    '*': 11,
    '/': 11,
    '%': 11,
    '-': 12,
    '~': 12,
}
"""Dict mapping the operators and their precedence"""

RIGHT_ASSOCIATIVE_OPERATORS = {'='}
"""Set containing the operators that are right associative"""


class TT(Enum):
    """Basic Token types"""
    SEPARATOR = 0
    OPERATOR = 1
    KEYWORD = 2
    IDENTIFIER = 3
    LITERAL = 4


class Token:
    """Gathers all the information regarding a token object (location on source code), type and value"""
    def __init__(self, token_type, value, start=-1, end=-1, line=-1):
        self.tt: TT = token_type
        """Token type"""

        self.value = value
        """Value of token"""

        self.start: int  = start
        """index of the first character of the token on the source code"""

        self.end: int = end
        """index of last character of the token on the source code"""

        self.line: int = line
        """index of the line of the chars of the token on the source code"""
        return

    def __eq__(self, other):
        return other is not None and self.tt == other.tt and self.value == other.value

    def __repr__(self):
        return f'<{self.line}:{self.start}:{self.end}: {self.tt.name}, {self.value}>'


class Literal(Token):
    """Special Case of Token, that has a token type (tt) of ``TT.LITERAL``.

    It contains an extra field ``self.literal_type`` for the krimson type of the literal value"""
    def __init__(self, value, t: 'Type', start=-1, end=-1, line=-1):
        super().__init__(TT.LITERAL, value, start, end, line)
        self.literal_type: Type = t
        """krimson Type of the Literal value of the token"""
        return


class Operators(Enum):
    """Enum of all Operators"""
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

    fn = Token(TT.OPERATOR, '->')
    func = Token(TT.OPERATOR, '()')
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
    """Enum of all separators"""
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
    """Enum of all Keywords"""
    if_ = Token(TT.KEYWORD, 'if')
    else_ = Token(TT.KEYWORD, 'else')
    break_ = Token(TT.KEYWORD, 'break')
    skip = Token(TT.KEYWORD, 'skip')
    while_ = Token(TT.KEYWORD, 'while')
    return_ = Token(TT.KEYWORD, 'return')
    fn = Token(TT.KEYWORD, 'fn')
    type = Token(TT.KEYWORD, 'type')
    macro = Token(TT.KEYWORD, 'macro')


class Error(Exception):
    """Gathers all information about an error that occurred with compiling krimson code, such as the characters in the
    file where it was detected and any additional information about it.

    It gets formatted in a user-friendly way"""
    def __init__(self, error: Enum, start: int, end: int, line: int, code_line: str, *args: str):
        self.e: Enum = error
        """Error enum (LexicalError | SyntaxError | TypeError) containing the message describing the error"""

        self.start: int = start
        """index of the first character where the error was detected on the source code"""

        self.end: int = end
        """index of last character where the error was detected on the source code"""

        self.line: int = line
        """index of the line of the chars where the error was detected on the source code"""

        self.code_line: str = code_line
        """Line containing the input krimson code where the error occurred"""

        self.file_name: str = global_vars.FILE_NAME
        """Name of the input file the error occurred"""

        self.args: tuple[str, ...] = args
        """Extra arguments needed for a better custom error message"""
        return

    def __repr__(self):
        string = f'{self.file_name}:{self.start}:{self.line}: {self.e.value.format(*self.args)}\n'
        string += self.code_line + '\n'
        string += ' ' * (self.start - 1)
        string += '^' * (self.end - self.start + 1)
        return string


# Type

class Type:
    """Class containing all information regarding a type in krimson, such as the name of the type, and extra generic
    types associated with it"""
    def __init__(self, name: Token):
        self.name: Token = name
        """name of the krimson type"""

        self.size = 1
        """compile-time size of an object of this type"""
        return

    def __eq__(self, other: 'Type'):
        return other is not None and self.name == other.name

    def __hash__(self):
        return self.name.value.__hash__()

    def is_subtype(self, super_type: 'Type') -> bool:
        """
        checks if ``super_type`` is a subtype of ``self``. In other words, if ``super_type`` can be used when ``self``
        type is required
        :param super_type: the type to be checking
        :return: true if ``super_type`` is a subtype of ``self``
        """
        if self.name == super_type.name:
            return True

        return True

    def get_type_label(self) -> str:
        """
        Generates and returns a string containing a unique label to represent the krimson type.
        :return: a string label of the type
        """
        return self.name.value

    def __repr__(self):
        return f'{self.name.value}'


class TupleType(Type):
    def __init__(self, types: list[Type]):
        super().__init__(Token(TT.IDENTIFIER, 'tuple'))
        self.types: list[Type] = types
        return

    def __eq__(self, other: 'TupleType'):
        return isinstance(other, TupleType) and self.types == other.types

    def get_type_label(self) -> str:
        return f'tuple_{"_".join([t.get_type_label() for t in self.types])}'

    def __repr__(self):
        return f'{self.name.value}({", ".join([str(t) for t in self.types])})'


class FunctionType(Type):
    def __init__(self, arg: Type, ret: Type):
        super().__init__(Token(TT.IDENTIFIER, 'fn'))
        self.args: Type = arg
        self.ret: Type = ret
        return

    def __eq__(self, other: 'FunctionType'):
        return isinstance(other, FunctionType) and self.args == other.args and self.ret == other.ret

    def get_type_label(self) -> str:
        return f'func_{self.args.get_type_label()}_to_{self.ret.get_type_label()}'

    def __repr__(self):
        return f'{self.name.value}({self.args} -> {self.ret})'


class ArrayType(Type):
    def __init__(self, arr_type: Type):
        super().__init__(Token(TT.IDENTIFIER, 'array'))
        self.arr_type: Type = arr_type
        return

    def __eq__(self, other: 'ArrayType'):
        return isinstance(other, ArrayType) and self.arr_type == other.arr_type

    def get_type_label(self) -> str:
        return f'array_{self.arr_type.get_type_label()}'

    def __repr__(self):
        return f'{self.name.value}[{self.arr_type}]'


class TypeDefType(Type):
    def __init__(self, type_name: Token, fields: list[Type]):
        super().__init__(type_name)
        self.fields: list[Type] = fields
        return

    def __eq__(self, other: 'TypeDefType'):
        return isinstance(other, TypeDefType) and self.name == other.name and self.fields == other.fields

    def get_type_label(self) -> str:
        return f'{self.name.value}_{"_".join([f.get_type_label() for f in self.fields])}'

    def __repr__(self):
        return f'{self.name.value}({", ".join([str(f) for f in self.fields])})'


class Types(Enum):
    """Enum containing all primitive types the compiler may need to use at compile-time"""

    type = Type(Token(TT.IDENTIFIER, 'type'))
    macro = Type(Token(TT.IDENTIFIER, 'macro'))
    fn = Type(Token(TT.IDENTIFIER, 'fn'))
    null = Type(Token(TT.IDENTIFIER, 'null'))
    bool = Type(Token(TT.IDENTIFIER, 'bool'))
    nat = Type(Token(TT.IDENTIFIER, 'nat'))
    int = Type(Token(TT.IDENTIFIER, 'int'))
    frac = Type(Token(TT.IDENTIFIER, 'frac'))
    char = Type(Token(TT.IDENTIFIER, 'char'))
    array = Type(Token(TT.IDENTIFIER, 'array'))
    str = Type(Token(TT.IDENTIFIER, 'str'))
    dict = Type(Token(TT.IDENTIFIER, 'dict'))
    set = Type(Token(TT.IDENTIFIER, 'set'))
