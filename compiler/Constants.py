from enum import Enum
from typing import Optional


# global variables

class Globals:
    """Class used to be able to set dependable variables from external modules"""
    def __init__(self, file_name = 'IDE', program_lines = None):
        self.FILE_NAME: str = file_name
        """Name of the input file containing krimson code"""

        self.PROGRAM_LINES: list[str] = program_lines if program_lines is not None else []
        """list containing the input program split between new line separators ``\\n``"""
        return


# constants
FILE_EXTENSION = '.krim'
"""Names terminated with this string are considered to be krimson source code files"""

END_OF_EXPRESSION = {';'}
"""Set of valid characters that end an expression"""

KEYWORDS = {'if', 'else', 'exit', 'skip', 'while', 'match'}
"""Set containing all the language's keywords"""

BOOLEANS = ['false', 'true']
"""List of the 2 boolean values"""

SYMBOLS = {'&', '|', '^', '+', '-', '*', '/', '=', '<', '>', '!', '(', ')', '{', '}', '[', ']', '.', ',', ':', ';', '?'}
"""Set of characters accepted as valid symbols"""

UNARY_OPERATORS = {'!', '-'}
"""Set of Characters accepted as valid values for unary operators"""

ASSIGN_OPERATORS = {'=', '+=', '-=', '*=', '/=', '%=', '<<=', '>>=', '|=', '&=', '^='}
"""Dict mapping all the valid assign&operate operators to their operator"""

OPERATOR_PRECENDENCE = {
    # these must end all expressions
    ';': -1,
    ']': -1,
    ')': -1,
    '}': -1,
    # valid operators
    ',': 0,
    '=': 1,
    ':': 2,
    '->': 3,
    '|': 4,
    '^': 5,
    '&': 6,
    '==': 7,
    '!=': 7,
    '<=': 7,
    '>=': 7,
    '>': 7,
    '<': 7,
    '>>': 11,
    '<<': 11,
    '+': 12,
    '- ': 12,
    '*': 13,
    '/': 13,
    '%': 13,
    '-': 14,
    '~': 14,
    '[': 15,
    '(': 15,
    '.': 15,
}
"""Dict mapping the operators and their precedence"""

TYPE_OP_PRECEDENCE = {
# these must end all expressions
    ';': -1,
    ']': -1,
    ')': -1,
    '}': -1,
    # valid operators
    ',': 0,
    ':': 2,
    '->': 3,
    # '|': 8,
    '?': 15,

}
"""Dict mapping the type operators and their precedence"""

RIGHT_ASSOCIATIVE_OPERATORS = {'=', '->'}
"""Set containing the operators that are right associative"""


class TT(Enum):
    """Basic Token types"""
    SEPARATOR = 0
    OPERATOR = 1
    KEYWORD = 2
    IDENTIFIER = 3
    LITERAL = 4


class FileRange:
    """Gathers all the information regarding a range of characters on the source code"""
    def __init__(self, start, line_start, end, line_end):
        self.start: int = start
        """index of the first character of the range on the source code (starts at 1)"""

        self.end: int = end
        """index of last character of the range on the source code (inclusive) (starts at 1)"""

        self.line_start: int = line_start
        """index of the line of the first character of the range on the source code. (starts at 1)"""

        self.line_end: int = line_end
        """index of the line of the last character of the range on the source code. (starts at 1)"""
        return

    def __sub__(self, other: 'FileRange'):
        if self.line_start < other.line_start:
            line_start = self.line_start
            start = self.start
        elif self.line_start > other.line_start:
            line_start = other.line_start
            start = other.start
        else:
            line_start = self.line_start
            start = min(self.start, other.start)

        if self.line_end > other.line_end:
            line_end = self.line_end
            end = self.end
        elif self.line_end < other.line_end:
            line_end = other.line_end
            end = other.end
        else:
            line_end = self.line_end
            end = max(self.end, other.end)

        return FileRange(start, line_start, end, line_end)


class Token:
    """Gathers all the information regarding a token object (location on source code), type and value"""
    def __init__(self, token_type, value, location = FileRange(-1, -1, -1, -1)):
        self.tt: TT = token_type
        """Token type"""

        self.value = value
        """Value of token"""

        self.location: FileRange = location
        """Location of token on source code"""
        return

    def __eq__(self, other):
        return other is not None and self.tt == other.tt and self.value == other.value

    def __repr__(self):
        return f'<{self.location.line_start}:{self.location.start}:{self.location.end}: {self.tt.name}, {self.value}>'


class Operators(Enum):
    """Enum of all Operators"""
    dif = Token(TT.OPERATOR, '!=')
    equ = Token(TT.OPERATOR, '==')
    gt = Token(TT.OPERATOR, '>')
    lt = Token(TT.OPERATOR, '<')
    gte = Token(TT.OPERATOR, '>=')
    lte = Token(TT.OPERATOR, '<=')

    shl = Token(TT.OPERATOR, '<<')
    shr = Token(TT.OPERATOR, '>>')
    and_ = Token(TT.OPERATOR, '&')
    or_ = Token(TT.OPERATOR, '|')
    xor = Token(TT.OPERATOR, '^')
    not_ = Token(TT.OPERATOR, '~')

    add = Token(TT.OPERATOR, '+')
    sub = Token(TT.OPERATOR, '- ')
    neg = Token(TT.OPERATOR, '-')
    mlt = Token(TT.OPERATOR, '*')
    div = Token(TT.OPERATOR, '/')
    mod = Token(TT.OPERATOR, '%')

    fn = Token(TT.OPERATOR, '->')
    opt = Token(TT.OPERATOR, '?')
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
    exit_ = Token(TT.KEYWORD, 'exit')
    skip = Token(TT.KEYWORD, 'skip')
    while_ = Token(TT.KEYWORD, 'while')
    fn = Token(TT.KEYWORD, 'fn')
    type = Token(TT.KEYWORD, 'type')
    match = Token(TT.KEYWORD, 'match')


class Error(Exception):
    """Gathers all information about an error that occurred with compiling krimson code, such as the characters in the
    file where it was detected and any additional information about it.

    It gets formatted in a user-friendly way"""
    def __init__(self, error: Enum, location: FileRange, global_vars: Globals, *args: str):
        self.e: Enum = error
        """Error enum (LexicalError | SyntaxError | TypeError) containing the message describing the error"""

        self.location: FileRange = location
        """Location of the error on the source code"""

        self.global_vars: Globals = global_vars

        self.code_lines: list[str] = self.get_file_lines()
        """Line containing the input krimson code where the error occurred"""

        self.file_name: str = global_vars.FILE_NAME
        """Name of the input file the error occurred"""

        self.args: tuple[str, ...] = args
        """Extra arguments needed for a better custom error message"""
        return

    def get_file_lines(self) -> list[str]:
        lines = []
        for i in range(self.location.line_start, self.location.line_end + 1):
            lines.append(self.global_vars.PROGRAM_LINES[i - 1])

        return lines

    def __str__(self):
        string = f'{self.file_name}:{self.location.start}:{self.location.line_start}: {self.e.value.format(*self.args)}\n'
        for i, line in enumerate(self.code_lines):
            string += f'{line}\n{" " * (self.location.start - 1)}{"^" * (self.location.end - self.location.start + 1)}'
        return string

    def __repr__(self):
        return self.__str__()


# Literals

class Literal(Token):
    """Special Case of Token, that has a token type (tt) of ``TT.LITERAL``.

    It contains an extra field ``self.literal_types`` for the krimson type of the literal value"""
    def __init__(self, value, types: set['Type'], location = FileRange(-1, -1, -1, -1)):
        super().__init__(TT.LITERAL, value, location)
        self.possible_types: set['Type'] = types

        (t,) = types if len(types) == 1 else (None,)
        self.type: Optional[Type] = t
        """krimson Type of the Literal value of the token"""
        return

    def update_literal(self, context, parent):
        """Updates the literal value of the token"""
        pass

    def type_check_literal(self, context, expected_types: Optional[set['Type']] = None) -> None:
        """Checks if the literal value of the token is of the expected type
        :param context:
        :param expected_types:
        """
        if expected_types is None:
            for t in self.possible_types:
                self.type = t
                return
            return

        types = self.possible_types.intersection(expected_types)
        if len(types) > 0:
            # don't need to error, because the typechecker will handle the rest
            self.type = types.pop()
        else:
            self.type = self.possible_types.pop()
        return

    def __str__(self):
        return f'{self.value}'

    def __repr__(self):
        return f'<{self.value}: {self.type}>'



# Types directly supported by the compiler

class Type:
    """Class containing all information regarding a type in krimson, such as the name of the type, and extra generic
    types associated with it"""
    def __init__(self, name: Optional[Token] = None):
        self.name_tok: Token = name if name is not None else Token(TT.IDENTIFIER, '')
        """name of the krimson type"""

        self.size = 1
        """compile-time size of an object of this type"""
        return

    def __eq__(self, other: 'Type'):
        return other is not None and self.name_tok == other.name_tok

    def __hash__(self):
        return self.name_tok.value.__hash__()

    def get_label(self) -> str:
        """
        Generates and returns a string containing a unique string to represent the krimson type.
        :return: a string label of the type
        """
        return f'{self.name_tok.value}'

    def get_id(self) -> tuple[str, 'Type']:
        return self.name_tok.value, Types.type.value

    def get_possible_attributes(self, name: str) -> set:
        """
        Returns the set of the possible types of the attribute with the given name
        :param name: str of the attribute
        :return: the type of the attribute
        """
        return set()

    def builtin_type(self):
        return self == Types.type.value   # allows 'type' keyword to be a valid built-in type

    def __str__(self):
        return f'{self.name_tok.value}'

    def __repr__(self):
        return f'<{self.name_tok.value}>'


class InferType(Type):
    def __init__(self, tok: Token):
        super().__init__(tok)
        return

    def __eq__(self, other: 'InferType'):
        return isinstance(other, InferType)

    def __hash__(self):
        return super().__hash__()


class TupleType(Type):
    def __init__(self, types: list[Type]):
        super().__init__()
        self.types: list[Type] = types
        return

    def __eq__(self, other: 'TupleType'):
        return isinstance(other, TupleType) and self.types == other.types

    def __hash__(self):
        return sum([f.__hash__() for f in self.types])

    def get_label(self) -> str:
        return f'tuple_{"_".join([t.get_label() for t in self.types])}'

    def builtin_type(self):
        return True

    def __str__(self):
        return f'({", ".join([str(t) for t in self.types])})'

    def __repr__(self):
        return f'<tuple({", ".join([str(t) for t in self.types])})>'


class VoidType(TupleType):
    def __init__(self):
        super().__init__([])
        return

    def __eq__(self, other: 'VoidType'):
        return isinstance(other, VoidType)

    def __hash__(self): # need to redefine this otherwise __eq__ will make the hash be 0
        return super().__hash__()

    def get_label(self) -> str:
        return f'{self.name_tok.value}'

    def __str__(self):
        return f'()'

    def __repr__(self):
        return f'<void>'


class FunctionType(Type):
    def __init__(self, arg: Type, ret: Type):
        super().__init__()
        self.arg: Type = arg
        self.ret: Type = ret
        return

    def __eq__(self, other: 'FunctionType'):
        return isinstance(other, FunctionType) and self.arg == other.arg and self.ret == other.ret

    def __hash__(self):
        return self.arg.__hash__() ^ self.ret.__hash__()

    def get_label(self) -> str:
        return f'func_{self.arg.get_label()}_to_{self.ret.get_label()}'

    def builtin_type(self):
        return True

    def __str__(self):
        return f'{self.arg} -> {self.ret}'

    def __repr__(self):
        return f'<fn({self.arg} -> {self.ret})>'


class ArrayType(Type):
    def __init__(self, arr_type: Type):
        super().__init__()
        self.arr_type: Type = arr_type
        return

    def __eq__(self, other: 'ArrayType'):
        return isinstance(other, ArrayType) and self.arr_type == other.arr_type

    def __hash__(self):
        return self.arr_type.__hash__() + 1 # just to change from the normal type hash

    def get_label(self) -> str:
        return f'array_{self.arr_type.get_label()}'

    def builtin_type(self):
        return True

    def __str__(self):
        return f'[{self.arr_type}]'

    def __repr__(self):
        return f'<array[{self.arr_type}]>'



class Types(Enum):
    """Enum containing all primitive types the compiler may need to use at compile-time"""
    void = VoidType()
    type = Type(Token(TT.IDENTIFIER, 'type'))
    infer = InferType(Token(TT.IDENTIFIER, '_'))
    bool = Type(Token(TT.IDENTIFIER, 'bool'))
    nat = Type(Token(TT.IDENTIFIER, 'nat'))
    int = Type(Token(TT.IDENTIFIER, 'int'))
    frac = Type(Token(TT.IDENTIFIER, 'frac'))
    char = Type(Token(TT.IDENTIFIER, 'char'))
    array = Type(Token(TT.IDENTIFIER, 'array'))
    str = Type(Token(TT.IDENTIFIER, 'str'))


class PriorityQueue:
    def __init__(self, num_priorities: int):
        assert num_priorities > 0
        self.num_priorities: int = num_priorities
        self.queue: list[list] = [[] for _ in range(num_priorities)]
        self.i: int = 0
        self.p: int = 0
        return

    def enqueue(self, item, priority: int = 0):
        if priority >= self.num_priorities:
            priority = self.num_priorities - 1
        elif priority < 0:
            priority = 0
        self.queue[priority].append(item)
        return

    def __iter__(self):
        self.i = 0
        self.p = 0
        return self

    def __next__(self):
        if self.p >= self.num_priorities:
            raise StopIteration
        if self.i >= len(self.queue[self.p]):
            self.p += 1
            self.i = 0
            return self.__next__()
        item = self.queue[self.p][self.i]
        self.i += 1
        return item

    def dequeue(self):
        for i in range(self.num_priorities):
            if len(self.queue[i]) > 0:
                return self.queue[i].pop(0)
        return None