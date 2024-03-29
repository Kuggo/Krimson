from Constants import *
from copy import copy


class LexicalError(Enum):
    """Enum containing all the custom error messages that the lexer may generate when tokenizing."""
    invalid_char = 'Invalid character'
    miss_close_sym = 'Missing single quote {!s}'
    invalid_escape_code = "Invalid escape sequence code '{!s}'"
    too_many_points = 'Too many radix points in number literal'
    invalid_num = 'Invalid numeric literal'


class Lexer:
    """
    Generates a collection of tokens from a string containing krimson code

    By convention after each function the next char to be read will be the one after the last char of the last generated
    token
    """

    def __init__(self, input_string: str, global_vars: Globals) -> None:
        self.input_string: str = input_string + '    '
        """the string containing the code"""

        self.global_vars: Globals = global_vars
        """Object that holds the variables related to the input code"""

        self.i: int = 0
        """index on the input string of the current char"""

        self.len: int = len(self.input_string)
        """total length of the input string. To avoid calling ``len()`` on ``self.has_next()`` multiple times"""

        self.line: int = 1
        """index of the line the index is on (starts at 1 and not 0)"""

        self.line_offset: int = -1
        """index before where the current line starts on (the last index of the previous line)"""

        self.peak: str = self.input_string[0]
        """current char pointed by index"""

        self.tokens: list[Token] = []
        """output collection of tokens generated upon calling ``self.tokenize()``"""

        self.errors: list[Error] = []
        """collection of Errors found by the Lexer upon calling ``self.tokenize()``"""
        return

    def tokenize(self) -> None:
        """
        Generates a collection of tokens from self.input_string using krimson's lexer rules, and outputs it to
        ``self.tokens`` and the errors found to ``self.errors``
        """

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
                    self.symbol(TT.OPERATOR, self.peak, self.i)
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
                self.error(LexicalError.invalid_char, self.i, self.i)

        return

    def make_word(self) -> None:
        """
        Creates an Identifier token using the next group of contiguous alphanumeric characters (underscore included).

        Note: The first character cannot be a digit.

        Outputs that token directly to the end of ``self.tokens``
        """

        start = self.i
        while self.has_next() and (self.peak.isalnum() or self.peak == '_'):
            self.advance()

        name = self.input_string[start:self.i]
        if name in KEYWORDS:
            token_type = TT.KEYWORD
        elif name in BOOLEANS:
            self.literal(name, {copy(Types.bool.value)}, start, self.i - 1)
            return
        else:
            token_type = TT.IDENTIFIER

        self.token(token_type, name, start, self.i - 1)
        return

    def make_number(self) -> None:
        """
        Creates a numeric Literal token using the next group of contiguous numeric characters.

        Outputs that token directly to the end of ``self.tokens``
        """

        start = self.i
        dot_count = 0
        while self.has_next() and self.peak.isalnum():
            self.advance()
            if self.peak == '.':
                dot_count += 1
                if dot_count > 1:
                    self.error(LexicalError.too_many_points, self.i, self.i)
                    self.advance(-1)
                    break   # will not reject what it found so far
                self.advance()
            continue

        if dot_count == 0:
            try:
                value = int(self.input_string[start:self.i], 0)
            except ValueError:
                self.error(LexicalError.invalid_num, start, self.i-1)
                return
            self.literal(value, {copy(Types.nat.value), copy(Types.int.value)}, start, self.i - 1)
        else:
            try:
                value = float(self.input_string[start:self.i])
            except ValueError:
                self.error(LexicalError.invalid_num, start, self.i-1)
                return
            self.literal(value, {copy(Types.frac.value)}, start, self.i - 1)
        return

    def make_symbol(self) -> None:
        """
        Creates an Operator/Separator token using the next group of contiguous symbol characters.

        Outputs that token directly to the end of ``self.tokens``
        """

        start = self.i
        if self.peak == '<':
            if self.preview() == '<':
                self.advance()
                if self.preview() == '=':
                    self.advance()
                    self.symbol(TT.OPERATOR, '<<=', start)
                else:
                    self.symbol(TT.OPERATOR, '<<', start)
            elif self.preview() == '=':
                self.advance()
                self.symbol(TT.OPERATOR, '<=', start)

            else:
                self.symbol(TT.OPERATOR, '<', start)

        elif self.peak == '>':
            if self.preview() == '>':
                self.advance()
                if self.preview() == '=':
                    self.advance()
                    self.symbol(TT.OPERATOR, '>>=', start)
                else:
                    self.symbol(TT.OPERATOR, '>>', start)
            elif self.preview() == '=':
                self.advance()
                self.symbol(TT.OPERATOR, '>=', start)

            else:
                self.symbol(TT.OPERATOR, '>', start)

        elif self.peak == '=':
            if self.preview() == '=':
                self.advance()
                self.symbol(TT.OPERATOR, '==', start)
            else:
                self.symbol(TT.OPERATOR, '=', start)

        elif self.peak == '!':
            if self.preview() == '=':
                self.advance()
                self.symbol(TT.OPERATOR, '!=', start)
            else:
                self.symbol(TT.OPERATOR, '!', start)

        elif self.peak == '&':
            if self.preview() == '=':
                self.advance()
                self.symbol(TT.OPERATOR, '&=', start)
            else:
                self.symbol(TT.OPERATOR, '&', start)

        elif self.peak == '|':
            if self.preview() == '=':
                self.advance()
                self.symbol(TT.OPERATOR, '|=', start)
            else:
                self.symbol(TT.OPERATOR, '|', start)

        elif self.peak == '^':
            if self.preview() == '=':
                self.advance()
                self.symbol(TT.OPERATOR, '^=', start)
            else:
                self.symbol(TT.OPERATOR, '^', start)

        elif self.peak == '+':
            if self.preview() == '=':
                self.advance()
                self.symbol(TT.OPERATOR, '+=', start)
            else:
                self.symbol(TT.OPERATOR, '+', start)

        elif self.peak == '-':
            if self.preview() == '=':
                self.advance()
                self.symbol(TT.OPERATOR, '-=', start)
            elif self.preview() == '>':
                self.advance()
                self.symbol(TT.OPERATOR, '->', start)
            else:
                self.symbol(TT.OPERATOR, '-', start)

        elif self.peak == '*':
            if self.preview() == '=':
                self.advance()
                self.symbol(TT.OPERATOR, '*=', start)
            else:
                self.symbol(TT.OPERATOR, '*', start)

        elif self.peak == '/':
            if self.preview() == '=':
                self.advance()
                self.symbol(TT.OPERATOR, '/=', start)
            else:
                self.symbol(TT.OPERATOR, '/', start)

        elif self.peak == '%':
            if self.preview() == '=':
                self.advance()
                self.symbol(TT.OPERATOR, '%=', start)
            else:
                self.symbol(TT.OPERATOR, '%', start)

        elif self.peak == '.':
            self.symbol(TT.OPERATOR, self.peak, start)

        elif self.peak == '?':
            self.symbol(TT.OPERATOR, self.peak, start)

        else:
            self.symbol(TT.SEPARATOR, self.peak, self.i)

        self.advance()
        return

    def make_string(self) -> None:
        """
        Creates a string Literal token using the next group of characters between double quotes ``""``.

        Note: Escape sequences work and are correctly parsed

        Outputs that token directly to the end of ``self.tokens``
        """

        self.advance()
        start = self.i

        while self.has_next() and self.peak != '"':
            self.advance()

        try:
            string = self.input_string[start:self.i].encode('raw_unicode_escape').decode('unicode_escape')
        except UnicodeEncodeError:
            self.error(LexicalError.invalid_escape_code, start, self.i, self.input_string[start:self.i])
            return

        t = copy(Types.str.value)
        t.size = len(string)
        self.literal(string, {t}, start - 1, self.i)
        self.advance()
        return

    def make_char(self) -> None:
        """
        Creates a char Literal token using the next group of characters between single quotes ``´´``. Is similar to
        ``self.make_string()`` but if the length of the string is not 1 then an error is added

        Note: Escape sequences work and are correctly parsed

        Outputs that token directly to the end of ``self.tokens``
        """

        self.advance()
        start = self.i

        while self.has_next() and self.peak != "'":
            if self.peak == '\n':
                self.error(LexicalError.miss_close_sym, start, self.i, "'")
                return
            self.advance()

        char = self.input_string[start:self.i].encode('raw_unicode_escape').decode('unicode_escape')
        if len(char) != 1:
            self.error(LexicalError.invalid_char, start - 1, self.i)
        else:
            self.literal(char, copy(Types.char.value), start - 1, self.i)
            self.advance()
        return

    def inline_comment(self) -> None:
        """
        Skips the next group of characters until a new line is found '``\\n``'.
        """

        while self.has_next() and self.peak != '\n':
            self.advance()
        return

    def multi_line_comment(self) -> None:
        """
        Recursively skips the next group of characters until a ``*/`` is found.
        Nested comments are allowed
        """

        self.advance(2)
        scope = 0
        while self.has_next(1):
            if self.peak == '/' and self.preview() == '*':
                self.advance(2)
                scope += 1
            if self.peak == '*' and self.preview() == '/':
                self.advance(2)
                if scope == 0:
                    break
                else:
                    scope -= 1

            self.advance()
        return

    def token(self, tt: TT, value, start, end) -> Token:
        """
        Creates a new Token and adds it to the output collection of tokens ``self.tokens``

        :param tt: token type of the new token
        :param value: value of the token
        :param start: start index of the token
        :param end: end index of the token
        :return: the recently created token
        """

        tok = Token(tt, value, FileRange(start - self.line_offset, self.line, end - self.line_offset, self.line))
        self.tokens.append(tok)
        return tok

    def literal(self, value, possible_types: set[Type], start, end) -> Token:
        """
        Creates a new Literal Token and adds it to the output collection of tokens ``self.tokens``

        :param value: value of the token
        :param possible_types: type of the literal
        :param start: start index of the token
        :param end: end index of the token
        :return: the recently created token
        """

        tok = Literal(value, possible_types, FileRange(start - self.line_offset, self.line, end - self.line_offset, self.line))
        self.tokens.append(tok)
        return tok

    def symbol(self, tt: TT, value: str, start) -> Token:
        """
        Creates a new Symbol Token and adds it to the output collection of tokens ``self.tokens``

        :param tt: value of the token
        :param value: type of the symbol
        :param start: start index of the token
        :return: the recently created token
        """

        tok = Token(tt, value, FileRange(start - self.line_offset, self.line, start - self.line_offset, self.line))
        self.tokens.append(tok)
        return tok

    def error(self, error: LexicalError, start, end, *args) -> None:
        """
        Creates a new Error and adds it to the error collection of tokens ``self.errors``

        :param error: the error found
        :param start: the position where it started
        :param end: the end of where the error occurred
        :param args: extra arguments for custom error message formatting
        """

        loc = FileRange(start - self.line_offset, self.line, end - self.line_offset, self.line)
        self.errors.append(Error(error, loc, self.global_vars, *args))
        self.advance()
        return

    def advance(self, i=1) -> None:
        """
        Moves the index to i characters ahead and updates all state variables

        :param i: number of characters to advance
        """

        if self.peak == '\n':
            self.line_offset = self.i
            self.line += 1
        self.i += i
        if self.has_next():
            self.peak = self.input_string[self.i]
        else:
            pass  # raise EOFError

    def preview(self, i=1) -> str:
        """
        returns the character after ``i`` positions of current char

        :param i: offset of the position to look ahead
        :return: the character at the specified position
        """
        return self.input_string[self.i + i]

    def has_next(self, i=0) -> bool:
        """
        checks if there are more characters to be processed

        :param i: offset of current index
        :return: True if there are more characters to be processed, False otherwise
        """
        return self.i + i < self.len
