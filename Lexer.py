from Constants import *


class LexicalError(Enum):
    invalid_char = 'Invalid character'
    miss_close_sym = 'Missing single quote {}'


class Lexer:
    """Generates a collection of tokens from a string containing krimson code

    By convention after each function the next char to be read will be the one after the last char of the last generated
    token"""

    def __init__(self, input_string) -> None:
        self.input_string: str = input_string + '    '
        """the string containing the code"""

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
        """Generates a collection of tokens from self.input_string using krimson's lexer rules, and outputs it to
        ``self.tokens`` and the errors found to ``self.errors``"""

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
                self.error(LexicalError.invalid_char, self.i, self.i)

        return

    def make_word(self) -> None:
        """Creates an Identifier token using the next group of contiguous alphanumeric characters (underscore included).

        Note: The first character cannot be a digit.

        Outputs that token directly to the end of ``self.tokens``"""

        start = self.i
        while self.has_next() and (self.peak.isalnum() or self.peak == '_'):
            self.advance()

        name = self.input_string[start:self.i]
        if name in KEYWORDS:
            token_type = TT.KEYWORD
        else:
            token_type = TT.IDENTIFIER

        self.token(token_type, name, start, self.i - 1)
        return

    def make_number(self) -> None:
        """Creates a numeric Literal token using the next group of contiguous numeric characters.

        Outputs that token directly to the end of ``self.tokens``"""

        start = self.i
        while self.has_next() and self.peak.isdigit():
            self.advance()
        value = int(self.input_string[start:self.i], 0)
        self.token(TT.LITERAL, value, start, self.i - 1)
        return

    def make_symbol(self) -> None:
        """Creates an Operator/Separator token using the next group of contiguous symbol characters.

        Outputs that token directly to the end of ``self.tokens``"""

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

        elif self.peak == '.':
            self.token(TT.OPERATOR, self.peak, start, self.i)

        else:
            self.token(TT.SEPARATOR, self.peak, self.i, self.i)

        self.advance()
        return

    def make_string(self) -> None:
        """Creates a string Literal token using the next group of characters between double quotes ``""``.

        Note: Escape sequences work and are correctly parsed

        Outputs that token directly to the end of ``self.tokens``"""

        self.advance()
        start = self.i

        while self.has_next() and self.peak != '"':
            self.advance()

        try:
            string = self.input_string[start:self.i].encode('raw_unicode_escape').decode('unicode_escape')
        except UnicodeEncodeError:
            self.error()
            return

        self.token(TT.LITERAL, string, start - 1, self.i)
        self.advance()
        return

    def make_char(self):
        """Creates a char Literal token using the next group of characters between single quotes ``´´``. Is similar to
        ``self.make_string()`` but if the length of the string is not 1 then an error is added

        Note: Escape sequences work and are correctly parsed

        Outputs that token directly to the end of ``self.tokens``"""

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
            self.token(TT.LITERAL, char, start - 1, self.i)
            self.advance()
        return

    def inline_comment(self) -> None:
        """Skips the next group of characters until a new line is found '``\\n``'."""

        while self.has_next() and self.peak != '\n':
            self.advance()
        return

    def multi_line_comment(self) -> None:
        """Skips the next group of characters until the end of comment is found '``*/``'."""

        self.advance(2)
        while self.has_next(1):
            if self.peak == '*':
                self.advance()
                if self.peak == '/':
                    self.advance()
                    return
            self.advance()

    def token(self, tt: TT, value, start, end) -> Token:
        """Creates a new Token and adds it to the output collection of tokens ``self.tokens``"""

        tok = Token(tt, value, start - self.line_offset, end - self.line_offset, self.line)
        self.tokens.append(tok)
        return tok

    def error(self, error: LexicalError, start, end, *args) -> None:
        """Creates a new Error and adds it to the error collection of tokens ``self.errors``"""

        self.errors.append(
            Error(error, start - self.line_offset, end - self.line_offset, self.line, global_vars.PROGRAM_LINES[self.line - 1],
                  args))
        self.advance()
        return

    def advance(self, i=1) -> None:
        """Moves the index to i characters ahead and updates all state variables"""

        if self.peak == '\n':
            self.line_offset = self.i
            self.line += 1
        self.i += i
        if self.has_next():
            self.peak = self.input_string[self.i]
        else:
            pass  # raise EOFError

    def preview(self, i=1) -> str:
        """returns the character after ``i`` positions of current char"""
        return self.input_string[self.i + i]

    def has_next(self, i=0) -> bool:
        """checks if there are more characters to be processed"""
        return self.i + i < self.len
