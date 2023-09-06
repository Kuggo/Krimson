from sys import argv, stdout, stderr
import os
from Lexer import *
from Parser import *


def main():
    src_name = argv[1] if len(argv) >= 2 else None
    dest_name = argv[2] if len(argv) >= 3 else None

    if src_name == '--help':
        print('usage: krimson <source_file> <destination_file>')

    source = r''''''
    src_name = os.path.join(os.path.dirname(__file__), '../input.krim')

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

    global_vars = Globals(src_name, source.split('\n'))
    # Object that holds the variables related to the input code (to be configurable from other modules when compiling)

    lexer = Lexer(source, global_vars)
    lexer.tokenize()

    if len(lexer.errors) > 0:
        for err in lexer.errors:
            print(err.__repr__(), file=stderr)
        exit(1)

    print(lexer.tokens, file=dest)

    parser = Parser(lexer.tokens, global_vars)

    parser.parse()

    if len(parser.errors) > 0:
        for err in parser.errors:
            print(err.__repr__(), file=stderr)
        exit(1)

    # primitives_ctx = get_primitives()
    context = Context(global_vars)  # Context(primitives_ctx)
    context.scope_level = 0

    ast = parser.ast.update(context, None)

    if len(context.errors) > 0:
        for err in context.errors:
            print(err.__repr__(), file=stderr)
        exit(1)

    print(ast)

    return


def test():
    """Checks if the compiler is working correctly"""

    src = """
var: int = 1
var = 2

inc: fn = (i: nat) -> out: nat {
    out = i + 1
}

plus1: nat -> nat = inc

Coord: type = (nat, nat)

XY: type = {
    x: nat
    y: nat
}

AllowedOptions: type = {
  | NoSizeOption
  | AliasOption = int
  | TupleOption = (int, int)
  | ProductOption = {x: int y: int}
  | EnumInception = {
      | A
      | B
    }
}
"""

    global_vars = Globals('testing', src.split('\n'))

    lexer = Lexer(src, global_vars)
    lexer.tokenize()

    if len(lexer.errors) > 0:
        for err in lexer.errors:
            print(err.__repr__(), file=stderr)
        exit(1)

    assert lexer.tokens == [
        Token(TT.IDENTIFIER, 'var'), Token(TT.SEPARATOR, ':'), Token(TT.IDENTIFIER, 'int'), Token(TT.OPERATOR, '='), Token(TT.LITERAL, 1),
        Token(TT.IDENTIFIER, 'var'), Token(TT.OPERATOR, '='), Token(TT.LITERAL, 2),
        Token(TT.IDENTIFIER, 'inc'), Token(TT.SEPARATOR, ':'), Token(TT.KEYWORD, 'fn'), Token(TT.OPERATOR, '='),
        Token(TT.SEPARATOR, '('), Token(TT.IDENTIFIER, 'i'), Token(TT.SEPARATOR, ':'), Token(TT.IDENTIFIER, 'nat'), Token(TT.SEPARATOR, ')'), Token(TT.OPERATOR, '->'), Token(TT.IDENTIFIER, 'out'), Token(TT.SEPARATOR, ':'), Token(TT.IDENTIFIER, 'nat'),
        Token(TT.SEPARATOR, '{'), Token(TT.IDENTIFIER, 'out'), Token(TT.OPERATOR, '='), Token(TT.IDENTIFIER, 'i'), Token(TT.OPERATOR, '+'), Token(TT.LITERAL, 1), Token(TT.SEPARATOR, '}'),
        Token(TT.IDENTIFIER, 'plus1'), Token(TT.SEPARATOR, ':'), Token(TT.IDENTIFIER, 'nat'), Token(TT.OPERATOR, '->'), Token(TT.IDENTIFIER, 'nat'), Token(TT.OPERATOR, '='), Token(TT.IDENTIFIER, 'inc'),
        Token(TT.IDENTIFIER, 'Coord'), Token(TT.SEPARATOR, ':'), Token(TT.KEYWORD, 'type'), Token(TT.OPERATOR, '='), Token(TT.SEPARATOR, '('), Token(TT.IDENTIFIER, 'nat'), Token(TT.SEPARATOR, ','), Token(TT.IDENTIFIER, 'nat'), Token(TT.SEPARATOR, ')'),
        Token(TT.IDENTIFIER, 'XY'), Token(TT.SEPARATOR, ':'), Token(TT.KEYWORD, 'type'), Token(TT.OPERATOR, '='), Token(TT.SEPARATOR, '{'), Token(TT.IDENTIFIER, 'x'), Token(TT.SEPARATOR, ':'), Token(TT.IDENTIFIER, 'nat'), Token(TT.IDENTIFIER, 'y'), Token(TT.SEPARATOR, ':'), Token(TT.IDENTIFIER, 'nat'), Token(TT.SEPARATOR, '}'),
        Token(TT.IDENTIFIER, 'AllowedOptions'), Token(TT.SEPARATOR, ':'), Token(TT.KEYWORD, 'type'), Token(TT.OPERATOR, '='), Token(TT.SEPARATOR, '{'), Token(TT.OPERATOR, '|'),
        Token(TT.IDENTIFIER, 'NoSizeOption'), Token(TT.OPERATOR, '|'),
        Token(TT.IDENTIFIER, 'AliasOption'), Token(TT.OPERATOR, '='), Token(TT.IDENTIFIER, 'int'), Token(TT.OPERATOR, '|'),
        Token(TT.IDENTIFIER, 'TupleOption'), Token(TT.OPERATOR, '='), Token(TT.SEPARATOR, '('), Token(TT.IDENTIFIER, 'int'), Token(TT.SEPARATOR, ','), Token(TT.IDENTIFIER, 'int'), Token(TT.SEPARATOR, ')'), Token(TT.OPERATOR, '|'),
        Token(TT.IDENTIFIER, 'ProductOption'), Token(TT.OPERATOR, '='), Token(TT.SEPARATOR, '{'), Token(TT.IDENTIFIER, 'x'), Token(TT.SEPARATOR, ':'), Token(TT.IDENTIFIER, 'int'), Token(TT.IDENTIFIER, 'y'), Token(TT.SEPARATOR, ':'), Token(TT.IDENTIFIER, 'int'), Token(TT.SEPARATOR, '}'), Token(TT.OPERATOR, '|'),
        Token(TT.IDENTIFIER, 'EnumInception'), Token(TT.OPERATOR, '='), Token(TT.SEPARATOR, '{'), Token(TT.OPERATOR, '|'), Token(TT.IDENTIFIER, 'A'), Token(TT.OPERATOR, '|'), Token(TT.IDENTIFIER, 'B'), Token(TT.SEPARATOR, '}'), Token(TT.SEPARATOR, '}'),
    ]

    parser = Parser(lexer.tokens, global_vars)

    parser.parse()

    if len(parser.errors) > 0:
        for err in parser.errors:
            print(err.__repr__(), file=stderr)
        exit(1)

    print("All tests passed!")
    return


def get_primitives() -> Context:
    file_path = os.path.join(os.path.dirname(__file__), '../libraries/primitives.krim')
    with open(file_path, 'r') as f:
        source = f.read()
        global_vars = Globals(file_path, source.split('\n'))
        lexer = Lexer(source, global_vars)
        lexer.tokenize()

        if len(lexer.errors) > 0:
            for err in lexer.errors:
                print(err.__repr__(), file=stderr)
            exit(1)

        parser = Parser(lexer.tokens, global_vars)
        parser.parse()

        if len(parser.errors) > 0:
            for err in parser.errors:
                print(err.__repr__(), file=stderr)
            exit(1)

        context = Context(global_vars)
        parser.ast.process_body(context)    # skipping some parts on self.update()

    return context


if __name__ == '__main__':
    main()
