from sys import argv, stdout, stderr
import os
import Constants
from Lexer import *
from Parser import *


def main():
    src_name = argv[1] if len(argv) >= 2 else None
    dest_name = argv[2] if len(argv) >= 3 else None

    if src_name == '--help':
        print('usage: krimson <source_file> <destination_file>')

    source = r'''
    /*class yeet {
        var int ver;
        
        fn nat func() {
            return 0
        }
    }
    
    yeet.func()*/
    
    var int ver
    var int vir
    ver = ver + vir
    
    while true {break}
    '''

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

    Constants.global_vars.PROGRAM_LINES = source.split('\n')
    Constants.global_vars.FILE_NAME = src_name

    lexer = Lexer(source)
    lexer.tokenize()

    if len(lexer.errors) > 0:
        for err in lexer.errors:
            print(err.__repr__(), file=stderr)
        exit(1)

    print(lexer.tokens, file=dest)

    parser = Parser(lexer.tokens)

    parser.parse()

    if len(parser.errors) > 0:
        for err in parser.errors:
            print(err.__repr__(), file=stderr)
        exit(1)

    # print(parser.ast)

    primitives_ctx = get_primitives()
    context = Context(primitives_ctx)
    context.scope_level = 0

    ast = parser.ast.update(context, None)
    parser.ast.alloc_vars(context)  # top level execution needs this

    if len(context.errors) > 0:
        for err in context.errors:
            print(err.__repr__(), file=stderr)
        exit(1)

    print(ast)

    return


def get_primitives() -> Context:
    with open('primitives.krim', 'r') as f:
        lexer = Lexer(f.read())
        lexer.tokenize()

        if len(lexer.errors) > 0:
            for err in lexer.errors:
                print(err.__repr__(), file=stderr)
            exit(1)

        parser = Parser(lexer.tokens)
        parser.parse()

        if len(parser.errors) > 0:
            for err in parser.errors:
                print(err.__repr__(), file=stderr)
            exit(1)

        context = Context()
        parser.ast.process_body(context)    # skipping some parts on self.update()

    return context


if __name__ == '__main__':
    main()
