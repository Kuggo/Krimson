# Krimson

Krimson is a custom-made programming language that aims to make learning its low level concepts easily.
Both beginners and old time programmers should find it intuitive. 

Its syntax is inspired by python and its type system is heavily based on set theory.

---

## How to use

wip

## Syntax

A krimson program is just a text file that contains the source code for a program. 
That code is examined, validated and compiled into another format so that the krimson virtual machine can execute it. 

Before diving into its syntax, lets clear up some useful concepts we are going to need:

We can formally define our syntax using a notation (BNF). Here are its rules:
- A specific keyword or symbol is wrapped around quotes `" "`.
- Several options that match a rule are separated by `|`.
- A rule that is applied 0 or more times has `*` after it.
- A rule that is optional is put between brackets `[]`.
- To change the precedence which rule to apply, we use parenthesis `()`. Inside parenthesis is evaluated first.
- To define a new rule we put the name of the rule and use ``:=`` to separate it  from the definition. These rules are recursive, meaning that a rule can reference itself or others that use it. 
To use a rule inside another one, we just use its name.

Here are some auxiliary definitions to help us define the rest of the syntax (in BNF):

    binary_operator := "+" | "-" | "*" | "/" | "%" | "==" | "!=" | "<" | "<=" | ">" | ">=" | "&" | "|" | "^" | "<<" | ">>"

    unary_operator := "-" | "!" | "~"

    separator := "(" | ")" | "{" | "}" | "[" | "]" | "," | ";" | "=" | ":" | "." | "->" | "?"

    number := [0-9]+

    comment := "//" [^"\n"]* "\n"


    identifier := [a-zA-Z_][a-zA-Z0-9_]*
    An identifier is a sequence of alphanumeric characters (letters of the english alphabet, 0 to 9 digits and underscore). 
    Note that it cannot start with a digit.

    literal := "true" | "false" | number
    A literal is the representation of a value. True, false or numbers fall into that.

    symbol := binary_operator | unary_operator | separator
    A symbol is a single/sequence of special characters.

    keyword := "if" | "else" | "while" | "skip" | "break" | "fn" | "type" | "macro"
    A keyword is a reserved sequence of characters that form a word that cannot be used as an identifier or literal.

And the rest of the rules of the syntax are as follows:

    program := statement*

    statement := expression | new_scope | if_statement | while_statement | skip_statement | break_statement | variable_definition | function_definition

    new_scope := "{" statement* "}"

    if_statement := "if" expression statement ["else" statement]
    Conditionally executes a statement based of the value of the boolean expression 

    while_statement := "while" expression statement
    Executes a statement as long as the value of the boolean expression is true 

    skip_statement := "skip" [number]
    Skips the current iteration of while loop and starts the next one 

    break_statement := "break" [number]
    Stops the current iteration of while loop and exits the loop

    expression := literal | identifier | function_call | assignment | binary_operation | unary_operation
    Expressions are anything that is or produces a value

    function_call := expression "(" expression* ")"

    assignment := expression "=" expression

    binary_operation := expression binary_operator expression

    unary_operation := unary_operator expression

    variable_declaration := identifier ":" type

    variable_definition := variable_declaration ["=" expression]
    Variables are values that can change throughout the program and are associated with a given identifier

    function := "(" variable_declaration ")" "->" variable_declaration statement

    function_definition := identifier ":" "fn" ["=" function]
    Functions are pieces of code that execute some statements with possibly different values everytime we call them. 
    Note that functions in krimson are not pure functions and can have side effects.
    If a function need multiple input parameters, then those are grouped as one tuple and passed as 1 argument. Same for the output parameter. If there is no parameter needed, empty tuple should be used.
    Both input and output parameters can have a custom name using variable declaration syntax. If the parameter is a tuple of multiple parameters then each member of the tuple can receive a name.
    There are 2 ways we can give a value to a function. The first one is by assigning it a value of the same function type.

    type_declaration := identifier ":" "type" "=" type

    type := identifier | tuple_type | enum_type | function_type | array_type
    Types are sets of values that a variable with that type must be part of at all times.

    tuple_type := "(" type* ")"

    enum_type := "{" type* "}"

    function_type := type "->" type

    array_type := "[" type "]"


Types:
Types are sets of values that a variable with that type must be part of at all times.
The compiler supports syntax on how to work with types, but the user can use a type alias to give a specific type a name.
To define a new one declares a variable and that has a type of keyword "type".
identifier : "type" = type
The language comes with a few defined types already. For more info, refer to that guide


Variables:
Variables are values that can change throughout the program and are associated with a given identifier.
To define a variable one must specify its name, type and a starting value.
After the variable was defined, its type is no longer required. If another variable with the same name is defined, then the first one will no longer be accessible. 

Functions:
Functions are pieces of code that execute some statements with possibly different values everytime we call them. 
Note that functions in krimson are not pure functions and can have side effects.
Defining functions works the same way as defining any other variable. The only special thing is that the type of the variable is a function type. To do that one uses the symbol -> to separate the input and the output.
If a function need multiple input parameters, then those are grouped as one tuple and passed as 1 argument. Same for the output parameter. If there is no parameter needed, empty tuple should be used.
Both input and output parameters can have a custom name using variable declaration syntax. If the parameter is a tuple of multiple parameters then each member of the tuple can receive a name.
There are 2 ways we can give a value to a function. The first one is by assigning it a value of the same function type.
function Name: input -> output = expression 

The second way is by providing code for the function like so:
function Name: fn = input -> output statement.

