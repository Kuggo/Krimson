# Krimson

---

## Index

- [Index](Documentation.md#index)
- [VM](Documentation.md#vm)
- Language specifications
  - Syntax
  - Grammar


---

## VM

4 registers used:
SP as SP
R1 as ACC
R2 as B
R3 as BP

The ACC will always contain the top of the stack. it's the destination of all operations and the source of all the
operations
If an operation needs more than 1 operand, the B register will be used as well.
BP is used as a means to better deal with local vars, and arguments on each function scope.


## Calling convention

1. save (push) BP
2. push arguments (reverse order) (args_size > index >= 2)
3. push output address (index 1)
4. copy SP into BP
5. push return address (index 0)
6. branch to function label
7. Function's scope
   - Allocate local variables (index <= 0)
   - Operations and expressions (>= locals_size)
8. move SP upwards to deallocate all below BP (args, out addr, ret addr)
9. restore (pop) BP

BP
args
ret arg address
PC+1 (ret address)  <- BP
local vars
expressions         <- SP


## Operations

### Memory ops

- load
- store

### Math ops

- add
- sub
- mlt
- div
- mod
- neg
- inc
- dec

### Logic ops

- AND
- OR
- XOR
- NOT
- RSH
- LSH

### IMM ops

- IMM ACC (extra byte)
- IMM B (extra byte)

### Register ops

- push ACC
- push B
- pop ACC
- pop B
- push BP
- pop BP

### Branch

- Branch (cnd)

### Call

- CALL
- RET

### I/O

- Input (offset)
- Output (offset)