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




