from Constants import *


class Instruction:
    def __init__(self, opcode: str, operand=0, imm=None):
        self.opcode: str = opcode
        self.operand: int = operand
        self.imm: Optional[str] = imm
        return

    def bytecode(self) -> list[int]:
        assert self.opcode in opcodes
        if self.imm is None:
            return [opcodes[self.opcode] | self.operand]
        else:
            return [opcodes[self.opcode] | self.operand, self.imm]

    def __repr__(self):
        string = self.opcode
        if self.operand is not None:
            string += f' {self.operand}'

        if self.imm is not None:
            string += f', {self.imm}'

        return string


class Operations(Enum):
    load = Instruction('load')
    store = Instruction('store')

    add = Instruction('add')
    sub = Instruction('sub')
    mlt = Instruction('mlt')
    div = Instruction('div')
    mod = Instruction('mod')
    neg = Instruction('neg')
    inc = Instruction('inc')
    dec = Instruction('dec')

    and_ = Instruction('and')
    or_ = Instruction('or')
    xor = Instruction('xor')
    not_ = Instruction('not')
    lsh = Instruction('lsh')
    rsh = Instruction('rsh')

    push_bp = Instruction('bp')
    push_sp = Instruction('sp')
    push_acc = Instruction('acc')
    push_b = Instruction('b')
    imm = Instruction('imm')

    call = Instruction('call')
    branch = Instruction('branch')
    branch_eq = Instruction('branch', 0)
    branch_dif = Instruction('branch', 1)
    branch_gt = Instruction('branch', 2)
    branch_gte = Instruction('branch', 3)
    branch_lt = Instruction('branch', 4)
    branch_lte = Instruction('branch', 5)
    branch_false = Instruction('branch', 6)
    branch_true = Instruction('branch', 7)

    print_nat = Instruction('out', 0)
    print_int = Instruction('out', 1)
    print_char = Instruction('out', 2)
    print_bool = Instruction('out', 3)
    print_frac = Instruction('out', 4)
    input = Instruction('in', 0)
    

minor_opcode_bits: int = 6

minor_opcodes_bit_mask: int = (1 << minor_opcode_bits) - 1

instruction_size: int = 8

opcodes: dict[str, int] = {
    'add': 0,
    'sub': 1,
    'mlt': 2,
    'div': 3,
    'mod': 4,
    'neg': 5,
    'inc': 6,
    'dec': 7,

    'and': 8,
    'or': 9,
    'xor': 10,
    'not': 11,
    'lsh': 12,
    'rsh': 13,

    'load': 14,
    'store': 15,

    'immA': 16,
    'immB': 17,

    'pushA': 18,
    'popA': 19,
    'pushB': 20,
    'popB': 21,
    'pushBP': 22,
    'popBP': 23,

    'call': 24,
    'ret': 25,
    'halt': 26,

    'branch': 1 << minor_opcode_bits,
    'in': 2 << minor_opcode_bits,
    'out': 3 << minor_opcode_bits,
}

opcodes_reverse: dict[int, str] = {}
"""Dict for getting opcode name from opcode value. Used in debugging"""
for key, value in opcodes.items():
    opcodes_reverse[value] = key

# some test


# programs

fibb = [
    Instruction('immA', imm=1),
    Instruction('immB', imm=0),
    Instruction('out', 0),
    Instruction('add'),

    Instruction('pushA'),
    Instruction('immA', imm=2),
    Instruction('popB'),
    Instruction('branch', 3),
    Instruction('halt')
]

fibb2 = [
    Instruction('immA', imm=1),
    Instruction('immB', imm=0),
    Instruction('pushA'),

    Instruction('popA'),
    Instruction('pushA'),
    Instruction('add'),
    Instruction('popB'),
    Instruction('pushA'),

    Instruction('out', 0),

    Instruction('immA', imm=3),
    Instruction('branch', 3),

    Instruction('halt'),
]

count_down = [
    Instruction('immB', imm=10),
    Instruction('pushB'),
    Instruction('popA'),
    Instruction('out', 0),
    Instruction('dec'),
    Instruction('pushA'),
    Instruction('immA', imm=1),
    Instruction('popB'),
    Instruction('branch', 1),
    Instruction('halt')
]
