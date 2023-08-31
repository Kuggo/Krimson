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

    # push_sp = Instruction('sp')
    push_bp = Instruction('pushBP')
    pop_bp = Instruction('popBP')
    push_a = Instruction('pushA')
    top_a = Instruction('topA')
    pop_a = Instruction('popA')
    push_b = Instruction('pushB')
    top_b = Instruction('topB')
    pop_b = Instruction('popB')
    move_ab = Instruction('moveAB')
    move_ba = Instruction('moveBA')

    imm_a = Instruction('immA')
    imm_b = Instruction('immB')

    call = Instruction('call')
    ret = Instruction('ret')
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

    halt = Instruction('halt')
    

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
    'topA': 19,
    'popA': 20,
    'pushB': 21,
    'topB': 22,
    'popB': 23,
    'pushBP': 24,
    'popBP': 25,
    'moveAB': 26,
    'moveBA': 27,

    'call': 28,
    'ret': 29,
    'halt': 30,

    'branch': 1 << minor_opcode_bits,
    'in': 2 << minor_opcode_bits,
    'out': 3 << minor_opcode_bits,
}

opcodes_reverse: dict[int, str] = {}
"""Dict for getting opcode name from opcode value. Used in debugging"""
for key, v in opcodes.items():
    opcodes_reverse[v] = key



# some test


# programs

fibb = [
    Instruction('immA', imm=1),
    Instruction('immB', imm=0),
    Instruction('pushB'),

    Instruction('moveAB'),
    Instruction('popA'),
    Instruction('add'),
    Instruction('out', 0),
    Instruction('pushB'),

    Instruction('immB', imm=5),
    Instruction('branch', 3),

    Instruction('halt'),
]

count_down = [
    Instruction('immB', imm=5),
    Instruction('pushB'),
    Instruction('immA', imm=10),
    Instruction('out', 0),
    Instruction('dec'),
    Instruction('branch', 1),
    Instruction('halt')
]
