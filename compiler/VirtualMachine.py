from Instructions import *


class ROM:
    def __init__(self, addr_size=8):
        assert addr_size > 0
        self.address_size: int = addr_size
        self.memory: list[int] = [0] * (1 << addr_size)
        self.mask: int = (1 << addr_size) - 1

    def set_mem(self, program: list[int]) -> None:
        self.memory = [0] * (1 << self.address_size)
        if len(program) > len(self.memory):
            program = program[:len(self.memory)+1]
        for i, byte in enumerate(program):
            self.memory[i] = byte & self.mask
        return

    def __getitem__(self, item: int) -> int:
        return self.memory[item & self.mask]


class RAM(ROM):
    def __init__(self, addr_size=8):
        assert addr_size > 0
        super().__init__(addr_size)

    def __setitem__(self, key: int, value: int):
        self.memory[key & self.mask] = value & self.mask


class VM:
    def __init__(self, data_size=8):
        self.instruction_size = instruction_size
        self.rom: ROM = ROM(instruction_size)
        self.data_size: int = data_size
        self.data_mask: int = (1 << data_size) - 1
        self.ram: RAM = RAM(data_size)
        self.pc: int = -1   # counteract the first fetch
        self.sp: int = (1 << data_size) - 1  # stack starts at the top and grows downwards
        self.bp: int = self.sp
        self.acc: int = 0
        self.b: int = 0
        self.running: bool = False
        return

    @staticmethod
    def bytecode(program: list[Instruction]) -> list[int]:
        return [byte for inst in program for byte in inst.bytecode()]

    def run(self, program: list[int], debug=False):
        self.rom.set_mem(program)

        if debug:
            print('Starting Krimson VM with the following specifications',
                  f'Instruction size: {self.instruction_size}',
                  f'Data size: {self.data_size}',
                  f'ROM: {self.rom}\n')

        self.running = True
        while self.running:
            if debug:
                print(self)

            # fetch
            instruction = self.fetch()

            # decode
            major, operand = self.decode(instruction)

            # execute
            self.execute(major, operand)
            continue

        return

    def fetch(self) -> int:
        self.pc = (self.pc + 1) & self.data_mask
        return self.rom[self.pc]

    def decode(self, instruction: int) -> tuple[int, int]:
        major = instruction >> minor_opcode_bits
        operand = instruction & minor_opcodes_bit_mask
        return major, operand

    def execute(self, major, operand) -> None:
        if major == 0:
            minor_opcodes[operand](self)
        else:
            major_opcodes[major](self, operand)
        return

    # utils
    def push(self, val) -> None:
        assert self.sp > 0
        self.ram[self.sp] = val & self.data_mask
        self.sp -= 1

    def pop(self) -> int:
        assert self.sp >= 0
        self.sp += 1
        return self.ram[self.sp] & self.data_mask

    ### Instructions

    # arithmetic operations
    def add(self) -> None:
        self.acc = (self.b + self.acc) & self.data_mask

    def sub(self) -> None:
        self.acc = (self.b - self.acc) & self.data_mask

    def mlt(self) -> None:
        self.acc = (self.b * self.acc) & self.data_mask

    def div(self) -> None:
        self.acc = (self.b // self.acc) & self.data_mask

    def mod(self) -> None:
        self.acc = (self.b % self.acc) & self.data_mask

    def neg(self) -> None:
        self.acc = -self.acc & self.data_mask

    def inc(self) -> None:
        self.acc = (self.acc + 1) & self.data_mask

    def dec(self) -> None:
        self.acc = (self.acc - 1) & self.data_mask

    # bitwise operations
    def lsh(self) -> None:
        self.acc = (self.b << self.acc) & self.data_mask

    def rsh(self) -> None:
        self.acc = (self.b >> self.acc) & self.data_mask

    def band(self) -> None:
        self.acc = (self.b & self.acc) & self.data_mask

    def bor(self) -> None:
        self.acc = (self.b | self.acc) & self.data_mask

    def bxor(self) -> None:
        self.acc = (self.b ^ self.acc) & self.data_mask

    def bnot(self) -> None:
        self.acc = ~self.acc & self.data_mask

    # memory operations
    def load(self) -> None:
        self.acc = self.ram[self.b]

    def store(self) -> None:
        self.ram[self.b] = self.acc & self.data_mask

    # Subroutines
    def call(self):
        self.push(self.pc)  # pc already incremented after fetch
        self.branch()

    def ret(self):
        self.pop_acc()
        self.branch()

    # Immediate values
    def imm_acc(self) -> None:
        self.acc = self.fetch()

    def imm_b(self) -> None:
        self.b = self.fetch()

    # Stack operations
    def push_acc(self) -> None:
        self.push(self.acc & self.data_mask)

    def pop_acc(self) -> None:
        self.acc = self.pop()

    def push_b(self) -> None:
        self.push(self.b & self.data_mask)

    def pop_b(self) -> None:
        self.b = self.pop()

    def push_bp(self) -> None:
        self.push(self.bp)

    def pop_bp(self) -> None:
        self.bp = self.pop()

    # Branches
    def branch(self, operand: int = 7):
        assert 0 <= operand < 8
        cnd = conditions[operand](self.b, self.data_size)
        if cnd:
            self.pc = (self.acc - 1) & self.data_mask    # counteract pre increment on fetch
        return

    # I/O
    def input(self, operand: int = 0) -> None:
        # operand is quite useless here unless I don't accept the input unless of the specified type
        i = input()
        try:
            i = int(i) & self.data_mask
        except:
            i = 0
        self.acc = i

    def output(self, operand: int = 0) -> None:
        print(io_channels[operand](self.acc, self.data_size))

    def halt(self) -> None:
        self.running = False

    def __repr__(self):
        return f'-----------------\npc: {self.pc}\nsp: {self.sp}\nbp: {self.bp}\nreg a: {self.acc}\nreg b: {self.b}\nram: {self.ram}\n' + \
                f'\nstack:\n{self.get_stack_repr()}\nInstruction: {self.get_current_instruction()}\n'

    def get_stack_repr(self) -> str:
        return '\n'.join([f'{i}: {self.ram[i]}' for i in range(self.data_mask, self.sp, -1)])

    def get_current_instruction(self) -> str:
        instruction = self.rom[(self.pc + 1) & self.data_mask]
        major, operand = self.decode(instruction)

        if major == 0:
            i = operand
        else:
            i = major << minor_opcode_bits

        assert i in opcodes_reverse

        if 16 <= i <= 17:
            return opcodes_reverse[i] + f' {self.rom[(self.pc + 2) & self.data_mask]}'
        else:
            return opcodes_reverse[i]


# krimson VM specifications

minor_opcodes: dict[int] = {
    0: VM.add,
    1: VM.sub,
    2: VM.mlt,
    3: VM.div,
    4: VM.mod,
    5: VM.neg,
    6: VM.inc,
    7: VM.dec,
    8: VM.lsh,
    9: VM.rsh,
    10: VM.band,
    11: VM.bor,
    12: VM.bxor,
    13: VM.bnot,
    14: VM.load,
    15: VM.store,
    16: VM.imm_acc,
    17: VM.imm_b,
    18: VM.push_acc,
    19: VM.pop_acc,
    20: VM.push_b,
    21: VM.pop_b,
    22: VM.push_bp,
    23: VM.pop_bp,
    24: VM.call,
    25: VM.ret,
    26: VM.halt,
}

major_opcodes: dict[int] = {
    0: 0,    # nop
    1: VM.branch,
    2: VM.input,
    3: VM.output,
}

conditions: dict[int] = {
    0: lambda a, bits: a == 0,                          # A == B, aka: A - B == 0
    1: lambda a, bits: a != 0,                          # A != B, aka: A - B != 0
    2: lambda a, bits: 0 < a < (1 << (bits-1)),         # A > B,  aka: A - B > 0
    3: lambda a, bits: a < (1 << (bits-1)),             # A >= B, aka: A - B >= 0
    4: lambda a, bits: a >= (1 << (bits-1)),            # A < B,  aka: A - B < 0
    5: lambda a, bits: a > (1 << (bits-1)) or a == 0,   # A <= B, aka: A - B <= 0
    6: lambda a, bits: False,
    7: lambda a, bits: True,
}

io_channels: dict[int] = {
    0: lambda a, bits: str(a),
    1: lambda a, bits: str(twos_comp_to_int(a, bits)),
    2: lambda a, bits: chr(a),
    3: lambda a, bits: str(bool(a)),
    4: lambda a, bits: str(a),
}

# auxiliary functions

def twos_comp_to_int(a: int, n: int) -> int:
    if a >= 1 << (n - 1):
        return a - (1 << n)
    else:
        return a


# main

def main():
    program = fibb
    vm = VM()
    bytecode = vm.bytecode(program)
    vm.run(bytecode, debug=False)
    return


if __name__ == '__main__':
    main()