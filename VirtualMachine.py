from Instructions import *


class ROM:
    def __init__(self, addr_size=8):
        assert addr_size > 0
        self.address_size = addr_size
        self.memory = [0] * (1 << addr_size)
        self.mask = (1 << addr_size) - 1

    def set_mem(self, program: list[int]) -> None:
        self.memory = [0] * (1 << self.address_size)
        if len(program) > len(self.memory):
            program = program[:len(self.memory)+1]
        for i, byte in enumerate(program):
            self.memory[i] = byte
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
    def __init__(self, instruction_size=8, data_size=8):
        self.instruction_size = instruction_size
        self.rom: ROM = ROM(instruction_size)
        self.data_size = data_size
        self.ram = RAM(data_size)
        self.pc = 0
        self.sp = (1 << data_size) - 1  # stack starts at the top and grows downwards
        self.bp = self.sp
        self.acc = 0
        self.b = 0
        self.running = False
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
            self.decode(instruction)
            continue

        return

    def fetch(self) -> int:
        self.pc += 1
        return self.rom[self.pc]

    def decode(self, instruction: int) -> None:
        major = instruction >> minor_opcode_bits
        operand = instruction & minor_opcodes_bit_mask
        if major == 0:
            minor_opcodes[operand](self)
        else:
            major_opcodes[major](self, operand)

    # utils
    def push(self, val) -> None:
        assert self.sp > 0
        self.ram[self.sp] = val
        self.sp -= 1

    def pop(self) -> int:
        assert self.sp >= 0
        self.sp += 1
        return self.ram[self.sp]

    ### Instructions

    # arithmetic operations
    def add(self) -> None:
        self.acc += self.b

    def sub(self) -> None:
        self.acc = self.b - self.acc

    def mlt(self) -> None:
        self.acc *= self.b

    def div(self) -> None:
        self.acc = self.b // self.acc

    def mod(self) -> None:
        self.acc = self.b % self.acc

    def neg(self) -> None:
        self.acc = -self.acc

    def inc(self) -> None:
        self.acc = self.acc + 1

    def dec(self) -> None:
        self.acc = self.acc - 1

    # bitwise operations
    def lsh(self) -> None:
        self.acc = self.b << self.acc

    def rsh(self) -> None:
        self.acc = self.b >> self.acc

    def band(self) -> None:
        self.acc &= self.b

    def bor(self) -> None:
        self.acc |= self.b

    def bxor(self) -> None:
        self.acc ^= self.b

    def bnot(self) -> None:
        self.acc = ~self.acc

    # memory operations
    def load(self) -> None:
        self.acc = self.ram[self.b]

    def store(self) -> None:
        self.ram[self.b] = self.acc

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
        self.push(self.acc)

    def pop_acc(self) -> None:
        self.acc = self.pop()

    def push_b(self) -> None:
        self.push(self.b)

    def pop_b(self) -> None:
        self.b = self.pop()

    def push_bp(self) -> None:
        self.push(self.bp)

    def pop_bp(self) -> None:
        self.bp = self.pop()

    # Branches
    def branch(self, operand: int = 7):
        assert 0 <= operand < 8
        cnd = conditions[operand]
        if cnd:
            self.pc = self.acc
        return

    # I/O
    # TODO based of which operand, outputting the integer as char/uint/int/frac
    def input(self, operand: int = 0) -> None:
        # operand is quite useless here unless i dont accept the input unless of the specified type
        i = input()
        try:
            i = int(i)
        except:
            i = 0
        self.acc = i

    def output(self, operand: int = 0) -> None:
        print(self.acc)

    def __repr__(self):
        return f'pc: {self.pc}\nsp: {self.sp}\nbp: {self.bp}\nreg a: {self.acc}\nreg b: {self.b}\n\nram: {self.ram}'


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
}

major_opcodes: dict[int] = {
    0: 0,    # nop
    1: VM.branch,
    2: VM.input,
    3: VM.output,
}

conditions: dict[int] = {
    0: lambda a, b: a == b,
    1: lambda a, b: a != b,
    2: lambda a, b: a > b,
    3: lambda a, b: a >= b,
    4: lambda a, b: a < b,
    5: lambda a, b: a <= b,
    6: lambda a, b: False,
    7: lambda a, b: True,
}

io_channels: dict[int] = {
}


def main():
    program = []
    vm = VM()
    vm.run(program)
    return


if __name__ == '__main__':
    main()