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
        self.a = 0
        self.b = 0
        self.running = False

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
            instruction = self.rom[self.pc]

            # decode
            self.decode(instruction)

            self.pc += 1

        return

    def decode(self, instruction: int) -> None:
        major = instruction >> 6
        operand = instruction & 63
        if major == 0:
            minor_opcodes[operand](self)
        else:
            major_opcodes[major](self, operand)

    # utils
    def push(self, value: int) -> None:
        assert self.sp > 0
        self.ram[self.sp] = value
        self.sp -= 1

    def pop(self) -> int:
        assert self.sp > 0
        self.sp += 1
        return self.ram[self.sp]

    ### 1 operand instructions

    def arg_load(self, operand: int):
        arg_address = self.bp + operand
        self.push(self.ram[arg_address])

    def arg_store(self, operand: int):
        arg_address = self.bp + operand
        self.ram[arg_address] = self.pop()

    def local_load(self, operand: int):
        local_address = self.bp - operand
        self.push(self.ram[local_address])

    def local_store(self, operand: int):
        local_address = self.bp - operand
        self.ram[local_address] = self.pop()

    def branch(self, operand: int = 7):
        assert 0 <= operand < 8
        cnd = conditions[operand]
        if cnd:
            self.pc = self.pop() - 1    # at the end of the cycle the pc increments, so contracting that
        return

    def imm(self, operand: int) -> None:
        self.pc += 1
        imm = self.rom[self.pc]
        self.push(imm)

    # Calling
    def call(self):
        self.push(self.pc + 1)
        self.branch()

    def ret(self):
        self.arg_load(0)
        self.branch()

    ### 0 operand instructions

    # memory operations
    def load(self) -> None:
        self.a = self.pop()
        self.push(self.ram[self.a])

    def store(self) -> None:
        self.b = self.pop()
        self.a = self.pop()
        self.ram[self.a] = self.b

    # arithmetic operations
    def add(self) -> None:
        self.b = self.pop()
        self.a = self.pop()
        self.push(self.a + self.b)

    def sub(self) -> None:
        self.b = self.pop()
        self.a = self.pop()
        self.push(self.a - self.b)

    def mlt(self) -> None:
        self.b = self.pop()
        self.a = self.pop()
        self.push(self.a * self.b)

    def div(self) -> None:
        self.b = self.pop()
        self.a = self.pop()
        self.push(self.a // self.b)

    def mod(self) -> None:
        self.b = self.pop()
        self.a = self.pop()
        self.push(self.a % self.b)

    def neg(self) -> None:
        a = self.pop()
        self.push(-a)

    # bitwise operations
    def lsh(self) -> None:
        self.b = self.pop()
        self.a = self.pop()
        self.push(self.a << self.b)

    def rsh(self) -> None:
        self.b = self.pop()
        self.a = self.pop()
        self.push(self.a >> self.b)

    def band(self) -> None:
        self.b = self.pop()
        self.a = self.pop()
        self.push(self.a & self.b)

    def bor(self) -> None:
        self.b = self.pop()
        self.a = self.pop()
        self.push(self.a | self.b)

    def bxor(self) -> None:
        self.b = self.pop()
        self.a = self.pop()
        self.push(self.a ^ self.b)

    def bnot(self) -> None:
        self.a = self.pop()
        self.push(~self.a)

    # I/O
    def input(self) -> None:
        i = input()
        try:
            i = int(i)
        except:
            i = 0
        self.push(i)

    def output(self) -> None:
        print(self.pop())

    def __repr__(self):
        return f'pc: {self.pc}\nsp: {self.sp}\nbp: {self.bp}\nreg a: {self.a}\nreg b: {self.b}\n\nram: {self.ram}'


# krimson VM specifications

minor_opcodes: dict[int] = {
    0: VM.add,
    1: VM.sub,
    2: VM.mlt,
    3: VM.div,
    4: VM.mod,
    5: VM.lsh,
    6: VM.rsh,
    7: VM.neg,
    8: VM.band,
    9: VM.bor,
    10: VM.bxor,
    11: VM.bnot,
    12: VM.load,
    13: VM.store,
    14: VM.input,
    15: VM.output,
}

major_opcodes: dict[int] = {
    0: 0,    # nop
    1: VM.imm,
    2: VM.call,
    3: VM.branch,
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


def main():
    program = []
    vm = VM()
    vm.run(program)
    return


if __name__ == '__main__':
    main()