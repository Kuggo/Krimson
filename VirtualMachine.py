
# krimson VM specifications

instruction_size = 8


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
    def __init__(self, data_size=8):
        self.rom: ROM = ROM(instruction_size)
        self.data_size = data_size
        self.ram = RAM(data_size)
        self.pc = 0
        self.sp = (1 << data_size) - 1
        self.bp = self.sp
        self.a = 0
        self.b = 0
        self.running = False

    def run(self, program: list[int], debug=False):
        self.rom.set_mem(program)

        if debug:
            print('Starting Krimson VM with the following specifications',
                  f'Instruction size: {instruction_size}',
                  f'Data size: {self.data_size}',
                  f'ROM: {self.rom}\n')

        self.running = True
        while self.running:
            if debug:
                print(self)

            # fetch
            instruction = self.rom[self.pc]

            # decode

            # execute

            self.pc += 1

        return

    def __repr__(self):
        return f'pc: {self.pc}\nsp: {self.sp}\nbp: {self.bp}\nreg a: {self.a}\nreg b: {self.b}\n\nram: {self.ram}'
