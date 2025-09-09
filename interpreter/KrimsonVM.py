import numpy as np


u16 = np.uint16

INST_SIZE = 16
WORD_SIZE = 16


class ROM:
    def __init__(self, addr_size=WORD_SIZE):
        assert addr_size > 0
        self.mask = u16((1 << addr_size) - 1)
        self.memory = np.zeros(1 << addr_size, dtype=u16)
        return

    def reset_memory(self):
        self.memory &= 0

    def set_mem(self, program: list[int]) -> None:
        self.reset_memory()
        if len(program) > len(self.memory):
            raise Exception(f"Program exceeds memory size ({len(program)} > {len(self.memory)})")
        for i, inst in enumerate(program):
            self.memory[i] = inst
        return

    def __getitem__(self, item: u16) -> u16:
        return self.memory[item & self.mask]


class RAM(ROM):
    def __init__(self, addr_size=WORD_SIZE):
        assert addr_size > 0
        super().__init__(addr_size)

    def __setitem__(self, key: u16, value: u16):
        self.memory[key & self.mask] = value


class VM:
    def __init__(self):
        self.rom: ROM = ROM(INST_SIZE)
        self.ram: RAM = RAM()
        self.pc: u16 = u16(0)
        self.running: bool = False
        return

    def reset(self):
        self.pc = u16(0)
        self.running = False
        self.ram.reset_memory()
        return

    def run(self, program: list[int], debug=False):
        self.rom.set_mem(program)

        if debug:
            print('Starting Krimson VM with the following specifications:\n'
                  f'Instruction size: {INST_SIZE}\n'
                  f'Data size: {WORD_SIZE}\n'
                  f'ROM size: {len(self.rom.memory)} words\n'
                  f'RAM size: {len(self.ram.memory)} words\n')

        self.running = True
        while self.running:
            inst = self.rom[self.pc]
            opcode = (inst >> u16(12)) & u16(0xF)
            self._execute(opcode)

            if debug:
                print(f'PC: {self.pc:04X} | INST: {inst:04X} | OPCODE: {opcode:X}')

            self.pc = u16(self.pc + 1)

        print(f"Program finished execution at PC: {self.pc:04X}")
        return

    def _execute(self, opcode):
        match opcode:
            case 0b0:  # NOP
                pass
            case 0b1:  # HALT
                self.running = False
            case _:
                raise Exception(f"Unknown opcode {opcode:B}")

def main():
    vm = VM()
    program = [
        0x0000,  # NOP
        0x1000,  # HALT
    ]
    vm.run(program, debug=True)
    return

if __name__ == '__main__':
    main()