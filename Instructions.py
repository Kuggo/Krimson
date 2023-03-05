from Constants import *


class Instruction:
    def __init__(self, opcode: str, operand=None, imm=None):
        self.opcode: str = opcode
        self.operand: Optional[str] = operand
        self.imm: Optional[str] = imm
        return

    def __repr__(self):
        string = self.opcode
        if self.operand is not None:
            string += f' {self.operand}'

        if self.imm is not None:
            string += f', {self.imm}'

        return string


class Operations(Enum):
    load_arg = Instruction('load_arg')
    store_arg = Instruction('store_arg')
    load_local = Instruction('load_local')
    store_local = Instruction('store_local')
    load_ram = Instruction('load')
    store_ram = Instruction('store')


class Registers(Enum):
    BP = Token(TT.IDENTIFIER, 'bp')
    SP = Token(TT.IDENTIFIER, 'sp')
    PC = Token(TT.IDENTIFIER, 'pc')


    
