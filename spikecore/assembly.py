"""SpikeCore instruction set, assembler, and disassembler.

SpikeCore ISA (5 opcodes):
  ACC  core_id  weight_bank  input_range   — weighted accumulate into core's accumulator
  FIRE core_id  threshold                  — if accumulator >= threshold, emit spike, reset
  LEAK core_id  decay_factor               — multiply membrane potential by decay (fixed-point)
  NOP                                      — no operation (pipeline bubble)
  HALT                                     — stop execution

Text format example:
  ACC  core_3  weight_bank_0  spike_in_[0:8]
  FIRE core_3  threshold_64
  LEAK core_3  decay_240
"""

from __future__ import annotations

import enum
import re
from dataclasses import dataclass, field


class Opcode(enum.IntEnum):
    ACC = 0
    FIRE = 1
    LEAK = 2
    NOP = 3
    HALT = 4


@dataclass(frozen=True)
class Instruction:
    """A single SpikeCore instruction."""

    opcode: Opcode
    core_id: int = 0
    operands: tuple[int, ...] = field(default_factory=tuple)

    def __repr__(self) -> str:
        return disassemble_one(self)


def disassemble_one(inst: Instruction) -> str:
    """Convert one instruction to text."""
    op = inst.opcode.name
    if inst.opcode == Opcode.ACC:
        bank, start, end = inst.operands
        return f"ACC  core_{inst.core_id}  weight_bank_{bank}  spike_in_[{start}:{end}]"
    if inst.opcode == Opcode.FIRE:
        threshold = inst.operands[0]
        shift = inst.operands[1] if len(inst.operands) > 1 else 0
        s = f"FIRE core_{inst.core_id}  threshold_{threshold}"
        if shift > 0:
            s += f"  shift_{shift}"
        return s
    if inst.opcode == Opcode.LEAK:
        (decay,) = inst.operands
        return f"LEAK core_{inst.core_id}  decay_{decay}"
    if inst.opcode == Opcode.NOP:
        return "NOP"
    if inst.opcode == Opcode.HALT:
        return "HALT"
    return f"{op} core_{inst.core_id} {inst.operands}"


def disassemble(program: list[Instruction]) -> str:
    """Convert a program to text assembly listing."""
    lines: list[str] = []
    for i, inst in enumerate(program):
        lines.append(f"  {i:04d}: {disassemble_one(inst)}")
    return "\n".join(lines)


# ---- Assembler (text → instructions) ----

_RE_ACC = re.compile(
    r"ACC\s+core_(\d+)\s+weight_bank_(\d+)\s+spike_in_\[(\d+):(\d+)\]",
    re.IGNORECASE,
)
_RE_FIRE = re.compile(r"FIRE\s+core_(\d+)\s+threshold_(\d+)(?:\s+shift_(\d+))?", re.IGNORECASE)
_RE_LEAK = re.compile(r"LEAK\s+core_(\d+)\s+decay_(\d+)", re.IGNORECASE)


def _parse_line(line: str) -> Instruction | None:
    line = line.strip()
    if not line or line.startswith("#") or line.startswith("//"):
        return None
    # Strip address prefix like "0000:"
    if re.match(r"\d{4}:", line):
        line = line[5:].strip()

    upper = line.upper().strip()
    if upper == "NOP":
        return Instruction(Opcode.NOP)
    if upper == "HALT":
        return Instruction(Opcode.HALT)

    m = _RE_ACC.match(line)
    if m:
        core, bank, start, end = int(m[1]), int(m[2]), int(m[3]), int(m[4])
        return Instruction(Opcode.ACC, core, (bank, start, end))

    m = _RE_FIRE.match(line)
    if m:
        core, thresh = int(m[1]), int(m[2])
        shift = int(m[3]) if m[3] else 0
        ops = (thresh, shift) if shift > 0 else (thresh,)
        return Instruction(Opcode.FIRE, core, ops)

    m = _RE_LEAK.match(line)
    if m:
        core, decay = int(m[1]), int(m[2])
        return Instruction(Opcode.LEAK, core, (decay,))

    raise ValueError(f"Cannot parse instruction: {line!r}")


def assemble(text: str) -> list[Instruction]:
    """Parse text assembly into a list of Instructions."""
    program: list[Instruction] = []
    for lineno, line in enumerate(text.splitlines(), 1):
        try:
            inst = _parse_line(line)
            if inst is not None:
                program.append(inst)
        except ValueError as e:
            raise ValueError(f"Line {lineno}: {e}") from e
    return program
