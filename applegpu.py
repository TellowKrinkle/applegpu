import fma
import os

from srgb import SRGB_TABLE

MAX_OPCODE_LEN = 16

ABS_FLAG = 'abs'
NEGATE_FLAG = 'neg'
SIGN_EXTEND_FLAG = 'sx'
CACHE_FLAG = 'cache'
DISCARD_FLAG = 'discard'

OPERAND_FLAGS = [
	ABS_FLAG,
	NEGATE_FLAG,
	SIGN_EXTEND_FLAG,
	CACHE_FLAG,
	DISCARD_FLAG,
]

CACHE_HINT = '$'

SR_NAMES = {
}

def opcode_to_number(opcode):
	n = 0
	for i, c in enumerate(opcode[:MAX_OPCODE_LEN]):
		n |= c << (8 * i)
	return n

def sign_extend(v, bits):
	v &= (1 << bits) - 1
	if v & (1 << (bits-1)):
		v |= (-1 << bits)
	return v

class Operand:
	pass

def _add_flags(base, flags):
	parts = [base]
	for i in flags:
		if 'APPLEGPU_CRYPTIC' in os.environ and i == 'cache':
			parts[0] = CACHE_HINT + parts[0]
		else:
			parts.append(i)
	return '.'.join(parts)


class RegisterTuple(Operand):
	def __init__(self, registers, flags=None):
		self.registers = list(registers)
		if flags is None:
			self.flags = []
		else:
			self.flags = list(flags)

	def __str__(self):
		return _add_flags('_'.join(map(str, self.registers)), self.flags)

	def __repr__(self):
		return 'RegisterTuple(%r)' % self.registers

	def __repr__(self):
		return 'RegisterTuple(%r)' % self.registers

	def __getitem__(self, i):
		return self.registers[i]

	def get_with_flags(self, i):
		r = self[i]
		return r.__class__(r.n, flags=self.flags)

	def __len__(self):
		return len(self.registers)

	def get_bit_size(self):
		raise NotImplementedError('get_bit_size')

	def set_thread(self, corestate, thread, result):
		raise NotImplementedError('set_thread')

	def get_thread(self, corestate, thread):
		raise NotImplementedError('get_thread')

class Immediate(Operand):
	# TODO: how should we handle bit_size?
	def __init__(self, value, bit_size=16, flags=None):
		self.value = value
		self._bit_size = bit_size
		if flags is None:
			self.flags = []
		else:
			self.flags = list(flags)

	def get_bit_size(self):
		return self._bit_size

	def get_thread(self, corestate, thread):
		return self.value

	def __str__(self):
		return '.'.join([str(self.value)] + self.flags)

	def __repr__(self):
		if self.flags:
			return 'Immediate(%r, flags=%r)' % (self.value, self.flags)
		return 'Immediate(%r)' % self.value

class RelativeOffset(Immediate):
	def __str__(self):
		base = getattr(self, 'base', None)
		if base is not None:
			v = '0x%X' % (base + self.value,)
		elif self.value >= 0:
			v = 'pc+%d' % (self.value,)
		else:
			v = 'pc-%d' % (-self.value,)
		return '.'.join([v] + self.flags)

	def __repr__(self):
		if self.flags:
			return 'RelativeOffset(%r, flags=%r)' % (self.value, self.flags)
		return 'RelativeOffset(%r)' % self.value

class Register(Operand):
	def __init__(self, n, flags=None):
		self.n = n
		if flags is None:
			self.flags = []
		else:
			self.flags = list(flags)

	def _str(self, names):
		return _add_flags(names[self.n], self.flags)

	def _repr(self, clsname):
		if self.flags:
			return '%s(%d, flags=%r)' % (clsname, self.n, self.flags)
		return '%s(%d)' % (clsname, self.n)

class BaseReg(Register):
	pass

class Reg16(BaseReg):
	def __str__(self):
		return self._str(reg16_names)

	def __repr__(self):
		return self._repr('Reg16')

	def get_bit_size(self):
		return 16

	def set_thread(self, corestate, thread, result):
		corestate.set_reg16(self.n, thread, result)

	def get_thread(self, corestate, thread):
		return corestate.get_reg16(self.n, thread)

class Reg32(BaseReg):
	def __str__(self):
		return self._str(reg32_names)

	def __repr__(self):
		return self._repr('Reg32')

	def get_bit_size(self):
		return 32

	def set_thread(self, corestate, thread, result):
		corestate.set_reg32(self.n, thread, result)

	def get_thread(self, corestate, thread):
		return corestate.get_reg32(self.n, thread)

class Reg64(BaseReg):
	def __str__(self):
		return self._str(reg64_names)

	def __repr__(self):
		return self._repr('Reg64')

	def get_bit_size(self):
		return 64

	def set_thread(self, corestate, thread, result):
		corestate.set_reg64(self.n, thread, result)

	def get_thread(self, corestate, thread):
		return corestate.get_reg64(self.n, thread)

class BaseUReg(Register):
	pass

class UReg16(BaseUReg):
	def __str__(self):
		return self._str(ureg16_names)

	def __repr__(self):
		return self._repr('UReg16')

	def get_thread(self, corestate, thread):
		return corestate.uniforms.get_reg16(self.n)

	def get_bit_size(self):
		return 16

class UReg32(BaseUReg):
	def __str__(self):
		return self._str(ureg32_names)

	def __repr__(self):
		return self._repr('UReg32')

	def get_thread(self, corestate, thread):
		return corestate.uniforms.get_reg32(self.n)

	def get_bit_size(self):
		return 32

class UReg64(BaseUReg):
	def __str__(self):
		return self._str(ureg64_names)

	def __repr__(self):
		return self._repr('UReg64')

	def get_thread(self, corestate, thread):
		return corestate.uniforms.get_reg64(self.n)

	def get_bit_size(self):
		return 64

class SReg32(Register):
	def __str__(self):
		name = 'sr%d' % (self.n)
		if self.n in SR_NAMES:
			name += ' (' + SR_NAMES[self.n] + ')'
		return name

	def __repr__(self):
		return self._repr('SReg32')

	def get_bit_size(self):
		return 32

class TextureState(Register):
	def __str__(self):
		return 'ts%d' % (self.n)

	def __repr__(self):
		return self._repr('TextureState')

	def get_bit_size(self):
		return 32 # ?

class SamplerState(Register):
	def __str__(self):
		return 'ss%d' % (self.n)

	def __repr__(self):
		return self._repr('SamplerState')

	def get_bit_size(self):
		return 32 # ?

class CF(Register):
	def __str__(self):
		return 'cf%d' % (self.n)

	def __repr__(self):
		return self._repr('CF')

	def get_bit_size(self):
		return 32 # ?

ureg16_names = []
ureg32_names = []
ureg64_names = []

for _i in range(256):
	ureg16_names.append('u%dl' % _i)
	ureg16_names.append('u%dh' % _i)
	ureg32_names.append('u%d' % _i)
	ureg64_names.append('u%d_u%d' % (_i, _i + 1))

reg16_names = []
reg32_names = []
reg64_names = []
reg96_names = []
reg128_names = []
for _i in range(128):
	reg16_names.append('r%dl' % _i)
	reg16_names.append('r%dh' % _i)
	reg32_names.append('r%d' % _i)
	# TODO: limit? can cross r31-r32 boundary?
	reg64_names.append('r%d_r%d' % (_i, _i + 1))
	reg96_names.append('r%d_r%d_r%d' % (_i, _i + 1, _i + 2))
	reg128_names.append('r%d_r%d_r%d_r%d' % (_i, _i + 1, _i + 2, _i + 3))


# TODO: is this the right number?
ts_names = []
ss_names = []
cf_names = []
for _i in range(256):
	ts_names.append('ts%d' % _i)
	ss_names.append('ss%d' % _i)
	cf_names.append('cf%d' % _i)



registers_by_name = {}

for _namelist, _c in [
	(reg16_names, Reg16),
	(reg32_names, Reg32),
	(reg64_names, Reg64),
	(ureg16_names, UReg16),
	(ureg32_names, UReg32),
	(ureg64_names, UReg64),
	(ts_names, TextureState),
	(ss_names, SamplerState),
	(cf_names, CF),
]:
	for _i, _name in enumerate(_namelist):
		registers_by_name[_name] = (_c, _i)


def try_parse_register(s):
	flags = []
	if s.startswith(CACHE_HINT):
		s = s[1:]
		flags.append(CACHE_FLAG)
	parts = s.split('.')
	if parts[0] not in registers_by_name:
		return None
	for i in parts[1:]:
		if i not in OPERAND_FLAGS:
			return None
		flags.append(i)

	c, n = registers_by_name[parts[0]]

	return c(n, flags=flags)

def try_parse_register_tuple(s):
	flags = []
	if s.startswith(CACHE_HINT):
		s = s[1:]
		flags.append(CACHE_FLAG)
	parts = s.split('.')
	regs = [try_parse_register(i) for i in parts[0].split('_')]
	if not all(isinstance(r, Reg32) for r in regs) and not all(isinstance(r, Reg16) for r in regs):
		return None
	if any(i.flags for i in regs):
		return None
	if not all(regs[i].n + 1 == regs[i+1].n for i in range(len(regs)-1)):
		return None

	for i in parts[1:]:
		if i not in OPERAND_FLAGS:
			return None
		flags.append(i)

	return RegisterTuple(regs, flags=flags)

SIMD_WIDTH = 32

class AsmInstruction:
	def __init__(self, mnem, operands=None):
		self.mnem = mnem
		self.operands = list(operands)

	def __str__(self):
		operands = ', '.join(filter(None, (str(i) for i in self.operands)))
		return self.mnem.ljust(16) + ' ' + operands

	def __repr__(self):
		return 'AsmInstruction(%r, %r)' % (self.mnem, self.operands)

class AddressSpace:
	def __init__(self):
		self.mappings = []

	def map(self, address, size):
		# TODO: check for overlap
		self.mappings.append((address, [0] * size))

	def set_byte(self, address, value):
		for start, values in self.mappings:
			if start < address and address - start < len(values):
				values[address - start] = value
				return
		assert False, 'bad address %x' % address

	def get_byte(self, address):
		for start, values in self.mappings:
			if start < address and address - start < len(values):
				return values[address - start]
		assert False, 'bad address %x' % address

	def get_u16(self, address):
		return self.get_byte(address) | (self.get_byte(address + 1) << 8)

	def get_u32(self, address):
		return self.get_u16(address) | (self.get_u16(address + 2) << 16)

class Uniforms:
	def __init__(self):
		self.reg16s = [0] * 256

	def get_reg16(self, regid):
		return self.reg16s[regid]

	def set_reg32(self, regid, value):
		self.reg16s[regid * 2] = value & 0xFFFF
		self.reg16s[regid * 2 + 1] = (value >> 16) & 0xFFFF

	def get_reg32(self, regid):
		return self.reg16s[regid * 2] | (self.reg16s[regid * 2 + 1] << 16)

	def get_reg64(self, regid):
		return self.get_reg32(regid) | (self.get_reg32(regid + 1) << 32)

	def set_reg64(self, regid, value):
		self.set_reg32(regid, value & 0xFFFFFFFF)
		self.set_reg32(regid + 1, (value >> 32) & 0xFFFFFFFF)

class CoreState:
	def __init__(self, num_registers=8, uniforms=None, device_memory=None):
		self.reg16s = [[0] * SIMD_WIDTH for i in range(num_registers * 2)]
		self.pc = 0
		self.exec = [True] * SIMD_WIDTH
		if uniforms is None:
			uniforms = Uniforms()
		self.uniforms = uniforms
		self.device_memory = device_memory

	def get_reg16(self, regid, thread):
		return self.reg16s[regid][thread]

	def set_reg16(self, regid, thread, value):
		self.reg16s[regid][thread] = value & 0xFFFF

	def get_reg32(self, regid, thread):
		return self.reg16s[regid * 2][thread] | (self.reg16s[regid * 2 + 1][thread] << 16)

	def set_reg32(self, regid, thread, value):
		self.reg16s[regid * 2][thread] = value & 0xFFFF
		self.reg16s[regid * 2 + 1][thread] = (value >> 16) & 0xFFFF

	def get_reg64(self, regid, thread):
		return self.get_reg32(regid, thread) | (self.get_reg32(regid + 1, thread) << 32)

	def set_reg64(self, regid, thread, value):
		self.set_reg32(regid, thread, value & 0xFFFFFFFF)
		self.set_reg32(regid + 1, thread, (value >> 32) & 0xFFFFFFFF)


class InstructionDesc:
	documentation_skip = False
	def __init__(self, name, size=2, length_bit_pos=16):
		self.name = name

		self.mask = 0
		self.bits = 0
		self.fields = []
		self.ordered_operands = []
		self.operands = {}
		self.constants = []

		self.merged_fields = []

		self.fields_mask = 0

		assert isinstance(size, (int, tuple))
		self.sizes = (size, size) if isinstance(size, int) else size
		self.length_bit_pos = length_bit_pos


	def matches(self, instr):
		instr = self.mask_instr(instr)
		return (instr & self.mask) == (self.bits & self.mask)

	def add_raw_field(self, start, size, name):
		# collision check
		mask = ((1 << size) - 1) << start
		assert (self.mask & mask) == 0, name
		assert (self.fields_mask & mask) == 0
		for _, _, existing_name in self.fields:
			assert existing_name != name, name

		self.fields_mask |= mask
		self.fields.append((start, size, name))

	def add_merged_field(self, name, subfields):
		pairs = []
		shift = 0
		for start, size, subname in subfields:
			self.add_raw_field(start, size, subname)
			pairs.append((subname, shift))
			shift += size
		self.merged_fields.append((name, pairs))

	def add_field(self, start, size, name):
		self.add_merged_field(name, [(start, size, name)])

	def add_suboperand(self, operand):
		# a "suboperand" is an operand which does not appear in the operand list,
		# but is used by other operands. currently unused.
		for start, size, name in operand.fields:
			self.add_field(start, size, name)
		for name, subfields in operand.merged_fields:
			self.add_merged_field(name, subfields)
		self.operands[operand.name] = operand

	def add_operand(self, operand):
		self.add_suboperand(operand)
		self.ordered_operands.append(operand)

	def add_constant(self, start, size, value):
		mask = (1 << size) - 1
		assert (value & ~mask) == 0
		assert (self.mask & (mask << start)) == 0
		self.mask |= mask << start
		self.bits |= value << start

		self.constants.append((start, size, value))

	def add_unsure_constant(self, start, size, value):
		mask = (1 << size) - 1
		assert (value & ~mask) == 0
		assert (self.mask & (mask << start)) == 0
		self.bits |= value << start

	def decode_raw_fields(self, instr):
		instr = self.mask_instr(instr)
		assert self.matches(instr)
		fields = []
		for start, size, name in self.fields:
			fields.append((name, (instr >> start) & ((1 << size) - 1)))
		return fields

	def decode_remainder(self, instr):
		instr = self.mask_instr(instr)
		assert self.matches(instr)
		instr &= ~self.mask
		for start, size, name in self.fields:
			instr &= ~(((1 << size) - 1) << start)
		if self.sizes[0] != self.sizes[1]:
			instr &= ~(1 << self.length_bit_pos)
		return instr

	def patch_raw_fields(self, encoded, fields):
		lookup = {name: (start, size) for start, size, name in self.fields}
		for name, value in fields.items():
			start, size = lookup[name]
			mask = (1 << size) - 1
			assert (value & ~mask) == 0
			encoded = (encoded & ~(mask << start)) | (value << start)

		if self.sizes[0] != self.sizes[1]:
			encoded &= ~(1 << self.length_bit_pos)
			if (encoded & (0xFFFF << (self.sizes[0] * 8))) != 0:
				# use long encoding
				encoded |= (1 << self.length_bit_pos)

		assert self.matches(encoded)
		encoded = self.mask_instr(encoded)

		return encoded

	def encode_raw_fields(self, fields):
		assert sorted(lookup.keys()) == sorted(name for start, size, name in self.fields)
		return self.patch_raw_fields(self.bits, fields)

	def patch_fields(self, encoded, fields):
		mf_lookup = dict(self.merged_fields)
		size_lookup = {name: size for start, size, name in self.fields}

		raw_fields = {}
		for name, value in fields.items():
			for subname, shift in mf_lookup[name]:
				mask = (1 << size_lookup[subname]) - 1
				raw_fields[subname] = (value >> shift) & mask

		return self.patch_raw_fields(encoded, raw_fields)

	def encode_fields(self, fields):
		if sorted(fields.keys()) != sorted(name for name, subfields in self.merged_fields):
			print(sorted(fields.keys()))
			print(sorted(name for name, subfields in self.merged_fields))
			assert False

		return self.patch_fields(self.bits, fields)

	def to_bytes(self, instr):
		return bytes((instr >> (i*8)) & 0xFF for i in range(self.decode_size(instr)))

	def decode_fields(self, instr):
		raw = dict(self.decode_raw_fields(instr))
		fields = []
		for name, subfields in self.merged_fields:
			value = 0
			for subname, shift in subfields:
				value |= raw[subname] << shift
			fields.append((name, value))
		return fields

	def decode_operands(self, instr):
		instr = self.mask_instr(instr)
		fields = dict(self.decode_fields(instr))
		return self.fields_to_operands(fields)

	def fields_to_operands(self, fields):
		ordered_operands = []
		for o in self.ordered_operands:
			ordered_operands.append(o.decode(fields))
		return ordered_operands

	def decode_size(self, instr):
		return self.sizes[(instr >> self.length_bit_pos) & 1]

	def mask_instr(self, instr):
		return instr & ((1 << (self.decode_size(instr) * 8)) - 1)

	def decode_mnem(self, instr):
		instr = self.mask_instr(instr)
		assert self.matches(instr)
		return self.fields_to_mnem(dict(self.decode_fields(instr)))


	def fields_to_mnem_base(self, fields):
		return self.name
	def fields_to_mnem_suffix(self, fields):
		return ''

	def fields_to_mnem(self, fields):
		return self.fields_to_mnem_base(fields) + self.fields_to_mnem_suffix(fields)

	def map_to_alias(self, mnem, operands):
		return mnem, operands

	def disassemble(self, n, pc=None):
		mnem = self.decode_mnem(n)
		operands = self.decode_operands(n)
		mnem, operands = self.map_to_alias(mnem, operands)
		mask = self.mask | self.fields_mask
		if (self.bits & ~mask) != (n & ~mask):
			mnem += '.todo'
		for operand in operands:
			if isinstance(operand, RelativeOffset):
				operand.base = pc
		return AsmInstruction(mnem, operands)

	def fields_for_mnem(self, mnem, operand_strings):
		if self.name == mnem:
			return {}

	def rewrite_operands_strings(self, mnem, opstrs):
		return opstrs

documentation_operands = []

def document_operand(cls):
	documentation_operands.append(cls)
	return cls

class OperandDesc:
	def __init__(self, name=None):
		self.name = name
		self.fields = []
		self.merged_fields = []

	def add_field(self, start, size, name):
		self.fields.append((start, size, name))

	def add_merged_field(self, name, subfields):
		self.merged_fields.append((name, subfields))

	def decode(self, fields):
		return '<TODO>'

	def get_bit_size(self, fields):
		r = self.decode(fields)
		return r.get_bit_size()

def add_dest_hint_modifier_m3(reg, cache):
	if cache:
		reg.flags.append(CACHE_FLAG)
	return reg

def add_dest_hint_modifier(reg, bits):
	if bits & 1:
		reg.flags.append(CACHE_FLAG)
	return reg

def add_hint_modifier(reg, bits):
	if bits == 0b10:
		reg.flags.append(CACHE_FLAG)
		return reg
	elif bits == 0b11:
		reg.flags.append(DISCARD_FLAG)
		return reg
	else:
		assert bits == 0b01
		return reg

def decode_float_immediate(n):
	sign = -1.0 if n & 0x80 else 1.0
	e = (n & 0x70) >> 4
	f = n & 0xF
	if e == 0:
		return sign * f / 64.0
	else:
		return sign * float(0x10 | f) * (2.0 ** (e - 7))

# okay, this is very lazy
float_immediate_lookup = {str(decode_float_immediate(i)): i for i in range(0x100)}

def add_float_modifier(r, modifier):
	if modifier & 1:
		r.flags.append(ABS_FLAG)
	if modifier & 2:
		r.flags.append(NEGATE_FLAG)
	return r

class WaitDesc(OperandDesc):
	def __init__(self, name, lo, hi=None, use_label=True):
		super().__init__(name)
		self.is_mask = hi is not None
		self.use_label = use_label
		if self.is_mask:
			if hi == lo + 3:
				self.add_field(lo, 6, self.name + 'm')
			else:
				self.add_merged_field(self.name, [
					(lo, 3, self.name + 'ml'),
					(hi, 3, self.name + 'mh')
				])
		else:
			self.add_field(self.name, lo, 3)

	def decode(self, fields):
		if self.is_mask:
			value = fields[self.name + 'm']
		else:
			value = fields[self.name]
			if value:
				value = 1 << (value - 1)
		res = '' if self.use_label else 'none'
		if value:
			res = 'wait ' if self.use_label else ''
			for i in range(6):
				if value & (1 << i):
					res += str(i)
		return res

class AbstractDstOperandDesc(OperandDesc):
	def set_thread(self, fields, corestate, thread, result):
		r = self.decode(fields)
		r.set_thread(corestate, thread, result)

class AbstractSrcOperandDesc(OperandDesc):
	def evaluate_thread(self, fields, corestate, thread):
		r = self.decode(fields)
		return r.get_thread(corestate, thread)

class ImplicitR0LDesc(AbstractDstOperandDesc):
	def __init__(self, name):
		super().__init__(name)
		self.add_field(7, 1, self.name + 't')

	def decode(self, fields):
		flags = fields[self.name + 't']
		r = Reg16(0)
		return add_dest_hint_modifier(r, flags)

	def encode_string(self, fields, opstr):
		reg = try_parse_register(opstr)
		if reg and isinstance(reg, Reg16) and reg.n == 0:
			flags = 0
			if CACHE_FLAG in reg.flags:
				flags |= 1
			fields[self.name + 't'] = flags
			return
		raise Exception('invalid ImplicitR0LDesc %r' % (opstr,))

@document_operand
class ALUDstDesc(AbstractDstOperandDesc):
	def __init__(self, name):
		super().__init__(name)

	def _allow64(self):
		return False

	def _allow32(self):
		return True

	def _paired(self):
		return False

	def _has_cache(self):
		return True

	def _value_shift(self):
		return 0

	def decode(self, fields):
		value = fields[self.name] << self._value_shift()
		# TODO: 64
		size = fields[self.name + 's'] if self._allow32() else 0
		# TODO: What's the default for mov_imm7?
		cache = fields[self.name + 'c'] if self._has_cache() else 0

		if size == 0:
			if self._paired():
				r = RegisterTuple(Reg16(value + i) for i in range(2))
			else:
				r = Reg16(value)
		else:
			if self._paired():
				r = RegisterTuple(Reg32((value >> 1) + i) for i in range(2))
			else:
				r = Reg32(value >> 1)

		return add_dest_hint_modifier_m3(r, cache)

	def encode(self, fields, operand):
		if self._paired() and isinstance(operand, RegisterTuple):
			# TODO: validate
			operand = operand.get_with_flags(0)
		cache = 0
		value = 0
		size = 0
		if isinstance(operand, BaseReg):
			if isinstance(operand, Reg16):
				value = operand.n
			elif isinstance(operand, Reg32):
				if not self._allow32():
					print('WARNING: encoding invalid 32-bit register')
				value = operand.n << 1
				size = 1
			else:
				assert isinstance(operand, Reg64)
				if not self._allow64():
					print('WARNING: encoding invalid 64-bit register')
				value = (operand.n << 1) | 1

			if CACHE_FLAG in operand.flags:
				cache = 1
		else:
			raise Exception('invalid ALUDstDesc %r' % (operand,))

		if self._has_cache():
			fields[self.name + 'c'] = flags
		fields[self.name + 's'] = size
		fields[self.name] = value >> self._value_shift()

	pseudocode = '''
	{name}(value, cache, size, max_size=32):
		if size and value & 1 and max_size >= 64:
			return Reg64Reference(value >> 1, cache=cache)
		elif size and max_size >= 32:
			return Reg32Reference(value >> 1, cache=cache)
		else:
			return Reg16Reference(value, cache=cache)
	'''

	def encode_string(self, fields, opstr):
		if self._paired():
			regs = try_parse_register_tuple(opstr)
			if regs and len(regs) == 2:
				return self.encode(fields, regs)
			raise Exception('invalid paired ALUDstDesc %r' % (opstr,))

		reg = try_parse_register(opstr)
		if reg:
			return self.encode(fields, reg)
		raise Exception('invalid ALUDstDesc %r' % (opstr,))

class ALUDstSDesc(ALUDstDesc):
	def __init__(self, name):
		super().__init__(name)
		self.add_field(4, 4, self.name)
		self.add_field(3, 1, self.name + 's')

	def _has_cache(self):
		return False

	def _value_shift(self):
		return 1

class ALUDstMDesc(ALUDstDesc):
	def __init__(self, name, size_off=3, lo_off=18):
		super().__init__(name)
		self.add_merged_field(self.name, [
			(lo_off, 1, self.name + 'l'),
			( 4, 4, self.name),
			(22, 2, self.name + 'x'),
		])
		self.add_field(size_off, 1, self.name + 's')
		self.add_field(21, 1, self.name + 'c')

class ALUDstLDesc(ALUDstDesc):
	def __init__(self, name, size_off=3, lo_off=18, hi_off=60):
		super().__init__(name)
		self.add_merged_field(self.name, [
			(lo_off, 1, self.name + 'l'),
			( 4, 4, self.name),
			(22, 2, self.name + 'x'),
			(hi_off, 1, self.name + 'h'),
		])
		self.add_field(size_off, 1, self.name + 's')
		self.add_field(21, 1, self.name + 'c')


class PairedALUDstDesc(ALUDstDesc):
	# converts r0 <-> r0_r1 and r0h <-> r0h_r1l
	def _paired(self):
		return True

class ALUDst64LDesc(ALUDstLDesc):
	pseudocode = '''
	{name}(value, flags):
		return ALUDst(value, flags, max_size=64)
	'''
	def _allow64(self):
		return True

class ALUDst16LDesc(ALUDstLDesc):
	pseudocode = '''
	{name}(value, flags):
		return ALUDst(value, flags, max_size=16)
	'''
	def _allow32(self):
		return False

@document_operand
class FloatDstDesc(ALUDstDesc):
	def __init__(self, name, bit_off_ex):
		super().__init__(name, bit_off_ex)
		self.add_field(6, 1, 'S')

	# so far this is the same, but conceptually i'd like the destination to
	# be responsible for converting the result to the correct size, which is
	# a very different operation for floats.
	pseudocode = '''
	{name}(value, flags, saturating, max_size=32):
		destination = ALUDst(value, flags, max_size=max_size)
		if destination.thread_bit_size == 32:
			wrapper = RoundToFloat32Wrapper(destination, flush_to_zero=True)
		else:
			wrapper = RoundToFloat16Wrapper(destination, flush_to_zero=False)

		if saturating:
			wrapper = SaturateRealWrapper(wrapper)

		return wrapper
	'''

@document_operand
class FloatDst16Desc(FloatDstDesc):
	def _allow32(self):
		return False
	pseudocode = '''
	{name}(value, flags, saturating):
		return FloatDst(value, flags, saturating, max_size=16)
	'''

@document_operand
class ALUSrcDesc(AbstractSrcOperandDesc):
	"Zero-extended 16 or 32 bit source field"

	def __init__(self, name, bit_off, bit_off_ex):
		super().__init__(name)
		self.add_merged_field(self.name, [
			(bit_off, 6, self.name),
			(bit_off_ex, 2, self.name + 'x')
		])
		self.add_field(bit_off + 6, self._type_size(), self.name + 't')

	def _type_size(self):
		return 4

	def _allow32(self):
		return True

	def _allow64(self):
		return False

	def _paired(self):
		return False

	def decode_impl(self, fields, allow64):
		flags = fields[self.name + 't']
		value = fields[self.name]
		return bitop_decode(flags, value, allow64)

	def decode_immediate(self, fields, value):
		return Immediate(value)

	def decode(self, fields):
		flags = fields[self.name + 't']
		value = fields[self.name]
		if flags == 0:
			return self.decode_immediate(fields, value)
		elif (flags & 0b1100) == 0b0100:
			value |= (flags & 1) << 8
			if flags & 2:
				return UReg32(value >> 1)
			else:
				return UReg16(value)
		elif (flags & 0b11) != 0: # in (0b0001, 0b0010, 0b0011):
			if self._allow64() and (flags & 0b1100) == 0b1100:
				assert (value & 1) == 0
				assert not self._paired()
				r = Reg64(value >> 1)
			elif self._allow32() and (flags & 0b1100) in (0b1000, 0b1100):
#				assert (value & 1) == 0
				if self._paired():
					r = RegisterTuple(Reg32((value >> 1) + i) for i in range(2))
				else:
					r = Reg32(value >> 1)
			elif (flags & 0b1100) in (0b0000, 0b1000, 0b1100):
				if self._paired():
					r = RegisterTuple(Reg16(value + i) for i in range(2))
				else:
					r = Reg16(value)
			else:
				return
			return add_hint_modifier(r, flags & 0b11)
		else:
			print('TODO: ' + format(flags, '04b'))

	def encode(self, fields, operand):
		if self._paired():
			if isinstance(operand, (Reg16,Reg32)):
				raise Exception('invalid paired operand %r' % (operand,))
			elif isinstance(operand, RegisterTuple):
				# TODO: validate
				operand = operand.get_with_flags(0)
		flags = 0
		value = 0
		if isinstance(operand, (UReg16, UReg32)):
			flags = 0b0100
			if isinstance(operand, UReg32):
				flags |= 2
				value = operand.n << 1
			else:
				value = operand.n
			flags |= (value >> 8) & 1
			value &= 0xFF
		elif isinstance(operand, BaseReg):
			if isinstance(operand, Reg16):
				flags = 0b0000
				value = operand.n
			elif isinstance(operand, Reg32):
				if not self._allow32():
					print('WARNING: encoding invalid 32-bit register')
				flags = 0b1000
				value = operand.n << 1
			else:
				flags = 0b1100
				assert isinstance(operand, Reg64)
				if not self._allow64():
					print('WARNING: encoding invalid 64-bit register')
				value = operand.n << 1

			if CACHE_FLAG in operand.flags:
				#if DISCARD_FLAG not in operand.flags
				flags |= 2
			elif DISCARD_FLAG in operand.flags:
				flags |= 3
			else:
				flags |= 1
		elif isinstance(operand, Immediate):
			flags = 0
			if not 0 <= operand.value < 256:
				raise Exception('out of range immediate %r' % (operand,))
			value = operand.value
		else:
			raise Exception('invalid ALUSrcDesc %r' % (operand,))

		fields[self.name + 't'] = flags
		fields[self.name] = value

	pseudocode = '''
	{name}(value, flags, max_size=32):
		if flags == 0b0000:
			return BroadcastImmediateReference(value)

		if flags >> 2 == 0b01:
			ureg = value | (flags & 1) << 8
			if flags & 0b10:
				if max_size < 32:
					UNDEFINED()
				return BroadcastUReg32Reference(ureg >> 1)
			else:
				return BroadcastUReg16Reference(ureg)

		if flags & 0b11 == 0b00: UNDEFINED()

		cache_flag   = (flags & 0b11) == 0b10
		discard_flag = (flags & 0b11) == 0b11

		if flags >> 2 == 0b11 and max_size >= 64:
			if value & 1: UNDEFINED()
			return Reg64Reference(value >> 1, cache=cache_flag, discard=discard_flag)

		if flags >> 2 >= 0b10 and max_size >= 32:
			if flags >> 2 != 0b10: UNDEFINED()
			if value & 1: UNDEFINED()
			return Reg32Reference(value >> 1, cache=cache_flag, discard=discard_flag)

		if max_size >= 16:
			if flags >> 2 != 0b00: UNDEFINED()
			return Reg16Reference(value, cache=cache_flag, discard=discard_flag)
	'''

	def encode_string(self, fields, opstr):
		if self._paired():
			regs = try_parse_register_tuple(opstr)
			if regs and len(regs) == 2:
				return self.encode(fields, regs)
		else:
			reg = try_parse_register(opstr)
			if reg:
				return self.encode(fields, reg)

		value = try_parse_integer(opstr)

		if value is None:
			raise Exception('invalid ALUSrcDesc %r' % (opstr,))

		self.encode(fields, Immediate(value))

def try_parse_integer(opstr):
	if opstr in float_immediate_lookup:
		return float_immediate_lookup[opstr]

	try:
		base = 10
		if '0b' in opstr:
			base = 2
		elif '0x' in opstr:
			base = 16
		return int(opstr, base)
	except ValueError:
		return None

assert try_parse_integer('11') == 11
assert try_parse_integer('0b11') == 3
assert try_parse_integer('0x11') == 17
assert try_parse_integer('-11') == -11
assert try_parse_integer('-0b11') == -3
assert try_parse_integer('-0x11') == -17

class ALUSrc64Desc(ALUSrcDesc):
	"Zero-extended 16, 32 or 64 bit source field"

	def _allow64(self):
		return True

class ALUSrc16Desc(ALUSrcDesc):
	"Zero-extended 16 bit source field"

	def _allow32(self):
		return False


@document_operand
class MulSrcDesc(ALUSrcDesc):
	"Sign-extendable 16 or 32 bit source field"

	def __init__(self, name, bit_off, bit_off_ex):
		super().__init__(name, bit_off, bit_off_ex)
		self.add_field(bit_off + 10, 1, self.name + 's')


	def decode(self, fields):
		r = super().decode(fields)
		if not isinstance(r, Register):
			return r

		if fields[self.name + 's'] & 1:
			r.flags.append(SIGN_EXTEND_FLAG)
		return r

	def evaluate_thread(self, fields, corestate, thread):
		r = self.decode(fields)

		value = r.get_thread(corestate, thread)
		size = r.get_bit_size()
		if SIGN_EXTEND_FLAG in r.flags:
			value = sign_extend(value, size)
		return value

	def encode(self, fields, operand):
		super().encode(fields, operand)
		sx = 0
		if isinstance(operand, Register):
			if SIGN_EXTEND_FLAG in operand.flags:
				sx = 1
		fields[self.name + 's'] = sx

	pseudocode = '''
	{name}(value, flags, sx):
		source = ALUSrc(value, flags, max_size=32)
		if sx:
			# Note: 8-bit immediates have already been zero-extended to 16-bit,
			# so do not get sign extended.
			return SignExtendWrapper(source, source.thread_bit_size)
		else:
			return source
	'''

@document_operand
class AddSrcDesc(MulSrcDesc):
	"Sign-extendable 16, 32 or 64 bit source field"

	pseudocode = '''
	{name}(value, flags, sx):
		source = ALUSrc(value, flags, max_size=64)
		if sx:
			# Note: 8-bit immediates have already been zero-extended to 16-bit,
			# so do not get sign extended.
			return SignExtendWrapper(source, source.thread_bit_size)
		else:
			return source
	'''

	def _allow64(self):
		return True

@document_operand
class CmpselSrcDesc(AbstractSrcOperandDesc):
	documentation_extra_arguments = ['Dt']

	def __init__(self, name, bit_off, bit_off_ex):
		super().__init__(name)
		self.add_merged_field(self.name, [
			(bit_off, 6, self.name),
			(bit_off_ex, 2, self.name + 'x')
		])
		self.add_field(bit_off + 6, 3, self.name + 't')

	def decode(self, fields):
		flags = fields[self.name + 't']
		value = fields[self.name]

		is32 = ((fields['Dt'] & 2) != 0)

		if flags == 0b100:
			return Immediate(value)
		elif flags in (0b001, 0b010, 0b011):
			if is32:
				assert (value & 1) == 0
				r = Reg32(value >> 1)
			else:
				r = Reg16(value)
			return add_hint_modifier(r, flags & 0b11)
		elif flags in (0b110, 0b111):
			if is32:
				assert (value & 1) == 0
				return UReg32(value >> 1)
			else:
				return UReg16(value)
		else:
			print('TODO: ' + format(flags, '04b'))

	pseudocode = '''
	{name}(value, flags, destination_flags):
		if flags == 0b100:
			return BroadcastImmediateReference(value)

		if flags >> 1 == 0b11:
			ureg = value | (flags & 1) << 8
			if destination_flags & 2:
				if ureg & 1: UNDEFINED()
				return BroadcastUReg32Reference(ureg >> 1)
			else:
				return BroadcastUReg16Reference(ureg)

		if flags >> 2 == 1: UNDEFINED()
		if flags & 0b11 == 0b00: UNDEFINED()

		cache_flag   = (flags & 0b11) == 0b10
		discard_flag = (flags & 0b11) == 0b11

		if destination_flags & 2:
			if value & 1: UNDEFINED()
			return Reg32Reference(value >> 1, cache=cache_flag, discard=discard_flag)
		else:
			return Reg16Reference(value, cache=cache_flag, discard=discard_flag)
	'''
	def encode(self, fields, operand):
		flags = 0
		value = 0
		if isinstance(operand, (UReg16, UReg32)):
			flags = 0b110
			if isinstance(operand, UReg16):
				if fields['Dt'] & 2:
					raise Exception('invalid CmpselSrcDesc (mismatch with dest) %r' % (operand,))
				value = operand.n
			else:
				assert isinstance(operand, UReg32)
				if not fields['Dt'] & 2:
					raise Exception('invalid CmpselSrcDesc (mismatch with dest) %r' % (operand,))
				value = operand.n << 1
			flags |= (value >> 8) & 1
			value &= 0xFF
		elif isinstance(operand, BaseReg):
			if isinstance(operand, Reg16):
				if fields['Dt'] & 2:
					raise Exception('invalid CmpselSrcDesc (mismatch with dest) %r' % (operand,))
				value = operand.n
			elif isinstance(operand, Reg32):
				if not fields['Dt'] & 2:
					raise Exception('invalid CmpselSrcDesc (mismatch with dest) %r' % (operand,))
				value = operand.n << 1
			else:
				raise Exception('invalid CmpselSrcDesc %r' % (operand,))

			if CACHE_FLAG in operand.flags:
				flags = 2
			elif DISCARD_FLAG in operand.flags:
				flags = 3
			else:
				flags = 1
		elif isinstance(operand, Immediate):
			flags = 0b100
			if not 0 <= operand.value < 256:
				raise Exception('out of range immediate %r' % (operand,))
			value = operand.value
		else:
			raise Exception('invalid CmpselSrcDesc %r' % (operand,))

		fields[self.name + 't'] = flags
		fields[self.name] = value

	def encode_string(self, fields, opstr):
		reg = try_parse_register(opstr)
		if reg:
			return self.encode(fields, reg)

		value = try_parse_integer(opstr)
		if value is None:
			raise Exception('invalid CmpselSrcDesc %r' % (opstr,))

		self.encode(fields, Immediate(value))

@document_operand
class FloatSrcDesc(ALUSrcDesc):
	def __init__(self, name, bit_off, bit_off_ex, bit_off_m=None):
		super().__init__(name, bit_off, bit_off_ex)
		if bit_off_m is None:
			bit_off_m = bit_off + 6 + self._type_size()
		self.add_field(bit_off_m, 2, self.name + 'm')

	def decode_immediate(self, fields, value):
		return Immediate(decode_float_immediate(value))

	def decode(self, fields):
		r = super().decode(fields)
		return add_float_modifier(r, fields[self.name + 'm'])

	def evaluate_thread_float(self, fields, corestate, thread):
		o = self.decode(fields)

		if isinstance(o, Immediate):
			r = fma.f64_to_u64(o.value)
		else:
			bits = o.get_thread(corestate, thread)
			bit_size = o.get_bit_size()
			if bit_size == 16:
				r = fma.f16_to_f64(bits, ftz=False)
			elif bit_size == 32:
				r = fma.f32_to_f64(bits, ftz=True)
			else:
				raise NotImplementedError()

		if ABS_FLAG in o.flags:
			r &= ~(1 << 63)
		if NEGATE_FLAG in o.flags:
			r ^= (1 << 63)

		return r


	pseudocode = '''
	{name}(value, flags, modifier, max_size=32):
		source = ALUSrc(value, flags, max_size)

		if source.is_immediate:
			float = BroadcastRealReference(decode_float_immediate(source))

		elif source.thread_bit_size == 16:
			float = Float16ToRealWrapper(source, flush_to_zero=False)
		elif source.thread_bit_size == 32:
			float = Float32ToRealWrapper(source, flush_to_zero=True)

		if modifier & 0b01: float = FloatAbsoluteValueWrapper(float)
		if modifier & 0b10: float = FloatNegateWrapper(float)

		return float
	'''

	def encode(self, fields, operand):
		super().encode(fields, operand)
		m = 0
		if isinstance(operand, Register):
			if ABS_FLAG in operand.flags:
				m |= 1
			if NEGATE_FLAG in operand.flags:
				m |= 2
		fields[self.name + 'm'] = m


class PairedFloatSrcDesc(FloatSrcDesc):
	# converts r0 <-> r0_r1 and r0h <-> r0h_r1l
	# TODO: not clear is uniform registers supported
	def _paired(self):
		return True

helper_pseudocode = '''

decode_float_immediate(value):
	sign = (value & 0x80) >> 7
	exponent = (value & 0x70) >> 4
	fraction = value & 0xF

	if exponent == 0:
		result = fraction / 64.0
	else:
		fraction = 16.0 + fraction
		exponent -= 7
		result = fraction * (2.0 ** exponent)

	if sign != 0:
		result = -result

	return result

'''

@document_operand
class FloatSrc16Desc(FloatSrcDesc):
	pseudocode = '''
	{name}(value, flags, modifier):
		return FloatSrcDesc(value, flags, modifier, max_size=16)
	'''
	def _type_size(self):
		return 3


class TruthTableDesc(OperandDesc):
	documentation_skip = True

	def __init__(self, name):
		super().__init__(name)

		self.add_field(26, 1, self.name + '0')
		self.add_field(27, 1, self.name + '1')
		self.add_field(38, 1, self.name + '2')
		self.add_field(39, 1, self.name + '3')

	def decode(self, fields):
		return ''.join(str(fields[self.name + str(i)]) for i in range(4))

	def encode_string(self, fields, opstr):
		if not all(i in '01' for i in opstr) or len(opstr) != 4:
			raise Exception('invalid TruthTable %r' % (opstr,))
		for i in range(4):
			fields['tt' + str(i)] = int(opstr[i])

class FieldDesc(OperandDesc):
	def __init__(self, name, x, size=None):
		super().__init__(name)
		if isinstance(x, list):
			subfields = x
			self.size = sum(size for start, size, name in subfields)
			self.add_merged_field(self.name, subfields)
		else:
			start = x
			assert isinstance(start, int)
			assert isinstance(size, int)
			self.size = size
			self.add_field(start, size, self.name)

class IntegerFieldDesc(FieldDesc):
	documentation_skip = True # (because it is what it is)

	def _signed(self):
		return False

	def encode_string(self, fields, opstr):
		value = try_parse_integer(opstr)

		if value is None:
			raise Exception('invalid IntegerFieldDesc %r' % (opstr,))

		lo = 0
		hi = 1 << self.size
		if self._signed():
			hi >>= 1
			lo = -hi
		if value < lo or value >= hi:
			value &= (1 << self.size) - 1
			print('WARNING: encoding out of range IntegerFieldDesc %r (0-%d) as %d' % (opstr, (1 << self.size) - 1, value))

		fields[self.name] = value

class BinaryDesc(IntegerFieldDesc):
	def decode(self, fields):
		return '0b' + format(fields[self.name], '0' + str(self.size) + 'b')

class ImmediateDesc(IntegerFieldDesc):
	def decode(self, fields):
		return fields[self.name]

	def evaluate_thread(self, fields, corestate, thread):
		return fields[self.name]

class SignedImmediateDesc(FieldDesc):
	def _signed(self):
		return True

	def decode(self, fields):
		return sign_extend(fields[self.name], self.size)

	def evaluate_thread(self, fields, corestate, thread):
		return self.decode(fields)

@document_operand
class Reg32Desc(FieldDesc):
	pseudocode = '''
	{name}(value):
		return Reg32Reference(value)
	'''
	def decode(self, fields):
		return Reg32(fields[self.name])

	def encode_string(self, fields, opstr):
		reg = try_parse_register(opstr)
		if reg and isinstance(reg, Reg32):
			fields[self.name] = reg.n
		else:
			raise Exception('invalid Reg32Desc %r' % (opstr,))


class EnumDesc(FieldDesc):
	documentation_skip = True

	def __init__(self, name, start, size, values):
		super().__init__(name, start, size)
		self.values = values

	def decode(self, fields):
		v = fields[self.name]
		return self.values.get(v, v)

	def encode_string(self, fields, opstr):
		for k, v in self.values.items():
			if v == opstr:
				fields[self.name] = k
				return

		v = try_parse_integer(opstr)
		if v is not None:
			fields[self.name] = v
			return

		raise Exception('invalid enum %r (%r)' % (opstr, list(self.values.values())))

class ShiftDesc(OperandDesc):
	documentation_no_name = True

	def __init__(self, name):
		super().__init__(name)
		self.add_merged_field('s', [
			(39, 1, 's1'),
			(52, 2, 's2'),
		])

	def decode(self, fields):
		shift = fields['s']
		return 'lsl %d' % (shift) if shift else ''

	def encode_string(self, fields, opstr):
		if opstr == '':
			s = 0
		elif opstr.startswith('lsl '):
			s = try_parse_integer(opstr[4:])
			if s is None:
				raise Exception('invalid ShiftDesc %r' % (opstr,))
		else:
			raise Exception('invalid ShiftDesc %r' % (opstr,))
		fields['s'] = s

class MaskDesc(OperandDesc):
	documentation_no_name = True

	def __init__(self, name):
		super().__init__(name)
		self.add_merged_field(self.name, [
			(38, 2, self.name + '1'),
			(50, 2, self.name + '2'),
			(63, 1, self.name + '3'),
		])

	def decode(self, fields):
		mask = fields[self.name]
		return 'mask 0x%X' % ((1 << mask) - 1) if mask else ''

	def encode_string(self, fields, opstr):
		if opstr == '':
			fields[self.name] = 0
			return

		if opstr.startswith('mask '):
			mask = try_parse_integer(opstr[len('mask '):])
			b = format(mask + 1, 'b')
			if b.count('1') == 1:
				m = len(b) - 1
				if 0 < m <= 32:
					if m == 32:
						m = 0
					fields[self.name] = m
					return

		raise Exception('invalid MaskDesc %r' % (opstr,))


class BranchOffsetDesc(FieldDesc):
	'''Signed offset in bytes from start of jump instruction (must be even)'''

	documentation_skip = True

	def decode(self, fields):
		v = fields[self.name]
		#assert (v & 1) == 0
		v = sign_extend(v, self.size)
		return RelativeOffset(v)

	def encode_string(self, fields, opstr):
		if opstr.startswith('pc'):
			s = opstr.replace(' ','')
			if s.startswith(('pc-', 'pc+')):
				value = try_parse_integer(s[2:])
				if value is not None:
					masked = value & ((1 << self.size) - 1)
					if value != sign_extend(masked, self.size):
						raise Exception('out of range BranchOffsetDesc %r' % (opstr,))
					fields[self.name] = masked
					return

		raise Exception('invalid BranchOffsetDesc %r' % (opstr,))


class StackAdjustmentDesc(OperandDesc):
	# maybe memory index desc?
	documentation_no_name = True

	def __init__(self, name):
		super().__init__(name)
		self.add_merged_field(self.name, [
			(20, 4, self.name + '1'),
			(32, 4, self.name + '2'),
			(56, 8, self.name + '3'),
		])

	def decode(self, fields):
		return Immediate(sign_extend(fields[self.name], 16))

	def encode_string(self, fields, opstr):
		value = try_parse_integer(opstr)

		if value is None:
			raise Exception('invalid StackAdjustmentDesc %r' % (opstr,))

		masked = value & 0xFFFF
		if sign_extend(masked, 16) != value:
			raise Exception('invalid StackAdjustmentDesc %r (out of range)' % (opstr,))

		fields[self.name] = masked

class StackReg32Desc(OperandDesc):
	# TODO: merge logic with Reg32Desc
	def __init__(self, name, parts):
		super().__init__(name)
		self.add_merged_field(self.name, parts)

	def decode(self, fields):
		return Reg32(fields[self.name] >> 1)

	def encode_string(self, fields, opstr):
		reg = try_parse_register(opstr)
		if reg and isinstance(reg, Reg32):
			fields[self.name] = reg.n << 1
		else:
			raise Exception('invalid StackReg32Desc %r' % (opstr,))

class BaseConditionDesc(OperandDesc):
	def __init__(self, cc_off=13, cc_n_off=8):
		super().__init__('cc')

		self.add_field(cc_off, 3, 'cc')
		if cc_n_off is not None:
			self.add_field(cc_n_off, 1, 'ccn')

	def encode_string(self, fields, opstr):
		for k, v in self.encodings.items():
			if v == opstr:
				cond = k
				break
		else:
			raise Exception('invalid condition %r' % (opstr,))
		fields['cc'] = cond & 0b111
		if len(self.fields) == 2:
			fields['ccn'] = (cond >> 3)
		elif cond > 0b111:
			raise Exception('invalid condition %r (no ccn fields)' % (opstr,))

	def decode(self, fields):
		v = fields['cc'] | (fields.get('ccn', 0) << 3)
		return self.encodings.get(v, v)

@document_operand
class IConditionDesc(BaseConditionDesc):
	pseudocode = '''
	{name}(value, n=0):
		sign_extend   = (value & 0b100) != 0
		condition     =  value & 0b011
		invert_result = (n != 0)

		if condition == 0b00:
			return IntEqualityComparison(sign_extend, invert_result)
		if condition == 0b01:
			return IntLessThanComparison(sign_extend, invert_result)
		if condition == 0b10:
			return IntGreaterThanComparison(sign_extend, invert_result)
	'''
	def __init__(self, cc_off=13, cc_n_off=8):
		super().__init__(cc_off, cc_n_off)

		self.encodings = {
			0b0000: 'ueq',
			0b0001: 'ult',
			0b0010: 'ugt',
			0b0100: 'seq',
			0b0101: 'slt',
			0b0110: 'sgt',

			0b1000: 'nueq',
			0b1001: 'ugte',
			0b1010: 'ulte',
			0b1100: 'nseq',
			0b1101: 'sgte',
			0b1110: 'slte',
		}


@document_operand
class FConditionDesc(BaseConditionDesc):
	pseudocode = '''
	{name}(condition, n=0):
		invert_result = (n != 0)

		if condition == 0b000:
			return FloatEqualityComparison(invert_result)
		if condition == 0b001:
			return FloatLessThanComparison(invert_result)
		if condition == 0b010:
			return FloatGreaterThanComparison(invert_result)
		if condition == 0b011:
			return FloatLessThanNanLosesComparison(invert_result)
		if condition == 0b101:
			return FloatLessThanOrEqualComparison(invert_result)
		if condition == 0b110:
			return FloatGreaterThanOrEqualComparison(invert_result)
		if condition == 0b111:
			return FloatGreaterThanNanLosesComparison(invert_result)
	'''

	def __init__(self, cc_off=13, cc_n_off=8):
		super().__init__(cc_off, cc_n_off)

		self.encodings = {
			0b000: 'eq',
			0b001: 'lt',
			0b010: 'gt',
			0b011: 'ltn',
			0b101: 'gte',
			0b110: 'lte',
			0b111: 'gtn',

			0b1000: 'neq',
			0b1001: 'nlt',
			0b1011: 'nltn', # unobserved
			0b1010: 'ngt',
			0b1101: 'ngte',
			0b1110: 'nlte',
			0b1111: 'ngtn', # unobserved
		}



class MemoryShiftDesc(OperandDesc):
	documentation_skip = True

	def __init__(self, name, offset):
		super().__init__(name)
		self.add_field(offset, 3, self.name)

	def decode(self, fields):
		shift = fields[self.name]
		if shift == 0:
			shift = 4
		else:
			shift -= 1
		return 'lsl %d' % shift if shift else ''

	def encode_string(self, fields, opstr):
		if opstr == '':
			s = 0
		elif opstr.startswith('lsl '):
			s = try_parse_integer(opstr[4:])
			if s is None:
				raise Exception('invalid MemoryShiftDesc %r' % (opstr,))
		else:
			raise Exception('invalid MemoryShiftDesc %r' % (opstr,))
		fields[self.name] = 0 if s == 4 else s + 1


@document_operand
class MemoryIndexDesc(OperandDesc):
	pseudocode = '''
	{name}(value, discard, size, sx, u):
		TODO()
	'''

	def __init__(self, name, sx_off, t_off):
		super().__init__(name)
		self.add_merged_field(self.name, [
			(39, 1, self.name + 'l'),
			(40, 7, self.name),
		])
		self.add_field(47, 1, self.name + 'd')
		self.add_field(48, 5, self.name + 'x')
		self.add_field(53, 1, self.name + 's')
		self.add_field(54, 1, self.name + 'h')
		self.add_field(sx_off, 1, self.name + 'sx')
		self.add_field(t_off, 2, self.name + 't')

	def decode_impl(self, fields, allow64):
		value   = fields[self.name]
		t       = fields[self.name + 't']
		discard = fields[self.name + 'd']
		size    = fields[self.name + 's']
		sx      = fields[self.name + 'sx']
		x       = fields[self.name + 'x']
		h       = fields[self.name + 'h']

		assert(t != 3)

		if t == 2:
			if size:
				reg = UReg32(value >> 1)
			else:
				reg = UReg16(value)
		elif t == 1:
			if size:
				reg = Reg32(value >> 1)
			else:
				reg = Reg16(value)
		else:
			return value + (discard << 8) + (x << 9) + (size << 14) + (h << 15)

		if sx:
			reg.flags.append(SIGN_EXTEND_FLAG)
		if discard:
			reg.flags.append(DISCARD_FLAG)

		return reg

	def decode(self, fields):
		return self.decode_impl(fields, allow64=False)

	def encode_string(self, fields, opstr):
		r = try_parse_register(opstr)
		if r is not None:
			if isinstance(r, Reg32):
				fields[self.name + 't'] = 0
				fields[self.name] = r.n << 1
				return

		v = try_parse_integer(opstr)
		if v is not None:
			assert 0 <= v < 0x100
			fields[self.name + 't'] = 1
			fields[self.name] = v
			return

		raise Exception('invalid MemoryIndexDesc %r' % (opstr,))

#@document_operand
class ThreadgroupIndexDesc(OperandDesc):
#	pseudocode = '''
#	{name}(value, flags):
#		if flags != 0:
#			return BroadcastImmediateReference(sign_extend(value, 16))
#		else:
#			if value & 1: UNDEFINED()
#			if value >= 0x100: UNDEFINED()
#			return Reg32Reference(value >> 1)
#	'''

	def __init__(self, name):
		super().__init__(name)
		self.add_merged_field(self.name, [
			(28, 6, self.name),
			(48, 10, self.name + 'x')
		])
		self.add_field(34, 1, self.name + 't')

	def decode_impl(self, fields, allow64):
		flags = fields[self.name + 't']
		value = fields[self.name]
		if flags:
			return sign_extend(value, 16)
		else:
			assert value < 0x100
			return Reg16(value & 0xFF)

	def decode(self, fields):
		return self.decode_impl(fields, allow64=False)

	def encode_string(self, fields, opstr):
		r = try_parse_register(opstr)
		if r is not None:
			if isinstance(r, Reg16):
				fields[self.name + 't'] = 0
				fields[self.name] = r.n << 1
				return

		v = try_parse_integer(opstr)
		if v is not None:
			assert -0x8000 <= v < 0x8000
			fields[self.name + 't'] = 1
			fields[self.name] = v & 0xFFFF
			return

		raise Exception('invalid ThreadgroupIndexDesc %r' % (opstr,))

@document_operand
class MemoryBaseDesc(OperandDesc):
	def __init__(self, name, r_off):
		super().__init__(name)
		self.add_field(32, 7, self.name)
		self.add_field(r_off, 1, self.name + 'r')

	pseudocode = '''
	{name}(value, r):
		if r:
			return Reg64Reference(value)
		else:
			return UReg64Reference(value << 1)
	'''

	def decode_impl(self, fields, allow64):
		r = fields[self.name + 'r']
		value = fields[self.name]
		if r:
			return Reg64(value)
		else:
			return UReg64(value << 1)

	def decode(self, fields):
		return self.decode_impl(fields, allow64=False)

	def encode_string(self, fields, opstr):
		r = try_parse_register(opstr)
		if not isinstance(r, (Reg64, UReg64)):
			raise Exception('invalid MemoryBaseDesc %r' % (opstr,))

		fields[self.name + 't'] = 1 if isinstance(r, UReg64) else 0
		fields[self.name] = r.n << 1

class SampleMaskDesc(OperandDesc):
	def __init__(self, name, off=42, offx=56, offt=22, flags_type=2):
		super().__init__(name)
		self.add_merged_field(self.name, [
			(off, 6, self.name),
			(offx, 2, self.name + 'x'),
		])
		self.add_field(offt, flags_type, self.name + 't')
		assert(flags_type in [1, 2])
		self.flags_type = flags_type

	def decode(self, fields):
		flags = fields[self.name + 't']
		value = fields[self.name]

		if flags >= 2:
			assert(0)

		if (flags == 0b0) == (self.flags_type == 2):
			return Immediate(value)
		else:
			return Reg16(value)

	def encode_string(self, fields, opstr):
		assert(0)

class DiscardMaskDesc(OperandDesc):
	def __init__(self, name):
		super().__init__(name)
		self.add_merged_field(self.name, [
			(16, 6, self.name),
			(26, 2, self.name + 'x')
		])
		self.add_field(23, 1, self.name + 't')

	def decode(self, fields):
		value = fields[self.name]
		flags = fields[self.name + 't']
		if flags == 0:
			return Reg16(value)
		else:
			return Immediate(value)

	def encode_string(self, fields, opstr):
		assert(0)


class ZSDesc(OperandDesc):
	def __init__(self, name):
		super().__init__(name)
		self.add_merged_field(self.name, [
			(16, 6, self.name),
			(26, 2, self.name + 'x')
		])
		self.add_field(29, 1, 'z')
		self.add_field(30, 1, 's')

	def decode(self, fields):
		value = fields[self.name]
		count = (2 if fields['z'] else 0) + (1 if fields['s'] else 0)
		assert(count > 0 and "otherwise the instr is pointless")
		# Unclear how alignment requirements work
		return RegisterTuple(Reg16(value + i) for i in range(count))

	def encode_string(self, fields, opstr):
		assert(0)

LOAD_STORE_MASK = {
	0x0: '',
	0x1: 'x',
	0x2: 'yw',
	0x3: 'xyw',
	0x4: 'zw',
	0x5: 'xzw',
	0x6: 'yzw',
	0x7: 'xyzw',
	0x8: 'y',
	0x9: 'xy',
	0xa: 'z',
	0xb: 'xz',
	0xc: 'yz',
	0xd: 'xyz',
	0xe: 'w',
	0xf: 'xw',
}

LOAD_STORE_TYPE = {
	0: 'i16',
	1: 'i32',
	2: 'i8',
	3: 'i32', # Unobserved
}

class MemoryRegDesc(OperandDesc):
	def __init__(self, name):
		super().__init__(name)
		self.add_merged_field(self.name, [
			(24, 1, self.name + 'l'),
			(25, 7, self.name),
		])

	def decode_impl(self, fields, allow64):
		value = fields[self.name]
		lstype = LOAD_STORE_TYPE[fields['type']]
		count = len(LOAD_STORE_MASK[fields['mask']])
		if lstype == 'i8':
			regcount = (count + 1) >> 1
			regtype = 16
		elif lstype == 'i16':
			regcount = count
			regtype = 16
		elif lstype == 'i32':
			regcount = count
			regtype = 32
		else:
			assert(0)

		if regtype == 16:
			return RegisterTuple(Reg16(value + i) for i in range(regcount))
		else:
			return RegisterTuple(Reg32((value >> 1) + i) for i in range(regcount))

	def decode(self, fields):
		return self.decode_impl(fields, allow64=False)

	def encode_string(self, fields, opstr):
		regs = [try_parse_register(i) for i in opstr.split('_')]
		if regs and all(isinstance(r, Reg32) for r in regs):
			flags = 1
			value = regs[0].n << 1
		elif regs and all(isinstance(r, Reg16) for r in regs):
			flags = 0
			value = regs[0].n
		else:
			raise Exception('invalid MemoryRegDesc %r' % (opstr,))

		for i in range(1, len(regs)):
			if regs[i].n != regs[i-1].n + 1:
				raise Exception('invalid MemoryRegDesc %r (must be consecutive)' % (opstr,))

		if not 0 < len(regs) <= 4:
			raise Exception('invalid MemoryRegDesc %r (1-4 values)' % (opstr,))

		#fields['mask'] = (1 << len(regs)) - 1
		fields[self.name] = value
		fields[self.name + 't'] = flags


class ThreadgroupMemoryRegDesc(OperandDesc):
	# TODO: exactly the same as MemoryRegDesc except for the offsets?
	def __init__(self, name, offa=None):
		super().__init__(name)
		self.add_merged_field(self.name, [
			(9, 6, self.name),
			(60, 2, self.name + 'x'),
		])
		self.add_field(8, 1, self.name + 't')
		if offa is not None:
			self.add_field(offa, 1, self.name + 'a')
			self.is_optional = True
		else:
			self.is_optional = False

	def decode_impl(self, fields, allow64):
		if self.is_optional and fields[self.name + 'a'] == 0:
			return None

		flags = fields[self.name + 't']

		value = fields[self.name]

		if self.is_optional:
			count = 1
		else:
			count = bin(fields['mask']).count('1')

		if flags == 0b0:
			return RegisterTuple(Reg16(value + i) for i in range(count))
		else:
			return RegisterTuple(Reg32((value >> 1) + i) for i in range(count))

	def decode(self, fields):
		return self.decode_impl(fields, allow64=False)

	def encode_string(self, fields, opstr):
		regs = [try_parse_register(i) for i in opstr.split('_')]
		if regs and all(isinstance(r, Reg32) for r in regs):
			flags = 0b1
			value = regs[0].n << 1
		elif regs and all(isinstance(r, Reg16) for r in regs):
			flags = 0b0
			value = regs[0].n
		else:
			raise Exception('invalid ThreadgroupMemoryRegDesc %r' % (opstr,))

		for i in range(1, len(regs)):
			if regs[i].n != regs[i-1].n + 1:
				raise Exception('invalid ThreadgroupMemoryRegDesc %r (must be consecutive)' % (opstr,))

		if not 0 < len(regs) <= 4:
			raise Exception('invalid ThreadgroupMemoryRegDesc %r (1-4 values)' % (opstr,))

		fields['mask'] = (1 << len(regs)) - 1
		fields[self.name] = value
		fields[self.name + 't'] = flags

class TileToMemoryRegDesc(OperandDesc):
	# TODO: exactly the same as MemoryRegDesc except for the offsets?
	def __init__(self, name):
		super().__init__(name)
		self.add_merged_field(self.name, [
			(9, 6, self.name),
			(56, 2, self.name + 'x'),
		])

	def decode_impl(self, fields, allow64):
		return Reg16(fields[self.name])

	def decode(self, fields):
		return self.decode_impl(fields, allow64=False)

#@document_operand
class ThreadgroupMemoryBaseDesc(OperandDesc):
	def __init__(self, name):
		super().__init__(name)
		self.add_merged_field(self.name, [
			(16, 6, self.name),
			(58, 2, self.name + 'x'),
		])
		self.add_field(22, 2, self.name + 't')

#	pseudocode = '''
#	{name}(value, flags):
#		if value & 1: UNDEFINED()
#		if flags != 0:
#			return UReg64Reference(value >> 1)
#		else:
#			return Reg64Reference(value >> 1)
#	'''

	def decode_impl(self, fields, allow64):
		flags = fields[self.name + 't']
		value = fields[self.name]
		if flags == 0b00:
			return Reg16(value)
		elif flags == 0b10:
			return Immediate(0)
		else:
			return UReg16(value | ((flags >> 1) << 8))

	def decode(self, fields):
		return self.decode_impl(fields, allow64=False)

#	def encode_string(self, fields, opstr):
#		r = try_parse_register(opstr)
#		if not isinstance(r, Reg16): # (Reg64, UReg64)):
#			raise Exception('invalid ThreadgroupMemoryBaseDesc %r' % (opstr,))
#
#		fields[self.name + 't'] = 0 if isinstance(r, Reg16) else 0b10
#		fields[self.name] = r.n << 1

class SReg32Desc(OperandDesc):
	def __init__(self, name):
		super().__init__(name)
		self.add_merged_field(self.name, [
			(8, 7, self.name),
		])

	pseudocode = '''
	{name}(value):
		return SpecialRegister(value)
	'''
	def decode(self, fields):
		return SReg32(fields[self.name])

	def encode_string(self, fields, opstr):
		s = opstr
		if ' (' in s and ')' in s:
			s = s.split(' (')[0]
		if s.startswith('sr'):
			try:
				v = int(s[2:])
			except ValueError:
				raise Exception('invalid SReg32Desc %r' % (opstr,))
			if 0 <= v < 256:
				fields[self.name] = v
				return
		raise Exception('invalid SReg32Desc %r' % (opstr,))



class ExReg32Desc(OperandDesc):
	def __init__(self, name, start, start_ex):
		super().__init__(name)

		# TODO: this ignores the low bit. Kinda confusing?
		self.add_merged_field(self.name, [
			(start, 5, self.name),
			(start_ex, 2, self.name + 'x'),
		])

	def decode(self, fields):
		v = fields[self.name]
		return Reg32(v)


class ExReg64Desc(OperandDesc):
	def __init__(self, name, start, start_ex):
		super().__init__(name)

		# TODO: this ignores the low bit. Kinda confusing?
		self.add_merged_field(self.name, [
			(start, 5, self.name),
			(start_ex, 2, self.name + 'x'),
		])

	def decode(self, fields):
		v = fields[self.name]
		return Reg64(v)



class ExReg16Desc(OperandDesc):
	def __init__(self, name, start, start_ex):
		super().__init__(name)

		self.add_merged_field(self.name, [
			(start, 6, self.name),
			(start_ex, 2, self.name + 'x'),
		])

	def decode(self, fields):
		v = fields[self.name]
		return Reg16(v)



instruction_descriptors = []
_instruction_descriptor_names = set()
def register(cls):
	assert cls.__name__ not in _instruction_descriptor_names, 'duplicate %r' % (cls.__name__,)
	_instruction_descriptor_names.add(cls.__name__)
	instruction_descriptors.append(cls())
	return cls

class MaskedInstructionDesc(InstructionDesc):
	def exec(self, instr, corestate):
		for thread in range(SIMD_WIDTH):
			if corestate.exec[thread]:
				self.exec_thread(instr, corestate, thread)

	def exec_thread(self, instr, corestate, thread):
		assert False, "TODO"

@register
class MovImm7InstructionDesc(MaskedInstructionDesc):
	documentation_begin_group = 'Move Instructions'
	documentation_name = 'Move 7-bit Immediate'
	def __init__(self):
		super().__init__('mov_imm', size=2)
		self.add_constant(0, 3, 0b100)
		self.add_constant(15, 1, 0)
		self.add_operand(ALUDstSDesc('D'))
		self.add_operand(ImmediateDesc('imm7', 8, 7))

@register
class MovImm32InstructionDesc(MaskedInstructionDesc):
	#documentation_begin_group = 'Miscellaneous Instructions'
	documentation_name = 'Move 32-bit immediate'
	def __init__(self):
		super().__init__('mov_imm', size=8)
		self.add_constant(0, 3, 0b100)
		self.add_constant(15, 1, 1)
		self.add_constant(16, 1, 0)
		self.add_constant(17, 1, 1) # Length = 8
		self.add_constant(20, 1, 0)
		self.add_operand(ALUDstLDesc('D'))
		self.add_operand(ImmediateDesc('imm32', [
			( 8,  7, 'immA'),
			(33,  4, 'immB'),
			(42,  2, 'immC'),
			(48, 12, 'immD'),
			(25,  7, 'immE'),
		]))

@register
class MovFromSrInstructionDesc(MaskedInstructionDesc):
	#documentation_begin_group = 'Miscellaneous Instructions'
	documentation_name = 'Move From Special Register'
	def __init__(self):
		super().__init__('get_sr', size=(4, 8))
		self.add_constant(0, 3, 0b100)
		self.add_constant(15, 1, 1)
		self.add_constant(20, 1, 1)
		self.add_operand(ALUDstLDesc('D'))
		self.add_operand(SReg32Desc('SR'))
		self.add_operand(ImmediateDesc('G', 29, 3))

class DeviceLoadStoreInstructionDesc(MaskedInstructionDesc):
	def __init__(self, name, is_load, high_base):
		super().__init__(name, size=14)
		self.add_constant(0, 12, 0x67 if is_load else 0xe7)
		self.add_operand(ImmediateDesc('g', 70, 3)) # wait group
		self.add_operand(EnumDesc('type', 68, 2, LOAD_STORE_TYPE))
		self.add_operand(EnumDesc('mask', 64, 4, LOAD_STORE_MASK))
		self.add_operand(MemoryRegDesc('R'))
		self.add_operand(MemoryBaseDesc('B', high_base + 26))
		self.add_operand(MemoryIndexDesc('I', high_base, high_base + 27))
		self.add_operand(MemoryShiftDesc('s', high_base + 22))
		self.add_operand(SignedImmediateDesc('offset', high_base + 2, 16))
		self.add_operand(WaitDesc('W', lo=12, hi=15))
		self.add_operand(ImmediateDesc('$', high_base + 19, 1)) # Should the load use L1 cache?
		self.add_unsure_constant(18, 2, 0b01)
		self.add_operand(ImmediateDesc('q0', 20, 1))
		self.add_unsure_constant(21, 3, 0b10)


@register
class DeviceLoadInstructionDesc(DeviceLoadStoreInstructionDesc):
	def __init__(self):
		super().__init__('device_load', is_load=True, high_base=75)
		self.add_operand(ImmediateDesc('q1', 74, 1))

@register
class DeviceStoreInstructionDesc(DeviceLoadStoreInstructionDesc):
	def __init__(self):
		super().__init__('device_store', is_load=False, high_base=73)

@register
class StopInstructionDesc(InstructionDesc):
	documentation_begin_group = 'Other Instructions'
	def __init__(self):
		super().__init__('stop', size=4)
		self.add_constant(0, 4, 0b1110)
		self.add_constant(4, 28, 0)

	pseudocode = '''
	end_execution()
	'''

# Techincally these are all 2-byte wait instructions
# But since all instructions have wait masks anyways, the only uses are as padding (nop)
# and as a workaround for some hardware issue in base M3s,
# where the compiler always inserts a wait followed by 3 nops
# To avoid filling disasm with 4 instructions for every one of those, pretend it's one 8-byte instruction
@register
class NopInstructionDesc(InstructionDesc):
	def __init__(self):
		super().__init__('nop', size=2)
		self.add_constant(0, 4, 0b0110)
		self.add_constant(4, 12, 0)

@register
class WaitInstructionDesc(InstructionDesc):
	def __init__(self):
		super().__init__('wait', size=8)
		self.add_constant(0, 4, 0b0110)
		self.add_constant(4, 6, 0)
		self.add_operand(WaitDesc('W', lo=10, hi=13, use_label=False))
		self.add_constant(16, 16, 0b0110) # nop
		self.add_constant(32, 16, 0b0110) # nop
		self.add_constant(48, 16, 0b0110) # nop

def get_instruction_descriptor(n):
	for o in instruction_descriptors:
		if o.matches(n):
			return o

def disassemble_n(n):
	for o in instruction_descriptors:
		if o.matches(n):
			return o.disassemble(n)

def disassemble_bytes(b):
	n = opcode_to_number(b)
	return disassemble_n(n)

