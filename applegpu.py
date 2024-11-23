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

def bit_count(num):
	if hasattr(num, 'bit_count'):
		return num.bit_count()
	else:
		return bin(num).count('1')

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

	def is_int(self):
		return SIGN_EXTEND_FLAG in self.flags

	def is_float(self):
		return NEGATE_FLAG in self.flags or ABS_FLAG in self.flags

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
		self.implicit_fields = {}

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
		assert (self.fields_mask & mask) == 0, name
		for _, _, existing_name in self.fields:
			assert existing_name != name, name

		self.fields_mask |= mask
		self.fields.append((start, size, name))

	def add_merged_field(self, name, subfields):
		pairs = []
		shift = 0
		for start, size, subname in subfields:
			self.add_raw_field(start, size, subname)
			pairs.append((subname, shift, size))
			shift += size
		self.merged_fields.append((name, pairs))

	def add_field(self, start, size, name):
		self.add_merged_field(name, [(start, size, name)])

	def add_implicit_field(self, name, value):
		self.implicit_fields[name] = value

	def add_suboperand(self, operand):
		# a "suboperand" is an operand which does not appear in the operand list,
		# but is used by other operands. currently unused.
		for start, size, name in operand.fields:
			self.add_field(start, size, name)
		for name, subfields in operand.merged_fields:
			self.add_merged_field(name, subfields)
		for name, value in operand.implicit_fields:
			self.add_implicit_field(name, value)
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

		return self.to_bytes(encoded)

	def encode_raw_fields(self, fields):
		assert sorted(lookup.keys()) == sorted(name for start, size, name in self.fields)
		return self.patch_raw_fields(self.bits, fields)

	def patch_fields(self, encoded, fields):
		mf_lookup = dict(self.merged_fields)

		raw_fields = {}
		for name, field in self.merged_fields:
			value = fields[name]
			for subname, shift, size in field:
				mask = (1 << size) - 1
				raw_fields[subname] = (value >> shift) & mask

		return self.patch_raw_fields(encoded, raw_fields)

	def encode_fields(self, fields):
		assert self._can_encode_fields(fields, print_err=True)
		return self.patch_fields(self.bits, fields)

	def to_bytes(self, instr):
		return bytes((instr >> (i*8)) & 0xFF for i in range(self.decode_size(instr)))

	def decode_fields(self, instr):
		raw = dict(self.decode_raw_fields(instr))
		fields = []
		for name, subfields in self.merged_fields:
			value = 0
			for subname, shift, size in subfields:
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
		if self.sizes[0] != self.sizes[1]:
			mask |= 1 << self.length_bit_pos
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
		for i, operand in enumerate(self.ordered_operands):
			if i < len(opstrs):
				insert = operand.encode_insert_optional_default(opstrs[i])
				if insert:
					opstrs.insert(i, insert)
			else:
				insert = operand.encode_insert_optional_default('')
				opstrs.append(insert or '')
		return opstrs

	def _can_encode_fields(self, fields, print_err=False):
		encodable = {name: parts for name, parts in self.merged_fields}
		for k, v in fields.items():
			try:
				size = sum([x[2] for x in encodable[k]])
				if v >= (1 << size):
					if print_err:
						print(f"{type(self).__name__} can't encode {k} of {v} in {size} bits")
					return False
			except KeyError:
				if v != self.implicit_fields.get(k, 0):
					if print_err:
						print(f"{type(self).__name__} missing field to encode {k} of {v}")
					return False
		return True

	def can_encode_fields(self, fields):
		return self._can_encode_fields(fields)

class InstructionGroup:
	def __init__(self, name, members):
		self.name = name
		self.members = members
		# The last instruction in the group should have the most complete operand list, use that
		self.ordered_operands = members[-1].ordered_operands

	def encode_fields(self, fields):
		for member in self.members[:-1]:
			if member.can_encode_fields(fields):
				return member.encode_fields(fields)
		return self.members[-1].encode_fields(fields)

	def fields_for_mnem(self, mnem, operand_strings):
		return self.members[-1].fields_for_mnem(mnem, operand_strings)

	def rewrite_operands_strings(self, mnem, opstrs):
		return self.members[-1].rewrite_operands_strings(mnem, opstrs)

documentation_operands = []

def document_operand(cls):
	documentation_operands.append(cls)
	return cls

class OperandDesc:
	def __init__(self, name=None):
		self.name = name
		self.fields = []
		self.merged_fields = []
		self.implicit_fields = []

	def add_field(self, start, size, name):
		self.fields.append((start, size, name))

	def add_merged_field(self, name, subfields):
		self.merged_fields.append((name, subfields))

	def add_implicit_field(self, name, value):
		self.implicit_fields.append((name, value))

	def decode(self, fields):
		return '<TODO>'

	def get_bit_size(self, fields):
		r = self.decode(fields)
		return r.get_bit_size()

	# For optional operands, return an element to insert if operand isn't of the right type
	def encode_insert_optional_default(self, opstr):
		return None

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
				self.add_merged_field(self.name + 'm', [
					(lo, 3, self.name + 'ml'),
					(hi, 3, self.name + 'mh')
				])
		else:
			self.add_field(lo, 3, self.name)

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

	def encode_insert_optional_default(self, opstr):
		if self.use_label and not opstr.startswith('wait '):
			return 'wait none'

	def encode_string(self, fields, opstr):
		name = self.name + ('m' if self.is_mask else '')
		if self.use_label:
			assert(opstr.startswith('wait '))
			opstr = opstr[5:]
		if opstr == 'none':
			fields[name] = 0
		elif self.is_mask:
			value = 0
			try:
				for idx in opstr:
					value |= 1 << int(idx)
			except ValueError:
				raise Exception(f'invalid wait mask {opstr}')
			fields[name] = value
		else:
			idx = try_parse_integer(opstr)
			if idx is None or idx >= 6:
				raise Exception(f'invalid single wait {opstr}')
			fields[name] = idx + 1

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

# Common base class for src registers of variable-length instructions
class VariableSrcDesc(AbstractSrcOperandDesc):
	def is_int(self, fields):
		return False
	def get_size(self, fields):
		return fields.get(self.name + 's', 1)

	def __init__(self, name, bit_off, l_off=None, c_off=None, d_off=None, common_layout=None, s_off=None, h_off=None, u_off=None, n_off=None, a_off=None, u_default=0):
		super().__init__(name)
		if common_layout is not None:
			c_off = c_off or bit_off + 6
			if s_off is None and (common_layout == 'A' or common_layout == 'B'):
				s_off = bit_off - 1
			if common_layout == 'A':
				d_off = d_off or 19
				l_off = l_off or 51
				u_off = u_off or l_off + 4
				h_off = h_off or l_off + 5
			elif common_layout == 'B':
				d_off = d_off or 20
				l_off = l_off or 52
				u_off = u_off or l_off + 5
				h_off = h_off or l_off + 6
				n_off = n_off or l_off + 7
			else:
				l_off = l_off or bit_off - 1
				d_off = d_off or bit_off - 2
				h_off = h_off or bit_off - 3
				u_off = u_off or bit_off - 4
				n_off = n_off or bit_off - 5
				a_off = a_off or bit_off - 6

		self.value_shift = 1 if l_off is None else 0
		if l_off is not None:
			self.add_merged_field(self.name, [
				(l_off, 1, self.name + 'l'),
				(bit_off, 6, self.name)
			])
		else:
			self.add_field(bit_off, 6, self.name)
		if h_off is not None:
			self.add_field(h_off, 1, self.name + 'h')

		self.add_field(c_off, 1, self.name + 'c')
		self.add_field(d_off, 1, self.name + 'd')
		if s_off is not None:
			self.add_field(s_off, 1, self.name + 's')
		else:
			self.add_implicit_field(self.name + 's', 1)
		if u_off is not None:
			self.add_field(u_off, 1, self.name + 'u')
		if n_off is not None:
			self.add_field(n_off, 1, self.name + 'n')
		if a_off is not None:
			self.add_field(a_off, 1, self.name + 'a')

		self.u_default = u_default

	def decode(self, fields):
		high_bit = 	fields.get(self.name + 'h', 0)

		value = fields[self.name] << self.value_shift
		uniform_bit = fields.get(self.name + 'u', self.u_default) # is uniform
		size_bit = self.get_size(fields) # is 32-bit
		discard_bit = fields[self.name + 'd']
		cache_bit = fields[self.name + 'c']

		negate_bit = fields.get(self.name + 'n', 0)
		abs_bit = fields.get(self.name + 'a', 0)

		if uniform_bit:
			value |= (discard_bit << 7) | (high_bit << 8)

			if cache_bit:
				if self.is_int(fields):
					return Immediate(value & 0xff)
				else:
					return Immediate(decode_float_immediate(value))
			else:
				if size_bit:
					r = UReg32(value >> 1)
				else:
					r = UReg16(value)
		else:
			value |= (high_bit << 7)

			if size_bit:
				r = Reg32(value >> 1)
			else:
				r = Reg16(value)

			if cache_bit:
				r.flags.append(CACHE_FLAG)
			if discard_bit:
				r.flags.append(DISCARD_FLAG)

		if self.is_int(fields):
			if negate_bit:
				r.flags.append(SIGN_EXTEND_FLAG)
		else:
			if abs_bit:
				r.flags.append(ABS_FLAG)
			if negate_bit:
				r.flags.append(NEGATE_FLAG)

		return r

	def encode_reg(self, fields, reg):
		u16 = isinstance(reg, UReg16)
		u32 = isinstance(reg, UReg32)
		r16 = isinstance(reg, Reg16)
		r32 = isinstance(reg, Reg32)
		u = u16 or u32
		s = u32 or r32
		c = 0
		h = 0
		d = 0
		n = NEGATE_FLAG in reg.flags or SIGN_EXTEND_FLAG in reg.flags
		a = ABS_FLAG in reg.flags
		value = reg.n
		if s:
			value <<= 1
		if u:
			h = (value >> 8) & 1
			d = (value >> 7) & 1
		else:
			h = (value >> 7) & 1
			d = DISCARD_FLAG in reg.flags
			c = CACHE_FLAG in reg.flags
		fields[self.name] = value & 0x7f
		fields[self.name + 'u'] = u
		fields[self.name + 's'] = s
		fields[self.name + 'c'] = c
		fields[self.name + 'h'] = h
		fields[self.name + 'd'] = d
		fields[self.name + 'n'] = n
		fields[self.name + 'a'] = a

	def encode_imm(self, fields, imm):
		fields[self.name] = imm & 0x7f
		fields[self.name + 'd'] = (imm >> 7) & 1
		fields[self.name + 'u'] = 1
		fields[self.name + 's'] = 0
		fields[self.name + 'c'] = 1
		fields[self.name + 'h'] = 0
		fields[self.name + 'n'] = 0
		fields[self.name + 'a'] = 0

	def encode_string(self, fields, opstr):
		reg = try_parse_register(opstr)
		if reg:
			self.encode_reg(fields, reg)
		elif opstr in float_immediate_lookup:
			self.encode_imm(fields, float_immediate_lookup[opstr])
		else:
			raise Exception(f'invalid VariableSrcDesc {opstr}')

class VariableDstDesc(AbstractDstOperandDesc):
	def __init__(self, name, bit_off=4, l_off=None, x_off=22, h_off=None, z_off=None, s_off=3, c_off=21, u_off=None):
		super().__init__(name)
		self.value_shift = 1 if l_off is None else 0
		main_fields = [
			(bit_off, 4, self.name),
			(x_off,   2, self.name + 'x')
		]
		if l_off is not None:
			main_fields.insert(0, (l_off, 1, self.name + 'l'))
		if h_off is not None:
			main_fields.append((h_off, 1, self.name + 'h'))
		self.add_merged_field(self.name, main_fields)
		if s_off is not None:
			self.add_field(s_off, 1, self.name + 's') # size
		if c_off is not None:
			self.add_field(c_off, 1, self.name + 'c') # cache
		if u_off is not None:
			self.add_field(u_off, 1, self.name + 'u') # is uniform
		if z_off is not None:
			self.add_field(z_off, 1, self.name + 'z') # high part of uniform

	def decode(self, fields):
		value = fields[self.name] << self.value_shift

		uniform_bit = fields.get(self.name + 'u', 0) # is uniform
		size_bit = fields[self.name + 's'] # is 32-bit
		cache_bit = fields[self.name + 'c']

		high_uniform_bit = fields.get(self.name + 'z', 0)

		if uniform_bit:
			value |= high_uniform_bit << 8
			if size_bit:
				r = UReg32(value >> 1)
			else:
				r = UReg16(value)
		else:
			if size_bit:
				r = Reg32(value >> 1)
			else:
				r = Reg16(value)

		if cache_bit:
			r.flags.append(CACHE_FLAG)

		return r

	def encode_reg(self, fields, reg):
		u16 = isinstance(reg, UReg16)
		u32 = isinstance(reg, UReg32)
		r16 = isinstance(reg, Reg16)
		r32 = isinstance(reg, Reg32)
		u = u16 or u32
		s = u32 or r32
		value = reg.n
		if s:
			value <<= 1
		fields[self.name] = value & 0xff
		fields[self.name + 'c'] = CACHE_FLAG in reg.flags
		fields[self.name + 'u'] = u
		fields[self.name + 's'] = s
		fields[self.name + 'z'] = value >> 8

	def encode_string(self, fields, opstr):
		reg = try_parse_register(opstr)
		if reg:
			self.encode_reg(fields, reg)
		else:
			raise Exception(f'invalid VariableDstDesc {opstr}')

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
		assert(opstr.startswith('lsl ')) # If not, we should have inserted an 'lsl 0' in encode_insert_optional_default
		try:
			fields['s'] = int(opstr[4:])
		except ValueError:
			raise Exception(f'invalid ShiftDesc {opstr}')
		fields['s'] = s

	def encode_insert_optional_default(self, opstr):
		if not opstr.startswith('lsl '):
			return 'lsl 0'

class NewShiftDesc(OperandDesc):
	documentation_no_name = True

	def __init__(self, name):
		super().__init__(name)
		self.add_merged_field('s', [
			(62, 1, 's1'),
			(76, 1, 's2'),
		])

	def decode(self, fields):
		shift = fields['s']
		return 'lsl %d' % (shift) if shift else 'lsl 4'


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
	{name}(value, discard, size, sx, t):
		match t:
			case 0:
				# TODO: Does sx affect this?  Apple's compiler uses this for positive values and the offset field for negative ones...
				return UnsignedImmediate(all 16 bits between l and h, which includes value, discard, and size)
			case 1:
				reg = size ?  Reg32Reference(value) :  Reg16Reference(value)
			case 2:
				reg = size ? UReg32Reference(value) : UReg16Reference(value)
		if sx:
			reg.sign_extend_to_64_bits
		if discard:
			reg.discard
		return reg
	'''

	def __init__(self, name, sx_off, t_off, is_load):
		super().__init__(name)
		self.is_load = is_load
		self.add_merged_field(self.name, [
			(39, 1, self.name + 'l'),
			(40, 7, self.name),
		])
		if is_load:
			self.add_field(47, 1, self.name + 'd')
			self.add_field(48, 1, self.name + 'z')
		else:
			self.add_field(47, 1, self.name + 'z')
			self.add_field(48, 1, self.name + 'd')
		self.add_field(49, 4, self.name + 'x')
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
		z       = fields[self.name + 'z']
		x       = fields[self.name + 'x']
		h       = fields[self.name + 'h']

		assert(t != 3)
		if t != 0:
			# We currently think these are only used for immediates
			assert(x == 0)
			assert(h == 0)

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
			# Same bits either way, but the fields are in different places for register encoding, so...
			if self.is_load:
				return value + (discard << 8) + (z << 9) + (x << 10) + (size << 14) + (h << 15)
			else:
				return value + (z << 8) + (discard << 9) + (x << 10) + (size << 14) + (h << 15)

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

# Helper for instructions that are a part of a group where the largest version uses Wm, but smaller versions use W
class EncodeWmAsWHelper:
	def can_encode_fields(self, fields):
		if bit_count(fields['Wm']) > 1:
			return False
		fields = dict(fields)
		del fields['Wm']
		return super().can_encode_fields(fields)

	def encode_fields(self, fields):
		fields['W'] = fields['Wm'].bit_length()
		del fields['Wm']
		return super().encode_fields(fields)

instruction_descriptors = []
instruction_descriptors_assemble = []
_instruction_descriptor_names = set()
def register(cls):
	group = [cls()]
	instruction_descriptors_assemble.append(group[0])
	if isinstance(group[0], InstructionGroup):
		group = group[0].members
	for member in group:
		name = type(member).__name__
		assert name not in _instruction_descriptor_names, 'duplicate %r' % (name,)
		_instruction_descriptor_names.add(name)
		instruction_descriptors.append(member)
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
		self.add_unsure_constant(24, 4, 0b0110)
		self.add_operand(ALUDstLDesc('D'))
		self.add_operand(SReg32Desc('SR'))
		self.add_operand(ImmediateDesc('g', 29, 3))
	pseudocode = '''
	g -> output wait group
	'''

class DeviceLoadStoreInstructionDesc(MaskedInstructionDesc):
	def __init__(self, name, is_load, high_base):
		super().__init__(name, size=14)
		self.add_constant(0, 12, 0x67 if is_load else 0xe7)
		self.add_operand(ImmediateDesc('g', 70, 3)) # wait group
		self.add_operand(EnumDesc('type', 68, 2, LOAD_STORE_TYPE))
		self.add_operand(EnumDesc('mask', 64, 4, LOAD_STORE_MASK))
		self.add_operand(MemoryRegDesc('R'))
		self.add_operand(MemoryBaseDesc('B', high_base + 26))
		self.add_operand(MemoryIndexDesc('I', high_base, high_base + 27, is_load))
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
	pseudocode = '''
	R = *(B + I * s + offset)

	s and offset are in bytes (unlike M1, where they were in elements)
	If type is i8, one 16-bit register is allocated per two input elements, and any unused bits are zeroed
	g -> Output wait group
	$ -> Unset on coherent buffers, probably controls whether to skip noncoherent cache levels
	q0 -> Seems to get unset on the last load of a shader?
	q1 -> ???
	'''

@register
class DeviceStoreInstructionDesc(DeviceLoadStoreInstructionDesc):
	def __init__(self):
		super().__init__('device_store', is_load=False, high_base=73)

	pseudocode = '''
	*(B + I * s + offset) = R

	s and offset are in bytes (unlike M1, where they were in elements)
	$ -> Unset on coherent buffers, probably controls whether to skip noncoherent cache levels
	q0 -> I thought I've seen this unset on stores before but now I can't get it to happen
	'''


class NewALUDstDesc(AbstractSrcOperandDesc):
	def __init__(self, name):
		super().__init__(name)

		# destination bits
		self.add_merged_field(self.name, [
			(24, 1, self.name + 'l'),
			(25, 7, self.name)
		])

		self.add_field(32, 1, self.name + 'c') # cache (or uniform register high)
		self.add_field(33, 1, self.name + 'r') # 1 = GPR, 0 = UReg
		self.add_field(59, 2, self.name + 's') # size: 0 = 16-bit, 1 = 32-bit, 2 = 64-bit


	def get_size(self, fields):
		return fields[self.name + 's']

	def decode(self, fields):
		value = fields[self.name]
		uniform_bit = not fields[self.name + 'r'] # is register
		size_bits = self.get_size(fields)
		cache_bit = fields[self.name + 'c']

		if uniform_bit:
			value |= cache_bit << 8
			if size_bits == 0:
				r = UReg16(value)
			elif size_bits == 1:
				r = UReg32(value >> 1)
			elif size_bits == 2:
				r = UReg64(value >> 1)
			else:
				raise Exception(f'unknown size bits for uniform in NewALUDstDesc ({size_bits})')
		else:
			if size_bits == 0:
				r = Reg16(value)
			elif size_bits == 1:
				r = Reg32(value >> 1)
			elif size_bits == 2:
				r = Reg64(value >> 1)
			else:
				raise Exception(f'unknown size bits for register in NewALUDstDesc ({size_bits})')
			if cache_bit:
				r.flags.append(CACHE_FLAG)

		return r

class NewALUSrcDesc(AbstractSrcOperandDesc):
	def __init__(self, name, bit_off, s_off, t_off, r_off, is_64):
		super().__init__(name)

		sx_off = r_off + 1

		# destination bits
		self.add_merged_field(self.name, [
			(bit_off, 1, self.name + 'l'),
			(bit_off+1, 7, self.name)
		])

		self.add_field(bit_off+8, 1, self.name + 'c') # cache (or uniform register high)

		if is_64:
			self.add_field(s_off, 2, self.name + 's')
		else:
			self.add_field(s_off, 1, self.name + 's')

		self.add_field(r_off, 1, self.name + 'r')
		self.add_field(t_off, 1, self.name + 't')
		self.add_field(sx_off, 1, self.name + 'sx')

	def get_size(self, fields):
		return fields[self.name + 's']

	def decode(self, fields):
		value = fields[self.name]
		register_bit = fields[self.name + 'r'] # is register
		size_bits = self.get_size(fields)
		cache_bit = fields[self.name + 'c']

		t_bit = fields[self.name + 't']
		sx_bit = fields[self.name + 'sx']

		if register_bit:
			if size_bits == 0:
				r = Reg16(value)
			elif size_bits == 1:
				r = Reg32(value >> 1)
			elif size_bits == 2:
				r = Reg64(value >> 1)
			else:
				raise Exception(f'unknown size bits for register in NewALUDstDesc ({size_bits})')
			if cache_bit:
				r.flags.append(CACHE_FLAG)
			if t_bit:
				r.flags.append(DISCARD_FLAG)
		else:
			value |= cache_bit << 8

			if t_bit:
				if size_bits == 0:
					r = UReg16(value)
				elif size_bits == 1:
					r = UReg32(value >> 1)
				elif size_bits == 2:
					r = UReg64(value >> 1)
				else:
					raise Exception(f'unknown size bits for uniform in NewALUDstDesc ({size_bits})')
			else:
				return Immediate(value & 0xff)

		if sx_bit:
			r.flags.append(SIGN_EXTEND_FLAG)
		return r

class NewFloatSrcDesc(VariableSrcDesc):
	def is_int(self, fields):
		return False
	def get_size(self, fields):
		return fields.get(self.name + 's', 1)

class FFMA4BDesc(NewFloatSrcDesc):
	def get_size(self, fields):
		if fields['Z']:
			return 1
		else:
			return fields[self.name + 's']

	def encode_reg(self, fields, reg):
		res = super().encode_reg(fields, reg)
		if fields['Z']:
			fields[self.name + 's'] = 0


class FFMAInstructionDescBase(MaskedInstructionDesc):
	def fields_to_mnem_suffix(self, fields):
		suffix = ''

		if fields.get('S', 0):
			suffix += '.sat'

		if fields.get('q0', 0):
			suffix += '.first'

		return suffix

	def fields_for_mnem(self, mnem, operand_strings):
		S = 0
		suffixes = {'sat': 'S', 'first': 'q0'}
		has = set()
		while True:
			rest, _, suffix = mnem.rpartition('.')
			if rest and suffix in fields:
				has.add(suffixes[suffix])
				mnem = rest
			else:
				break
		fields = self.fields_for_mnem_base(mnem)
		if fields is not None:
			for suffix in suffixes.values():
				fields[suffix] = suffix in has
		return fields

	def fields_for_mnem_base(self, mnem):
		if mnem == self.name: return {}



class IAddSubInstructionDesc(MaskedInstructionDesc):

	def fields_to_mnem_base(self, fields):
		return 'iadd' if fields['P'] else 'isub'

	def fields_for_mnem_base(self, mnem):
		if mnem == 'isub': return {'P': 0}
		if mnem == 'iadd': return {'P': 1}

	def __init__(self, mnem, imm, is_64=False, is_shifted=False):
		super().__init__(mnem, size=10)
		self.add_constant(0, 7, 0b0011111)
		self.add_constant(8, 1, 0b1)


		# both usually one, zero only on the first op in a function.
		self.add_field(20, 1, 'q1')
		self.add_field(22, 1, 'q2')

		self.add_operand(NewALUDstDesc('D'))

		if is_64:
			self.add_constant(59, 2, 0b10)
		else:
			self.add_constant(60, 1, 0)

		self.add_operand(NewALUSrcDesc('A', bit_off=41, s_off=61, t_off=65, r_off=72, is_64=is_64))
		self.add_operand(NewALUSrcDesc('B', bit_off=50, s_off=63, t_off=66, r_off=74, is_64=is_64))

		if not is_64:
			if is_shifted:
				self.add_constant(64, 1, 0) # !shift
				self.add_operand(NewShiftDesc('shift'))
			else:
				self.add_constant(64, 1, 1) # !shift
				self.add_field(62, 1, 'S') # saturate, or shift low bit

		self.add_operand(WaitDesc('W', lo=12, hi=15))

		self.add_constant(18, 1, 1) # maybe flag?
		#self.add_field(18, 1, 'q18')

		self.add_constant(68, 1, 1) # maybe flag?
		#self.add_field(68, 1, 'q68')

		self.add_field(7, 1, 'P')


	def fields_to_mnem_suffix(self, fields):
		suffix = ''

		if fields.get('S', 0):
			suffix += '.sat'

		if fields.get('q0', 0):
			suffix += '.first'

		return suffix

	def fields_for_mnem(self, mnem, operand_strings):
		S = 0
		suffixes = {'sat': 'S', 'q1': 'q1', 'q2': 'q2'}
		has = set()
		while True:
			rest, _, suffix = mnem.rpartition('.')
			if rest and suffix in suffixes:
				has.add(suffixes[suffix])
				mnem = rest
			else:
				break
		fields = self.fields_for_mnem_base(mnem)
		if fields is not None:
			for suffix in suffixes.values():
				fields[suffix] = suffix in has
		return fields

	def fields_for_mnem_base(self, mnem):
		if mnem == self.name: return {}


@register
class IAddInstructionDesc(IAddSubInstructionDesc):
	documentation_begin_group = 'Integer Arithmetic'
	documentation_name = '16/32-bit Integer Add/Subtract'
	def __init__(self):
		super().__init__('iadd', 0b110011111, is_64=False)

	pseudocode = '''
	for each active thread:
		a = A[thread]
		b = B[thread]

		saturating = S

		if P == 0:
			b = -b

		result = a + b

		if saturating:
			# TODO: signed/unsigned/width
			result = saturate(result)

		D[thread] = result
	'''

@register
class IAddShiftedInstructionDesc(IAddSubInstructionDesc):
	documentation_name = '16/32-bit Integer Add/Subtract with Shift'
	def __init__(self):
		super().__init__('iadd', 0b110011111, is_64=False, is_shifted=True)

	pseudocode = '''
	for each active thread:
		a = A[thread]
		b = B[thread]

		if shift == 0:
			shift_distance = 4
		else:
			shift_distance = shift

		if P == 0:
			b = -b

		result = a + (b << shift_distance)

		D[thread] = result
	'''

@register
class IAdd64InstructionDesc(IAddSubInstructionDesc):
	documentation_name = '64-bit Integer Add/Subtract'
	def __init__(self):
		super().__init__('iadd', 0b110011111, is_64=True)

	pseudocode = '''
	for each active thread:
		a = A[thread]
		b = B[thread]

		if P == 0:
			b = -b

		result = a + b

		D[thread] = result
	'''


class FFMA4InstructionDesc(FFMAInstructionDescBase):
	documentation_begin_group = 'Floating-Point Arithmetic'
	def __init__(self):
		super().__init__('ffma', size=4)
		self.add_constant(0, 3, 0b001)

		self.add_operand(VariableDstDesc('D',  4))
		self.add_operand(NewFloatSrcDesc('A',  9, s_off= 8, c_off=15, d_off=19))
		self.add_operand(FFMA4BDesc('B', 25, s_off=24, c_off=31, d_off=20))

		self.add_field(16, 1, 'Z')

		self.add_constant(17, 1, 0b1)
		self.add_constant(18, 1, 0b0)

	def fields_to_operands(self, fields):
		operands = super().fields_to_operands(fields)
		if fields['Z']:
			operands.insert(2, operands[0])
		else:
			operands.insert(3, operands[0])
		return operands

	def can_alias_field(self, fields, field):
		if field == 'B' and not fields['Cs']:
			return False # If Z = 1, B (which will contain the C param) must be 32-bit
		# Ignore discard, if we're writing to it it'll be discarded regardless of the flag
		# Ignore cache flag as well, we can merge those
		return (fields[field] == fields['D'] and fields[field + 's'] == fields['Ds'])

	def remove_c(self, fields):
		del fields['C']
		del fields['Cs']
		del fields['Cc']
		del fields['Cd']

	def can_encode_fields(self, fields):
		if (fields['A'] & 1) or (fields['B'] & 1) or (fields['C'] & 1) or (fields['D'] & 1):
			return False
		if self.can_alias_field(fields, 'B'):
			fields = dict(fields)
			# Copy C onto B, the flag bits will validate fine regardless so we can skip those
			fields['B'] = fields['C']
		elif self.can_alias_field(fields, 'C'):
			fields = dict(fields)
		else:
			return False

		self.remove_c(fields)
		return super().can_encode_fields(fields)

	def encode_fields(self, fields):
		if self.can_alias_field(fields, 'B'):
			fields['Z'] = 1
			fields['Dc'] |= fields['Bc']
			fields['B']  = fields['C']
			fields['Bs'] = fields['Cs']
			fields['Bc'] = fields['Cc']
			fields['Bd'] = fields['Cd']
		else:
			fields['Z'] = 0
			fields['Dc'] |= fields['Cc']
		# This was encoded by FFMA10, so we need to shift the register numbers since we have no l bits
		fields['A'] >>= 1
		fields['B'] >>= 1
		fields['D'] >>= 1
		self.remove_c(fields)
		return super().encode_fields(fields)

	pseudocode = '''
	if Z == 1:
		# B is 32-bit, Bs is ignored
		D = A * D + B
	else:
		D = A * B + D
	'''

class FFMA6InstructionDesc(FFMAInstructionDescBase):
	def __init__(self):
		super().__init__('ffma', size=6)
		self.add_constant(0, 3, 0b001)

		self.add_constant(16, 2, 0b10)

		self.add_constant(18, 1, 0b1) # 'L'
		self.add_constant(32, 2, 0b00)

		self.add_operand(VariableDstDesc('D', l_off=34))

		self.add_operand(NewFloatSrcDesc('A',
			9,
			l_off=35,
			s_off=8,
			c_off=15,
			d_off=19,
		))

		self.add_operand(NewFloatSrcDesc('B',
			25,
			l_off=36,
			h_off=38,
			s_off=24,
			c_off=31,
			d_off=20,
			u_off=37,
		))

		self.add_operand(NewFloatSrcDesc('C',
			41,
			l_off=40,
			c_off=47,
			d_off=39,
		))

class FFMA8InstructionDesc(EncodeWmAsWHelper, FFMAInstructionDescBase):
	def __init__(self):
		super().__init__('ffma', size=8)
		self.add_constant(0, 3, 0b001)
		self.add_constant(16, 2, 0b10)

		self.add_constant(18, 1, 0b1) # 'L'
		self.add_constant(32, 2, 0b01)

		self.add_operand(VariableDstDesc('D', l_off=50, h_off=60, u_off=54))
		self.add_operand(NewFloatSrcDesc('A',  9, common_layout='A'))
		self.add_operand(NewFloatSrcDesc('B', 25, common_layout='B'))
		self.add_operand(NewFloatSrcDesc('C', 41, common_layout='C', s_off=49))

		#self.add_field(61, 3, 'W') # wait
		self.add_operand(WaitDesc('W', 61))

class FFMA10InstructionDesc(FFMAInstructionDescBase):
	def __init__(self):
		super().__init__('ffma', size=(10,12), length_bit_pos=32)
		self.add_constant(0, 3, 0b001)
		self.add_constant(16, 2, 0b10)
		self.add_constant(18, 1, 0b1) # 'L'
		self.add_constant(33, 1, 0b1)

		self.add_operand(VariableDstDesc('D', l_off=50, h_off=60, z_off=66, u_off=54))
		self.add_operand(NewFloatSrcDesc('A',  9, common_layout='A', a_off=80, n_off=65))
		self.add_operand(NewFloatSrcDesc('B', 25, common_layout='B', a_off=81))
		self.add_operand(NewFloatSrcDesc('C', 41, common_layout='C', s_off=49))

		self.add_operand(WaitDesc('W', 61, 77))

		self.add_field(73, 1, 'S') # saturate
		self.add_field(88, 1, 'q0') # set if first in function

	pseudocode = '''
	op: D = A * B + C

	W -> wait for loads
	S -> saturate
	u -> is uniform (dest can be uniform)
	s -> size (is 32-bit)
	n -> negate
	a -> abs
	d -> discard
	c -> cache
	L, L2 -> length modifiers
	dest = Dz:Dh:Dx:D:Dl
	src = Ah:Ad:A:Al if Au, otherwise Ah:A:Al
	q0 -> set if first in function?

	TODO: undiscovered bits, verification
	'''

@register
class FFMAInstructionDesc(InstructionGroup):
	def __init__(self):
		super().__init__('ffma', [
			FFMA4InstructionDesc(),
			FFMA6InstructionDesc(),
			FFMA8InstructionDesc(),
			FFMA10InstructionDesc(),
		])

class CmpSrcDesc(VariableSrcDesc):
	documentation_extra_arguments = ['cc']
	def is_int(self, fields):
		return (fields['cc'] & 4) != 0

	def encode_string(self, fields, opstr):
		reg = try_parse_register(opstr)
		if reg:
			if reg.is_int() and not self.is_int(fields):
				raise Exception(f'register {opstr} incompatible with floating point compare {CMPSEL_CC[fields["cc"]]}')
			if reg.is_float() and self.is_int(fields):
				raise Exception(f'register {opstr} incompatible with integer compare {CMPSEL_CC[fields["cc"]]}')
			self.encode_reg(fields, reg)
		elif opstr in float_immediate_lookup:
			if self.is_int(fields):
				raise Exception(f'float immediate {opstr} incompatible with integer compare {CMPSEL_CC[fields["cc"]]}')
			self.encode_imm(fields, float_immediate_lookup[opstr])
		else:
			imm = try_parse_integer(opstr)
			if imm is not None:
				if not self.is_int(fields):
					raise Exception(f'integer immediate {opstr} incompatible with float point compare {CMPSEL_CC[fields["cc"]]}')
				self.encode_imm(fields, imm)
			else:
				raise Exception(f'invalid CmpSrcDesc {opstr}')

class SelSrcDesc(VariableSrcDesc):
	documentation_extra_arguments = ['Di', 'Ds']
	def is_int(self, fields):
		return fields['Di']
	def get_size(self, fields):
		return fields['Ds']

	def try_set_is_int(self, fields, is_int):
		if 'Di' in fields:
			if fields['Di'] != is_int:
				raise Exception(f'SelSrcDesc inputs disagree on what Di should be')
		else:
			fields['Di'] = is_int

	def encode_string(self, fields, opstr):
		reg = try_parse_register(opstr)
		if reg:
			if reg.is_int():
				self.try_set_is_int(fields, 1)
			if reg.is_float():
				self.try_set_is_int(fields, 0)
			if (reg.get_bit_size() == 16) != (not fields['Ds']):
				raise Exception(f"Size of {opstr} doesn't match size of output")
			self.encode_reg(fields, reg)
		elif opstr in float_immediate_lookup:
			self.try_set_is_int(fields, 0)
			self.encode_imm(fields, float_immediate_lookup[opstr])
		else:
			imm = try_parse_integer(opstr)
			if imm is not None:
				self.try_set_is_int(fields, 1)
				self.encode_imm(fields, imm)
			else:
				raise Exception(f'invalid CmpSrcDesc {opstr}')
		del fields[self.name + 's']

CMPSEL_CC = {
	0x0: 'fgtn',
	0x1: 'fltn',
	0x2: 'fgt',
	0x3: 'flt',
	0x4: 'ugt',
	0x5: 'ult',
	0x6: 'sgt',
	0x7: 'slt',
	0x8: 'feq',
	# 0x9: 'fneq?',
	0xa: 'fge',
	0xb: 'fle',
	# 0xc: ?,
	# 0xd: ?,
	0xe: 'test', # (a & b) != 0
	0xf: 'ieq',
}

class CmpSelInstructionBase(MaskedInstructionDesc):
	def __init__(self, name, size):
		super().__init__(name, size=size)
		self.add_constant(0, 3, 0b010)
		self.add_field(18, 1, 'Di') # Is output (and therefore sel inputs) integer?

class CmpSel6InstructionDesc(CmpSelInstructionBase):
	documentation_begin_group = 'Compare-Select Instructions'
	cc_to_minmax = {
		0: 'fmax',
		1: 'fmin',
		2: 'fmaxsse', # Like SSE maxps (a > b ? a : b)
		3: 'fminsse', # Like SSE minps (a < b ? a : b)
		4: 'umax',
		5: 'umin',
		6: 'smax',
		7: 'smin'
	}
	minmax_to_cc = { v: k for k, v in cc_to_minmax.items() }

	def __init__(self):
		super().__init__('cmpsel', size=6)
		self.add_constant(16, 1, 0) # Length = 6
		self.add_operand(EnumDesc('cc', 32, 3, CMPSEL_CC))
		self.add_operand(VariableDstDesc('D', l_off=3, s_off=17, u_off=38, h_off=44))
		self.add_operand(CmpSrcDesc('A',  9, common_layout='A', l_off=35))
		self.add_operand(CmpSrcDesc('B', 25, common_layout='B', l_off=36))
		self.add_operand(WaitDesc('W', 45))
		# Shadow operands for making a virtual full 5-arg cmpsel
		self.sel_a = SelSrcDesc('A',  9, common_layout='A', l_off=35)
		self.sel_b = SelSrcDesc('B', 25, common_layout='B', l_off=36)

	def is_cursed(self, fields):
		a_imm = fields['Au'] and fields['Ac']
		b_imm = fields['Bu'] and fields['Bc']
		# Input and output size doesn't match
		if (fields['As'] != fields['Ds'] and not a_imm) or (fields['Bs'] != fields['Ds'] and not b_imm):
			return True
		# Things can go funny if cmp and sel integer-ness don't match
		if ((fields['cc'] & 4) != 0) != (fields['Di'] != 0):
			# Negate affects floating point and integer fields differently
			if fields['Bn']:
				return True
			# Immediates are interpreted differently for integer fields
			if a_imm or b_imm:
				return True
		return False

	def fields_to_operands(self, fields):
		operands = super().fields_to_operands(fields)
		if self.is_cursed(fields):
			# Insert the select operands to make a full cmpsel
			operands[4:4] = [self.sel_a.decode(fields), self.sel_b.decode(fields)]
		else:
			# cc will be converted to mnemonic
			del operands[0]
		return operands

	def fields_to_mnem_base(self, fields):
		if self.is_cursed(fields):
			return 'cmpsel'
		return self.cc_to_minmax[fields['cc']]

	def can_alias_fields(self, fields, cmp, sel):
		if fields[cmp] != fields[sel]:
			return False
		if fields[cmp + 'h'] != fields[sel + 'h']:
			return False
		if fields[cmp + 'u'] != fields[sel + 'u']:
			return False
		# Usually we can combine cache flags, but if u, cache flag is actually an imm flag
		if fields[cmp + 'u'] and (fields[cmp + 'c'] != fields[sel + 'c']):
			return False
		return True

	def remove_fields(self, fields, sel):
		del fields[sel]
		del fields[sel + 'h']
		del fields[sel + 'u']
		del fields[sel + 'c']
		del fields[sel + 'd']

	def remove_x_y_wm(self, fields):
		self.remove_fields(fields, 'X')
		self.remove_fields(fields, 'Y')
		del fields['Yn']
		del fields['Wm']

	def can_encode_fields(self, fields):
		if bit_count(fields['Wm']) > 1:
			return False
		if not self.can_alias_fields(fields, 'A', 'X'):
			return False
		if not self.can_alias_fields(fields, 'B', 'Y'):
			return False
		if fields['Bn'] != fields['Yn']:
			# If we can set Di, we can prevent the n flag from doing anything to Y
			if fields['Yn'] or not fields.get('Di', 1):
				return False
		fields = dict(fields)
		self.remove_x_y_wm(fields)
		return super().can_encode_fields(fields)

	def encode_fields(self, fields):
		fields['W'] = fields['Wm'].bit_length()
		fields['Ac'] |= fields['Xc']
		fields['Ad'] |= fields['Xd']
		fields['Bc'] |= fields['Yc']
		fields['Bd'] |= fields['Yd']
		if 'Di' not in fields:
			fields['Di'] = fields['Bn'] and not fields['Yn']
		self.remove_x_y_wm(fields)
		return super().encode_fields(fields)

	pseudocode = '''
	D = A cc B ? A : B
	# Note: The LHS A and B are treated as CmpSrc, but the RHS are treated as SelSrc, so they may not actually be identical
	'''

class CmpSel8InstructionDesc(CmpSelInstructionBase):
	invert = {
		'fgtn': '!fgtn',
		'fltn': '!fltn',
		'fgt':  '!fgt',
		'flt':  '!flt',
		'ugt':  'ule',
		'ult':  'uge',
		'sgt':  'sle',
		'slt':  'sge',
		'feq':  'fne',
		'fge':  '!fge',
		'fle':  '!fle',
		'test': '!test',
		'ieq':  'ine'
	}
	invert_invert = { v: k for k, v in invert.items() }

	def __init__(self):
		super().__init__('cmpsel', size=8)
		self.add_constant(16, 1, 1) # Length > 6
		self.add_constant(33, 1, 0) # Length = 8

		cc_to_name = {}
		for k, v in CMPSEL_CC.items():
			# Treat the 8-byte instruction as a cmov, which means Z turns into a negation of the cc
			cc_to_name[k] = v
			cc_to_name[k + 0x10] = self.invert[v]
		self.add_operand(EnumDesc('cc', [
			(48, 3, 'cc'),
			(34, 1, 'ccx'),
			(32, 1, 'Z'),
		], None, cc_to_name))
		self.add_operand(VariableDstDesc('D', l_off=3, s_off=17, u_off=54, h_off=60))
		self.add_operand(CmpSrcDesc('A',  9, common_layout='A'))
		self.add_operand(CmpSrcDesc('B', 25, common_layout='B'))
		self.add_operand(SelSrcDesc('X', 41, common_layout='X'))
		self.add_operand(WaitDesc('W', 61))

	def fields_to_mnem_base(self, fields):
		return 'cmov'

	def can_alias_field(self, fields, field):
		# Dst can't be constant, negated, or abs'd
		if fields[field + 'u'] or fields[field + 'n'] or fields[field + 'a']:
			return False
		# Ignore discard, if we're writing to it it'll be discarded regardless of the flag
		# Ignore cache flag as well, we can merge those
		return (fields[field] | (fields[field + 'h'] << 7)) == fields['D']

	def remove_y_wm(self, fields):
		del fields['Y']
		del fields['Yh']
		del fields['Yc']
		del fields['Yd']
		del fields['Yu']
		del fields['Yn']
		del fields['Ya']
		del fields['Wm']

	def can_encode_fields(self, fields):
		if bit_count(fields['Wm']) > 1:
			return False
		if not self.can_alias_field(fields, 'X') and not self.can_alias_field(fields, 'Y'):
			return False
		fields = dict(fields)
		self.remove_y_wm(fields)
		return super().can_encode_fields(fields)

	def encode_fields(self, fields):
		fields['W'] = fields['Wm'].bit_length()
		if self.can_alias_field(fields, 'Y'):
			fields['Dc'] |= fields['Yc']
		else:
			fields['Dc'] |= fields['Xc']
			fields['cc'] |= 0x10
			fields['X']  = fields['Y']
			fields['Xh'] = fields['Yh']
			fields['Xc'] = fields['Yc']
			fields['Xd'] = fields['Yd']
			fields['Xu'] = fields['Yu']
			fields['Xn'] = fields['Yn']
			fields['Xa'] = fields['Ya']
		self.remove_y_wm(fields)
		if 'Di' not in fields:
			fields['Di'] = 0
		return super().encode_fields(fields)

	pseudocode = '''
	if Z == 0:
		D = A cc B ? X : D
	else:
		D = A cc B ? D : X
	'''

class CmpSel10InstructionDesc(EncodeWmAsWHelper, CmpSelInstructionBase):
	def __init__(self):
		super().__init__('cmpsel', size=10)
		self.add_constant(16, 1, 1) # Length > 6
		self.add_constant(32, 2, 0b10) # Length = 10
		self.add_operand(EnumDesc('cc', [
			(48, 3, 'cc'),
			(34, 1, 'ccx'),
		], None, CMPSEL_CC))
		self.add_operand(VariableDstDesc('D', l_off=3, s_off=17, u_off=54, h_off=60))
		self.add_operand(CmpSrcDesc('A',  9, common_layout='A', n_off=65))
		self.add_operand(CmpSrcDesc('B', 25, common_layout='B'))
		self.add_operand(SelSrcDesc('X', 41, common_layout='X'))
		self.add_operand(SelSrcDesc('Y', 73, common_layout='Y'))
		self.add_operand(WaitDesc('W', 61))

	def encode_fields(self, fields):
		if 'Di' not in fields:
			fields['Di'] = 0
		return super().encode_fields(fields)

	pseudocode = '''
	D = A cc B ? X : Y
	'''

class CmpSel14InstructionDesc(CmpSelInstructionBase):
	def __init__(self):
		super().__init__('cmpsel', size=14)
		self.add_constant(16, 1, 1) # Length > 6
		self.add_constant(32, 2, 0b11) # Length = 14
		self.add_operand(EnumDesc('cc', [
			(48, 3, 'cc'),
			(34, 1, 'ccx'),
		], None, CMPSEL_CC))
		self.add_operand(VariableDstDesc('D', l_off=3, s_off=17, u_off=54, h_off=60, z_off=82))
		self.add_operand(CmpSrcDesc('A',  9, common_layout='A', a_off=96, n_off=65))
		self.add_operand(CmpSrcDesc('B', 25, common_layout='B', a_off=97))
		self.add_operand(SelSrcDesc('X', 41, common_layout='X'))
		self.add_operand(SelSrcDesc('Y', 73, common_layout='Y'))
		self.add_operand(WaitDesc('W', 61, 93))

	def encode_fields(self, fields):
		if 'Di' not in fields:
			fields['Di'] = 0
		return super().encode_fields(fields)

	pseudocode = '''
	D = A cc B ? X : Y
	'''

@register
class CmpSelInstructionDesc(InstructionGroup):
	def __init__(self):
		super().__init__('cmpsel', [
			CmpSel6InstructionDesc(),
			CmpSel8InstructionDesc(),
			CmpSel10InstructionDesc(),
			CmpSel14InstructionDesc(),
		])

	def fields_for_mnem(self, mnem, operand_strings):
		if mnem in CmpSel6InstructionDesc.minmax_to_cc:
			cc = CmpSel6InstructionDesc.minmax_to_cc[mnem]
			operand_strings.insert(0, CMPSEL_CC[cc])
			operand_strings[4:4] = operand_strings[2:4]
			return {}
		elif mnem == 'cmov':
			if operand_strings[0] in CmpSel8InstructionDesc.invert:
				operand_strings.insert(5, operand_strings[1])
			else:
				operand_strings[0] = CmpSel8InstructionDesc.invert_invert[operand_strings[0]]
				operand_strings.insert(4, operand_strings[1])
			return {}
		elif mnem == 'cmpsel':
			return {}

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

