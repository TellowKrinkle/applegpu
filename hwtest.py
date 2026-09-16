import struct
import subprocess
import hashlib
import os
import tempfile
import applegpu
import assemble
import argparse
import hwtestbed
# TODO: this could be vastly improved by using the assembler for things

# TODO: many of these tests miss edge cases - it'd be great to do floating
# point edge case combinations, and verify that breaking FTZ behaviour or
# rounding actually breaks the tests. And lots of little oversights like
# test_exec_ops ignoring all floating point exec operations.

VERBOSE = False

parser = argparse.ArgumentParser(description='AGX test suite')
parser.add_argument('-t', '--tempdir', help='Directory to place temporary files')
args = parser.parse_args()

tmpdir = tempfile.TemporaryDirectory(dir=args.tempdir)
testbed = hwtestbed.HWTestBed(os.path.join(tmpdir.name, 'compute.metallib'))

os.chdir('hwtestbed')

def slice_length(item, offset, length):
	return item[offset:offset+length]

def mov_imm(reg, value):
	return b'\x62' + struct.pack('<BI', reg << 2 | 1, value)

def test(test_opcodes, state=None, n=32, extra_data=None):
	# load r0-r7 from buffer 7
	# r8 = thread position in grid
	code = bytes.fromhex(
		'fca05006'                     # get_sr 0, r31, sr32 (thread_position_in_grid.x)
		'0604060006000600'             # wait_long 0
		'9f11543e0200f8c81004'         # iadd r31, 0, r31, lsl 1, wait 0
		'67005400061f2000570100404000' # device_load 5, i32, xyzw, r0_r1_r2_r3, u12_u13, r31, lsl 4,   0, 1, 1, 0
		'67005408061f2000570102404000' # device_load 5, i32, xyzw, r4_r5_r6_r7, u12_u13, r31, lsl 4,  16, 1, 1, 0
		'0680060006000600'             # wait_long 5
		'0b0001c0'                     # mov r0, r0, wait 5
	)

	for i in range(8, 16):
		code += bytes((0xc + i * 16, 0)) # mov_imm r[i], 0
	for i in range(15):
		code += bytes((0xb + i * 16, 0, 0x40, 0)) # bitop_unary zero, r[16 + i]
	code += bytes.fromhex(
		'0680060006000600' # wait_long 5
		'fb0040c0'         # bitop_unary zero, r31, wait 5
	)

	code += test_opcodes

	code += bytes.fromhex(
		'fca05006'                     # get_sr 0, r31, sr32 (thread_position_in_grid.x)
		'06fc060006000600'             # wait_long 012345
		'9ff1573e0200f8c81014'         # iadd r31, 0, r31, lsl 3, wait 012345
		'e7005400071f2000170000101000' # device_store 0, i32, xyzw, r0_r1_r2_r3,     u14_u15, r31, lsl 4,   0, 1, 1
		'e7005408071f2000178000101000' # device_store 0, i32, xyzw, r4_r5_r6_r7,     u14_u15, r31, lsl 4,  16, 1, 1
		'e7005410071f2000170001101000' # device_store 0, i32, xyzw, r8_r9_r10_r11,   u14_u15, r31, lsl 4,  32, 1, 1
		'e7005418071f2000178001101000' # device_store 0, i32, xyzw, r12_r13_r14_r15, u14_u15, r31, lsl 4,  48, 1, 1
		'e7005420071f2000170002101000' # device_store 0, i32, xyzw, r16_r17_r18_r19, u14_u15, r31, lsl 4,  64, 1, 1
		'e7005428071f2000178002101000' # device_store 0, i32, xyzw, r20_r21_r22_r23, u14_u15, r31, lsl 4,  80, 1, 1
		'e7005430071f2000170003101000' # device_store 0, i32, xyzw, r24_r25_r26_r27, u14_u15, r31, lsl 4,  96, 1, 1
		'e7005438071f2000178003101000' # device_store 0, i32, xyzw, r28_r29_r30_r31, u14_u15, r31, lsl 4, 112, 1, 1
		'0e000000'                     # stop
	)

	sz = 32 * 4 * n
	buf_sz = 4 * 4 * n

	assert len(code) <= 1024

	buffer = bytearray()
	for i in range(n):
		for j in range(8):
			buffer.extend(struct.pack('=I', state[j][i]))
	request = hwtestbed.HWTestBedRequest(shader=code, tg_size=(n, 1, 1), tgsm=0x100)
	request.set_buffer(6, buffer)
	request.request_result(7, sz)
	if extra_data:
		request.set_buffer(0, extra_data)
		request.request_result(0, len(extra_data))
	result = testbed.run(request)

	result_registers = result.buffers[7]
	res = [[struct.unpack('=I', slice_length(result_registers, (j * 32 + i) * 4, 4))[0] for j in range(n)] for i in range(32)]
	# Add the extra memory to the end
	res.append(result.buffers[0] if extra_data else b'')
	return res

def core_state_to_state(cs):
	state = []
	for regid in range(8):
		state.append([cs.get_reg32(regid, thread) for thread in range(applegpu.SIMD_WIDTH)])
	return state

def diff_core_state_and_state(cs, state):
	matches = True
	for regid in range(8):
		for thread in range(applegpu.SIMD_WIDTH):
			sim, real = cs.get_reg32(regid, thread), state[regid][thread]
			if sim != real:
				print('[%d][%2d]: real=0x%x sim=0x%x' % (regid, thread, real, sim))
				matches = False
		mem = state[32]
		for i in range(0, len(mem), 4):
			real = struct.unpack('<I', mem[i:i+4])
			sim = cs.device_memory.get_u32(TEST_ADDRESS + 512 + i)
			if sim != real:
				print('Mem[%3d]: real=0x%x sim=0x%x' % (512 + i, real, sim))
				matches = False
	return matches

def cs_set(cs, state):
	for regid in range(8):
		for thread in range(applegpu.SIMD_WIDTH):
			cs.set_reg32(regid, thread, state[regid][thread])

RANDOM_INITIAL_STATE = [
	[0x90384f9b, 0x60f38a27, 0x06d347a7, 0xbcbb809c, 0xabb815ca, 0xce36a5d6, 0x05a836bb, 0x0829174a, 0x226abd8e, 0xe0cfe7da, 0xb3951896, 0x288932c9, 0x21e8f9c0, 0xd03ef2fc, 0xea3f5e6a, 0x720cc3a5, 0xb7b38620, 0x78f33846, 0xf81706c3, 0x64cc700e, 0x34193026, 0x7a4eca20, 0x31c41fa5, 0x40ad8608, 0xeff60911, 0xae940dfc, 0x6b27e2a1, 0x41e30434, 0x519265d9, 0xdcaeba8a, 0x5d85c7e0, 0x9abe6bb2],
	[0x90072532, 0xc0abec7a, 0x10f82c3c, 0xd1976ce0, 0xa34b1fa1, 0x0b36cc6e, 0xf27c9241, 0x9f0b9eee, 0x35ed4cf6, 0xa9a5a5f5, 0x58249c3b, 0x49444952, 0xcdd312db, 0x8cdf6dcb, 0xeba911bc, 0x82899c0c, 0xc73e6ce4, 0xb0ecfb51, 0x69b3b862, 0x836551d9, 0x64870451, 0x01711baa, 0x5c3c0dac, 0x761bc043, 0x338a0fe9, 0xf0093278, 0x3359b425, 0x93673a15, 0x69f048b4, 0xe7ac5f1c, 0x51d64884, 0xd86d0523],
	[0xd21acf10, 0xe4fbaf23, 0x14e37c2d, 0x89d1346e, 0x8550cff6, 0x43eb4557, 0xa491733c, 0xd2e4eeb7, 0x7e5c4d3c, 0x1ae6c83c, 0xe008794d, 0x2cd0b430, 0x202e4548, 0xec9d978b, 0x2c1a0205, 0x094dc9af, 0x0b6c7a68, 0xeda74740, 0x5a9daf3f, 0xb8702602, 0x32ebd408, 0x2f8e96d3, 0xd9b8acbd, 0xd34c857d, 0x5b43ecee, 0x645f195d, 0x1b1c362f, 0x169e073a, 0x38b9873c, 0x7cc6482b, 0x72659f90, 0x327b3e3a],
	[0x12a37385, 0xae245c4e, 0x425fd3a9, 0xf6cf1f7c, 0xad5dc3ed, 0x8a2614db, 0x0293fd10, 0x32d5b69f, 0xac007dca, 0xe944a723, 0xc3db8f63, 0x4064a265, 0x07144789, 0x369e71bd, 0x55d1d20f, 0x884f1c82, 0xbf7b7298, 0x088dc44e, 0x7fa663f3, 0x74ec2f93, 0xba560031, 0x9633ad23, 0x4d3e7422, 0xea35c243, 0x08cbdb0a, 0x24196a7c, 0x556e295f, 0x75348ccf, 0x654f74bf, 0xc5b1648b, 0x71abaac1, 0x8ad4d76c],
	[0xb381e0b9, 0x9dd50cde, 0x60ee9f42, 0x7d8c3062, 0xfb16add6, 0xf6ec65ac, 0x014226bf, 0xfe89cdc2, 0x922988f3, 0x74996a87, 0x5b03bdcc, 0x786a3b16, 0x3f9b0268, 0xe8c1f024, 0x09c74c24, 0x3964e3cb, 0x89211bc8, 0x680b4184, 0x93e2f320, 0x97b82f86, 0xc72e4296, 0x315d8ed0, 0x49ce223e, 0xec4d0ba2, 0x7fd019ea, 0x9965cb6e, 0x0e58fe52, 0x1d692839, 0x7d29c12a, 0xb3209888, 0x7a2b260a, 0x57941327],
	[0xb80a0170, 0x73c4708f, 0x26c32578, 0x0c3faf4a, 0xfae79c53, 0x2c827f6b, 0x136aaa54, 0x88b09345, 0xae17ef58, 0x75e60068, 0xa1045c64, 0xd56274ff, 0xaa672a81, 0x398a2db7, 0xcbf353ab, 0x13fda9c8, 0x5dbb50cf, 0x825e7397, 0x4dec564e, 0xeea99c9d, 0x927909b9, 0xdef6bfd7, 0xfcdfeb69, 0x00911dd0, 0xbf776e91, 0x55d23f32, 0x54c70e14, 0xcdf2f665, 0x0ef6cbda, 0xbe691a7e, 0xcdca10c0, 0xc8a38a84],
	[0xe15360d4, 0xb818c6c0, 0x2c1468d2, 0x29561ee4, 0x76104eb1, 0xce035d3c, 0x999a54d6, 0x27d6f4b0, 0x1adc45b5, 0x5d0da72c, 0x5f68eff3, 0x75fef63d, 0xe6a77ae9, 0xbc4c69ce, 0xab42af19, 0x1fae7813, 0xf8a3c2ca, 0x0c748879, 0xf38b6432, 0x4cc9e6b0, 0x63bfca87, 0x89cad78f, 0x59fc6cbf, 0x0d301561, 0x661d0897, 0x062fb2c2, 0xda2f8725, 0x751701fc, 0x851d84e3, 0xeb312e5d, 0xe381628a, 0x39360174],
	[0x8993daef, 0xa13aa4e4, 0x42415833, 0x44d8451f, 0x68eaeeb0, 0x285ff866, 0x0a976bc1, 0xdfbc0f57, 0xabf29785, 0x4c9ddf17, 0xfdd8b8bb, 0xcd15ec3e, 0xc007ec69, 0xba14bf2b, 0x5847b51a, 0x596f041f, 0x0af218fd, 0x6e1e75b4, 0x001639cb, 0x51c0c43a, 0x4c309c03, 0x96a2e740, 0xaba775f9, 0xd62150ac, 0x61d48cd9, 0x718a31b7, 0x9ba190b4, 0x5376eea9, 0x9a8c75a8, 0x70ab10dd, 0x72165574, 0xde330fa0],
]

TEST_ADDRESS = 0x55FF000000

def run_test(instructions, state, device_memory=None, extra_data=None):
	uniforms = applegpu.Uniforms()
	uniforms.set_reg64(0, TEST_ADDRESS)
	# threads per grid
	uniforms.set_reg32(20, 32)
	uniforms.set_reg32(21, 1)
	uniforms.set_reg32(22, 1)
	cs = applegpu.CoreState(uniforms=uniforms, device_memory=device_memory)


	cs_set(cs, state)

	result = test(instructions, state=core_state_to_state(cs), extra_data=extra_data)

	remaining = instructions
	while remaining:
		n = applegpu.opcode_to_number(remaining)
		desc = applegpu.get_instruction_descriptor(n)
		size = desc.decode_size(n)
		if VERBOSE:
			hexdump = ' '.join(f'{b:02x}' for b in remaining[:size])
			disasm  = desc.disassemble(applegpu.opcode_to_number(remaining[:size]))
			print(f'{hexdump:41s} {disasm}')
		desc.exec(n, cs)
		assert size <= len(remaining)
		assert size >= 2 and size % 2 == 0
		remaining = remaining[size:]

	if not diff_core_state_and_state(cs=cs, state=result):
		print('bad:', instructions.hex(), desc.disassemble(n))
	else:
		pass

def split_bits(val, nbits):
	return [(val >> i) & 1 for i in range(nbits)]

def test_bitop():
	for encoding in ('03000000', '03000200', '03000600000000000000'):
		nbytes = len(encoding) // 2
		n = applegpu.opcode_to_number(bytes.fromhex(encoding))
		desc = applegpu.get_instruction_descriptor(n)
		unary = encoding == '03000000'
		if nbytes == 10:
			ops = range(16)
		elif unary:
			ops = (0, 1, 5)
		else:
			ops = (2, 3, 4)
		for op in ops:
			if nbytes == 10:
				if op == 3: continue # Hangs GPU
				tt0, tt1, tt2, tt3 = split_bits(op, 4)
				n = desc.patch_fields(n, {'tt0': tt0, 'tt1': tt1, 'tt2': tt2, 'tt3': tt3})
			else:
				n = desc.patch_fields(n, {'op': op})

			for sizes in range(2 if unary else 8):
				Ds, As, Bs = split_bits(sizes, 3)
				if unary:
					n = desc.patch_fields(n, {'Ds': Ds})
				else:
					n = desc.patch_fields(n, {'Ds': Ds, 'As': As, 'Bs': Bs})
				for nl in range(3 if unary else 1 if nbytes < 10 else 4):
					Dl, Al, Bl = (nl > i for i in range(3))
					for d, a, b in [(3, 1, 2), (3, 2, 0)]:
						if unary:
							n = desc.patch_fields(n, {'D': d * 2 + Dl, 'A': a * 2 + Al})
						elif nbytes < 10:
							n = desc.patch_fields(n, {'D': d, 'A': a, 'B': b})
						else:
							n = desc.patch_fields(n, {'D': d * 2 + Dl, 'A': a * 2 + Al, 'B': b * 2 + Bl})
						for imm in range(2 if unary else 1 if nbytes < 10 else 3):
							aimm, bimm = (imm == i for i in range(1, 3))
							if unary:
								n = desc.patch_fields(n, {'Au': aimm, 'Ac': aimm})
							elif nbytes >= 10:
								n = desc.patch_fields(n, {'Au': aimm, 'Ac': aimm, 'Bu': bimm, 'Bc': bimm})
							run_test(desc.to_bytes(n), RANDOM_INITIAL_STATE)

def test_add():
	for encoding in ('1f015400020000001105', '1f015400020000001005', '1f015400020000101005'):
		n = applegpu.opcode_to_number(bytes.fromhex(encoding))
		base    = encoding == '1f015400020000001105'
		shifted = encoding == '1f015400020000001005'
		wide    = encoding == '1f015400020000101005'
		desc = applegpu.get_instruction_descriptor(n)
		for sizes in range(9 if wide else 8):
			if wide:
				n = desc.patch_fields(n, {'As': sizes % 3, 'Bs': sizes // 3})
			else:
				Ds, As, Bs = split_bits(sizes, 3)
				n = desc.patch_fields(n, {'Ds': Ds, 'As': As, 'Bs': Bs})
			for params in range(8):
				Asx, Bsx, P = split_bits(params, 3)
				n = desc.patch_fields(n, {'P': P, 'Asx': Asx, 'Bsx': Bsx})
				for nl in range(4):
					Dl, Al, Bl = (nl > i for i in range(3))
					for d, a, b in [(6, 2, 4)]:
						n = desc.patch_fields(n, {'D': d * 2 + Dl, 'A': a * 2 + Al, 'B': b * 2 + Bl})
						for imm in range(3):
							aimm, bimm = (imm == i for i in range(1, 3))
							if aimm and Asx or bimm and Bsx: continue
							n = desc.patch_fields(n, {'Ar': not aimm, 'Br': not bimm})
							for extra in range(1 if wide else 4 if shifted else 2):
								if base:
									n = desc.patch_fields(n, {'S': extra})
								elif shifted:
									n = desc.patch_fields(n, {'s': extra})
								run_test(desc.to_bytes(n), RANDOM_INITIAL_STATE)

def test_madd():
	n = applegpu.opcode_to_number(bytes.fromhex('9f0054000200000000202a00'))
	desc = applegpu.get_instruction_descriptor(n)
	for sizes in range(24):
		Ds = sizes % 3
		As, Bs, Cs = split_bits(sizes // 3, 3)
		n = desc.patch_fields(n, {'Ds': Ds, 'As': As, 'Bs': Bs, 'Cs': Cs})
		for params in range(16):
			Asx, Bsx, Csx, P = split_bits(params, 4)
			n = desc.patch_fields(n, {'P': P, 'Asx': Asx, 'Bsx': Bsx, 'Csx': Csx})
			for nl in range(5):
				Dl, Al, Bl, Cl = (nl > i for i in range(4))
				for d, a, b, c in [(6, 0, 2, 4)]:
					n = desc.patch_fields(n, {'D': d * 2 + Dl, 'A': a * 2 + Al, 'B': b * 2 + Bl, 'C': c * 2 + Cl})
					for imm in range(4):
						aimm, bimm, cimm = (imm == i for i in range(1, 4))
						if aimm and Asx or bimm and Bsx or cimm and Csx: continue
						n = desc.patch_fields(n, {'Ar': not aimm, 'Br': not bimm, 'Cr': not cimm})
						run_test(desc.to_bytes(n), RANDOM_INITIAL_STATE)

def test_fmadd():
	n = applegpu.opcode_to_number(bytes.fromhex('3aad5ca2255e0200'))
	desc = applegpu.get_instruction_descriptor(n)
	assert desc.decode_remainder(n) == 0, hex(desc.decode_remainder(n))

	for sizes in range(8):
		Dt = 2 if sizes & 1 else 0
		At = 9 if sizes & 2 else 1
		Ct = 9 if sizes & 4 else 1
		for Bt in [0, 1, 9]:
			n = desc.patch_fields(n, {'Dt': Dt, 'At': At, 'Bt': Bt, 'Ct': Ct})
			for Bm in range(4):
				n = desc.patch_fields(n, {'Bm': Bm})
				for (d,a,b,c) in [(6, 0, 2, 4)]:
					d_ors = [0] #[0,1] if Dt == 2 else [0]
					for d_or in d_ors:
						D = (d << 1) | d_or
						n = desc.patch_fields(n, {'D': D, 'A': a << 1, 'B': b << 1, 'C': c << 1})
						for S in [1, 0]:
							n = desc.patch_fields(n, {'S': S})
							run_test(desc.to_bytes(n), RANDOM_INITIAL_STATE)


def test_fadd():
	n = applegpu.opcode_to_number(bytes.fromhex('2aad5ec22500'))
	desc = applegpu.get_instruction_descriptor(n)
	assert desc.decode_remainder(n) == 0, hex(desc.decode_remainder(n))

	for sizes in range(4):
		Dt = 2 if sizes & 1 else 0
		At = 9 if sizes & 2 else 1
		for Bt in [0, 1, 9]:
			n = desc.patch_fields(n, {'Dt': Dt, 'At': At, 'Bt': Bt})
			for Bm in range(4):
				n = desc.patch_fields(n, {'Bm': Bm})
				for (d,a,b) in [(6, 0, 2)]:
					d_ors = [0]
					for d_or in d_ors:
						D = (d << 1) | d_or
						n = desc.patch_fields(n, {'D': D, 'A': a << 1, 'B': b << 1})
						for S in [1, 0]:
							n = desc.patch_fields(n, {'S': S})
							run_test(desc.to_bytes(n), RANDOM_INITIAL_STATE)

def test_fmul():
	n = applegpu.opcode_to_number(bytes.fromhex('1aad5ec22500'))
	desc = applegpu.get_instruction_descriptor(n)
	assert desc.decode_remainder(n) == 0, hex(desc.decode_remainder(n))

	for sizes in range(4):
		Dt = 2 if sizes & 1 else 0
		At = 9 if sizes & 2 else 1
		for Bt in [0, 1, 9]:
			n = desc.patch_fields(n, {'Dt': Dt, 'At': At, 'Bt': Bt})
			for Bm in range(4):
				n = desc.patch_fields(n, {'Bm': Bm})
				for (d,a,b) in [(6, 0, 2)]:
					d_ors = [0]
					for d_or in d_ors:
						D = (d << 1) | d_or
						n = desc.patch_fields(n, {'D': D, 'A': a << 1, 'B': b << 1})
						for S in [1, 0]:
							n = desc.patch_fields(n, {'S': S})
							run_test(desc.to_bytes(n), RANDOM_INITIAL_STATE)

def test_fmadd16():
	n = applegpu.opcode_to_number(bytes.fromhex('362c5dc0055e'))
	desc = applegpu.get_instruction_descriptor(n)
	assert desc.decode_remainder(n) == 0, hex(desc.decode_remainder(n))
	for Dt in [2, 0]:
		n = desc.patch_fields(n, {'Dt': Dt})
		At = 1
		Bt = 1
		Ct = 1
		n = desc.patch_fields(n, {'At': At, 'Bt': Bt, 'Ct': Ct})
		for Am in range(4):
			for Bm in range(4):
				for Cm in range(4):
					n = desc.patch_fields(n, {'Bm': Bm, 'Am': Am, 'Cm': Cm})
					for (d,a,b,c) in [(6, 0, 2, 4)]:
						d_ors = [0]
						for d_or in d_ors:
							D = (d << 1) | d_or
							n = desc.patch_fields(n, {'D': D, 'A': a << 1, 'B': b << 1, 'C': c << 1})
							for S in [1, 0]:
								n = desc.patch_fields(n, {'S': S})
								run_test(desc.to_bytes(n), RANDOM_INITIAL_STATE)


def test_shift():
	# TODO: rewrite in terms of the new instructions
	class ShiftInstructionDesc(applegpu.MaskedInstructionDesc):
		def __init__(self):
			super().__init__('shift', size=8)
			self.add_constant(0, 7, 0x2E)

			self.add_operand(applegpu.ImmediateDesc('i0', 15, 1))
			self.add_operand(applegpu.ImmediateDesc('i1', 26, 2))

			self.add_operand(applegpu.ALUDstDesc('D', 60))
			self.add_operand(applegpu.ALUSrcDesc('A', 16, 58))
			self.add_operand(applegpu.ALUSrcDesc('B', 28, 56))
			self.add_operand(applegpu.ALUSrcDesc('C', 40, 54))

			self.add_operand(applegpu.MaskDesc('m'))

	n = applegpu.opcode_to_number(bytes.fromhex('2e2d00c025460000'))
	desc = ShiftInstructionDesc()
	assert desc.decode_remainder(n) == 0, hex(desc.decode_remainder(n))
	for i in range(32*2):
		i0 = i & 1
		i1 = (i >> 1) & 3
		i2 = i >> 3
		m = i2 * 4
		if (i0, i1) == (0, 3):
			continue
		n &= ~(1 << 15)
		n &= ~(3 << 26)
		n |= i0 << 15
		n |= i1 << 26
		n = desc.patch_fields(n, {'m': m})
		for Dt in [2, 0]:
			n = desc.patch_fields(n, {'Dt': Dt})
			for At in [9,1,0]:
				for Bt in [9,1,0]:
					for Ct in [9,1,0]:
						n = desc.patch_fields(n, {'At': At, 'Bt': Bt, 'Ct': Ct})
						for (d,a,b,c) in [(6, 0, 2, 4)]:
							d_ors = [0]
							for d_or in d_ors:
								D = (d << 1) | d_or
								n = desc.patch_fields(n, {'D': D, 'A': a << 1, 'B': b << 1, 'C': c << 1})
								run_test(desc.to_bytes(n), RANDOM_INITIAL_STATE)


def mov_reg32(dest, src):
	n = applegpu.opcode_to_number(bytes.fromhex('7e315e0a8000'))
	desc = applegpu.get_instruction_descriptor(n)
	assert desc.decode_remainder(n) == 0, hex(desc.decode_remainder(n))

	n = desc.patch_fields(n, {'D': dest << 1, 'A': src << 1})
	return desc.to_bytes(n)

def pop_exec(unk=1):
	n = applegpu.opcode_to_number(bytes.fromhex('520e00000000'))
	desc = applegpu.get_instruction_descriptor(n)
	n = desc.patch_fields(n, {'n': unk})
	return desc.to_bytes(n)


def test_icmp_ballot():
	for cc in [
		0b0000,
		0b0001,
		0b0010,
		0b0100,
		0b0101,
		0b0110,
	]:
		code = b''
		code += mov_imm(0, 0)

		for At in [1, 9]:
			for Bt in [1,9]:
				n = applegpu.opcode_to_number(bytes.fromhex('322d602226800180'))
				desc = applegpu.get_instruction_descriptor(n)
				assert desc.decode_remainder(n) == 0, hex(desc.decode_remainder(n))
				n = desc.patch_fields(n, {'A': 1 << 1, 'B': 3 << 1, 'At': At, 'Bt': Bt, 'D': 6 << 1, 'ccn': 1, 'cc': cc})
				code += desc.to_bytes(n)

				run_test(code, RANDOM_INITIAL_STATE)


def pushexec_cmp_andexec(b, n, op):
	n = applegpu.opcode_to_number(bytes.fromhex('52885a020000'))
	desc = applegpu.get_instruction_descriptor(n)
	assert desc.decode_remainder(n) == 0, hex(desc.decode_remainder(n))
	n = desc.patch_fields(n, {'cc': 0b0001, 'ccn': 0, 'A': 1 << 1,  'At': 9, 'Bt': 0, 'B': b})

	n = desc.patch_fields(n, {'n': n})
	n = n & ~(3 << 9)
	n |= (op << 9)

	return desc.to_bytes(n)

def pushexec_cmp_neq_andexec(comparison_value, count, op, equal=False):
	n = applegpu.opcode_to_number(bytes.fromhex('52885a020000'))
	desc = applegpu.get_instruction_descriptor(n)
	assert desc.decode_remainder(n) == 0, hex(desc.decode_remainder(n))
	n = desc.patch_fields(n, {'cc': 0, 'ccn': 0 if equal else 1, 'A': 1 << 1,  'At': 9, 'Bt': 0, 'B': comparison_value})

	n = desc.patch_fields(n, {'n': count})
	n = n & ~(3 << 9)
	n |= (op << 9)

	return desc.to_bytes(n)


def or_imm(reg, imm):
	n = applegpu.opcode_to_number(bytes.fromhex('7e2d582ac000'))
	desc = applegpu.get_instruction_descriptor(n)

	n = desc.patch_fields(n, {'A': reg << 1, 'D': reg << 1, 'Bt': 9, 'B': 5 << 1})
	return mov_imm(5, imm) + desc.to_bytes(n)

def move_to_lane(lane, value):
	n = applegpu.opcode_to_number(bytes.fromhex('122d583200096185'))
	desc = applegpu.get_instruction_descriptor(n)
	reg = 0
	n = desc.patch_fields(n, {'Y': reg << 1, 'D': reg << 1, 'A': 1 << 1, 'B': lane, 'X': value})
	return desc.to_bytes(n)

def test_simd_shuffle_down():
	n = applegpu.opcode_to_number(bytes.fromhex('6f2c5614c000'))
	desc = applegpu.get_instruction_descriptor(n)
	for B in range(32):
		n = desc.patch_fields(n, {'A': 5 << 1, 'D': 6 << 1, 'B': B})
		code = b''
		code += desc.to_bytes(n)
		run_test(code, RANDOM_INITIAL_STATE)

def test_simd_shuffle():
	initial_state = [
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		list(i  for i in range(32)),
		[20, 26, 14, 14, 10, 11, 8, 8, 0, 12, 0, 19, 16, 21, 25, 0, 6, 28, 2, 23, 9, 9, 29, 15, 1, 7, 14, 25, 28, 12, 12, 15],
		[0, 0xae245c4e, 0x425fd3a9, 0xf6cf1f7c, 0xad5dc3ed, 0x8a2614db, 0x0293fd10, 0x32d5b69f, 0xac007dca, 0xe944a723, 0xc3db8f63, 0x4064a265, 0x07144789, 0x369e71bd, 0x55d1d20f, 0x884f1c82, 0xbf7b7298, 0x088dc44e, 0x7fa663f3, 0x74ec2f93, 0xba560031, 0x9633ad23, 0x4d3e7422, 0xea35c243, 0x08cbdb0a, 0x24196a7c, 0x556e295f, 0x75348ccf, 0x654f74bf, 0xc5b1648b, 0x71abaac1, 0x8ad4d76c],
		[0xb381e0b9, 0x9dd50cde, 0x60ee9f42, 0x7d8c3062, 0xfb16add6, 0xf6ec65ac, 0x014226bf, 0xfe89cdc2, 0x922988f3, 0x74996a87, 0x5b03bdcc, 0x786a3b16, 0x3f9b0268, 0xe8c1f024, 0x09c74c24, 0x3964e3cb, 0x89211bc8, 0x680b4184, 0x93e2f320, 0x97b82f86, 0xc72e4296, 0x315d8ed0, 0x49ce223e, 0xec4d0ba2, 0x7fd019ea, 0x9965cb6e, 0x0e58fe52, 0x1d692839, 0x7d29c12a, 0xb3209888, 0x7a2b260a, 0x57941327],
		[0xb80a0170, 0x73c4708f, 0x26c32578, 0x0c3faf4a, 0xfae79c53, 0x2c827f6b, 0x136aaa54, 0x88b09345, 0xae17ef58, 0x75e60068, 0xa1045c64, 0xd56274ff, 0xaa672a81, 0x398a2db7, 0xcbf353ab, 0x13fda9c8, 0x5dbb50cf, 0x825e7397, 0x4dec564e, 0xeea99c9d, 0x927909b9, 0xdef6bfd7, 0xfcdfeb69, 0x00911dd0, 0xbf776e91, 0x55d23f32, 0x54c70e14, 0xcdf2f665, 0x0ef6cbda, 0xbe691a7e, 0xcdca10c0, 0xc8a38a84],
		[0xe15360d4, 0xb818c6c0, 0x2c1468d2, 0x29561ee4, 0x76104eb1, 0xce035d3c, 0x999a54d6, 0x27d6f4b0, 0x1adc45b5, 0x5d0da72c, 0x5f68eff3, 0x75fef63d, 0xe6a77ae9, 0xbc4c69ce, 0xab42af19, 0x1fae7813, 0xf8a3c2ca, 0x0c748879, 0xf38b6432, 0x4cc9e6b0, 0x63bfca87, 0x89cad78f, 0x59fc6cbf, 0x0d301561, 0x661d0897, 0x062fb2c2, 0xda2f8725, 0x751701fc, 0x851d84e3, 0xeb312e5d, 0xe381628a, 0x39360174],
		[0x8993daef, 0xa13aa4e4, 0x42415833, 0x44d8451f, 0x68eaeeb0, 0x285ff866, 0x0a976bc1, 0xdfbc0f57, 0xabf29785, 0x4c9ddf17, 0xfdd8b8bb, 0xcd15ec3e, 0xc007ec69, 0xba14bf2b, 0x5847b51a, 0x596f041f, 0x0af218fd, 0x6e1e75b4, 0x001639cb, 0x51c0c43a, 0x4c309c03, 0x96a2e740, 0xaba775f9, 0xd62150ac, 0x61d48cd9, 0x718a31b7, 0x9ba190b4, 0x5376eea9, 0x9a8c75a8, 0x70ab10dd, 0x72165574, 0xde330fa0],
	]

	n = applegpu.opcode_to_number(bytes.fromhex('6f2d56160000'))
	desc = applegpu.get_instruction_descriptor(n)
	n = desc.patch_fields(n, {'D': 6 << 1, 'Dt': 2,  'A': 1 << 1, 'B': 2 << 1, 'Bt': 1})

	code = b''
	code += mov_imm(2, 0)
	code += desc.to_bytes(n)
	run_test(code, initial_state)

	initial_state = [
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		list(i  for i in range(32)),
		[0, 1] + [0] * 30,
		[0, 0xae245c4e, 0x425fd3a9, 0xf6cf1f7c, 0xad5dc3ed, 0x8a2614db, 0x0293fd10, 0x32d5b69f, 0xac007dca, 0xe944a723, 0xc3db8f63, 0x4064a265, 0x07144789, 0x369e71bd, 0x55d1d20f, 0x884f1c82, 0xbf7b7298, 0x088dc44e, 0x7fa663f3, 0x74ec2f93, 0xba560031, 0x9633ad23, 0x4d3e7422, 0xea35c243, 0x08cbdb0a, 0x24196a7c, 0x556e295f, 0x75348ccf, 0x654f74bf, 0xc5b1648b, 0x71abaac1, 0x8ad4d76c],
		[0xb381e0b9, 0x9dd50cde, 0x60ee9f42, 0x7d8c3062, 0xfb16add6, 0xf6ec65ac, 0x014226bf, 0xfe89cdc2, 0x922988f3, 0x74996a87, 0x5b03bdcc, 0x786a3b16, 0x3f9b0268, 0xe8c1f024, 0x09c74c24, 0x3964e3cb, 0x89211bc8, 0x680b4184, 0x93e2f320, 0x97b82f86, 0xc72e4296, 0x315d8ed0, 0x49ce223e, 0xec4d0ba2, 0x7fd019ea, 0x9965cb6e, 0x0e58fe52, 0x1d692839, 0x7d29c12a, 0xb3209888, 0x7a2b260a, 0x57941327],
		[0xb80a0170, 0x73c4708f, 0x26c32578, 0x0c3faf4a, 0xfae79c53, 0x2c827f6b, 0x136aaa54, 0x88b09345, 0xae17ef58, 0x75e60068, 0xa1045c64, 0xd56274ff, 0xaa672a81, 0x398a2db7, 0xcbf353ab, 0x13fda9c8, 0x5dbb50cf, 0x825e7397, 0x4dec564e, 0xeea99c9d, 0x927909b9, 0xdef6bfd7, 0xfcdfeb69, 0x00911dd0, 0xbf776e91, 0x55d23f32, 0x54c70e14, 0xcdf2f665, 0x0ef6cbda, 0xbe691a7e, 0xcdca10c0, 0xc8a38a84],
		[0xe15360d4, 0xb818c6c0, 0x2c1468d2, 0x29561ee4, 0x76104eb1, 0xce035d3c, 0x999a54d6, 0x27d6f4b0, 0x1adc45b5, 0x5d0da72c, 0x5f68eff3, 0x75fef63d, 0xe6a77ae9, 0xbc4c69ce, 0xab42af19, 0x1fae7813, 0xf8a3c2ca, 0x0c748879, 0xf38b6432, 0x4cc9e6b0, 0x63bfca87, 0x89cad78f, 0x59fc6cbf, 0x0d301561, 0x661d0897, 0x062fb2c2, 0xda2f8725, 0x751701fc, 0x851d84e3, 0xeb312e5d, 0xe381628a, 0x39360174],
		[0x8993daef, 0xa13aa4e4, 0x42415833, 0x44d8451f, 0x68eaeeb0, 0x285ff866, 0x0a976bc1, 0xdfbc0f57, 0xabf29785, 0x4c9ddf17, 0xfdd8b8bb, 0xcd15ec3e, 0xc007ec69, 0xba14bf2b, 0x5847b51a, 0x596f041f, 0x0af218fd, 0x6e1e75b4, 0x001639cb, 0x51c0c43a, 0x4c309c03, 0x96a2e740, 0xaba775f9, 0xd62150ac, 0x61d48cd9, 0x718a31b7, 0x9ba190b4, 0x5376eea9, 0x9a8c75a8, 0x70ab10dd, 0x72165574, 0xde330fa0],
	]

	n = applegpu.opcode_to_number(bytes.fromhex('6f2d56160000'))
	desc = applegpu.get_instruction_descriptor(n)
	n = desc.patch_fields(n, {'D': 6 << 1, 'Dt': 2,  'A': 1 << 1, 'B': 2 << 1, 'Bt': 1})

	code = b''

	code = pushexec_cmp_neq_andexec(1, 1, 0)
	code += mov_imm(2, 0)
	code += desc.to_bytes(n)
	code += pop_exec(1)
	run_test(code, initial_state)


def test_exec_ops():
	initial_state = [
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		list(range(32)),
		[0xd21acf10, 0xe4fbaf23, 0x14e37c2d, 0x89d1346e, 0x8550cff6, 0x43eb4557, 0xa491733c, 0xd2e4eeb7, 0x7e5c4d3c, 0x1ae6c83c, 0xe008794d, 0x2cd0b430, 0x202e4548, 0xec9d978b, 0x2c1a0205, 0x094dc9af, 0x0b6c7a68, 0xeda74740, 0x5a9daf3f, 0xb8702602, 0x32ebd408, 0x2f8e96d3, 0xd9b8acbd, 0xd34c857d, 0x5b43ecee, 0x645f195d, 0x1b1c362f, 0x169e073a, 0x38b9873c, 0x7cc6482b, 0x72659f90, 0x327b3e3a],
		[0, 0xae245c4e, 0x425fd3a9, 0xf6cf1f7c, 0xad5dc3ed, 0x8a2614db, 0x0293fd10, 0x32d5b69f, 0xac007dca, 0xe944a723, 0xc3db8f63, 0x4064a265, 0x07144789, 0x369e71bd, 0x55d1d20f, 0x884f1c82, 0xbf7b7298, 0x088dc44e, 0x7fa663f3, 0x74ec2f93, 0xba560031, 0x9633ad23, 0x4d3e7422, 0xea35c243, 0x08cbdb0a, 0x24196a7c, 0x556e295f, 0x75348ccf, 0x654f74bf, 0xc5b1648b, 0x71abaac1, 0x8ad4d76c],
		[0xb381e0b9, 0x9dd50cde, 0x60ee9f42, 0x7d8c3062, 0xfb16add6, 0xf6ec65ac, 0x014226bf, 0xfe89cdc2, 0x922988f3, 0x74996a87, 0x5b03bdcc, 0x786a3b16, 0x3f9b0268, 0xe8c1f024, 0x09c74c24, 0x3964e3cb, 0x89211bc8, 0x680b4184, 0x93e2f320, 0x97b82f86, 0xc72e4296, 0x315d8ed0, 0x49ce223e, 0xec4d0ba2, 0x7fd019ea, 0x9965cb6e, 0x0e58fe52, 0x1d692839, 0x7d29c12a, 0xb3209888, 0x7a2b260a, 0x57941327],
		[0xb80a0170, 0x73c4708f, 0x26c32578, 0x0c3faf4a, 0xfae79c53, 0x2c827f6b, 0x136aaa54, 0x88b09345, 0xae17ef58, 0x75e60068, 0xa1045c64, 0xd56274ff, 0xaa672a81, 0x398a2db7, 0xcbf353ab, 0x13fda9c8, 0x5dbb50cf, 0x825e7397, 0x4dec564e, 0xeea99c9d, 0x927909b9, 0xdef6bfd7, 0xfcdfeb69, 0x00911dd0, 0xbf776e91, 0x55d23f32, 0x54c70e14, 0xcdf2f665, 0x0ef6cbda, 0xbe691a7e, 0xcdca10c0, 0xc8a38a84],
		[0xe15360d4, 0xb818c6c0, 0x2c1468d2, 0x29561ee4, 0x76104eb1, 0xce035d3c, 0x999a54d6, 0x27d6f4b0, 0x1adc45b5, 0x5d0da72c, 0x5f68eff3, 0x75fef63d, 0xe6a77ae9, 0xbc4c69ce, 0xab42af19, 0x1fae7813, 0xf8a3c2ca, 0x0c748879, 0xf38b6432, 0x4cc9e6b0, 0x63bfca87, 0x89cad78f, 0x59fc6cbf, 0x0d301561, 0x661d0897, 0x062fb2c2, 0xda2f8725, 0x751701fc, 0x851d84e3, 0xeb312e5d, 0xe381628a, 0x39360174],
		[0x8993daef, 0xa13aa4e4, 0x42415833, 0x44d8451f, 0x68eaeeb0, 0x285ff866, 0x0a976bc1, 0xdfbc0f57, 0xabf29785, 0x4c9ddf17, 0xfdd8b8bb, 0xcd15ec3e, 0xc007ec69, 0xba14bf2b, 0x5847b51a, 0x596f041f, 0x0af218fd, 0x6e1e75b4, 0x001639cb, 0x51c0c43a, 0x4c309c03, 0x96a2e740, 0xaba775f9, 0xd62150ac, 0x61d48cd9, 0x718a31b7, 0x9ba190b4, 0x5376eea9, 0x9a8c75a8, 0x70ab10dd, 0x72165574, 0xde330fa0],
	]

	for op in range(3):
		for eq in (True, False):
			for nnn in range(4):
				for i in range(14):
					n = applegpu.opcode_to_number(bytes.fromhex('6f2c5614c000'))
					desc = applegpu.get_instruction_descriptor(n)

					n = desc.patch_fields(n, {'A': 0 << 1, 'D': 6 << 1})
					code = b''
					N = 1
					code += mov_imm(0, 0)

					k = 0
					for lane in range(1, 10, 2):
						code += move_to_lane(lane, k)
						k += 1
					code += mov_imm(4, 0)
					code += mov_imm(6, 0)

					code += or_imm(4, N); N <<= 1;

					code += pushexec_cmp_neq_andexec(i, nnn, op, equal=eq)

					code += or_imm(4, N); N <<= 1;

					code += desc.to_bytes(n)

					for i in range(20):
						code += pop_exec(1)
						if N < (1 << 32):
							code += or_imm(4, N); N <<= 1;

					run_test(code, initial_state)

def test_fcmpsel():
	n = applegpu.opcode_to_number(bytes.fromhex('022d568225034130'))
	desc = applegpu.get_instruction_descriptor(n)
	for cc in [0, 1, 2, 5, 6]:
		n = desc.patch_fields(n, {'D': 6 << 1, 'A': 2 << 1, 'B': 4 << 1, 'cc': cc})
		code = b''
		code += desc.to_bytes(n)
		run_test(code, RANDOM_INITIAL_STATE)

def test_popcount():
	n = applegpu.opcode_to_number(bytes.fromhex('3e2d4e0a0004'))
	desc = applegpu.get_instruction_descriptor(n)
	for Dt in [0, 2]:
		d_ors = [0,1]
		for d_or in d_ors:
			D = (6 << 1) | d_or
			for At in [0, 1, 9, 0xD]:
				n = desc.patch_fields(n, {'D': D, 'A': 2 << 1, 'Dt': Dt, 'At': At})

def test_bitrev():
	n = applegpu.opcode_to_number(bytes.fromhex('3e2d4e060004'))
	desc = applegpu.get_instruction_descriptor(n)
	for Dt in [0, 2]:
		d_ors = [0,1]
		for d_or in d_ors:
			D = (6 << 1) | d_or
			for At in [0, 1, 9, 0xD]:
				n = desc.patch_fields(n, {'D': D, 'A': 2 << 1, 'Dt': Dt, 'At': At})
				code = b''
				code += desc.to_bytes(n)
				run_test(code, RANDOM_INITIAL_STATE)


def test_bitrev():
	n = applegpu.opcode_to_number(bytes.fromhex('3e2d4e060004'))
	desc = applegpu.get_instruction_descriptor(n)

	for Dt in [0, 2]:
		d_ors = [0, 1]
		for d_or in d_ors:
			D = (6 << 1) | d_or
			for At in [0, 1, 9, 0xD]:
				n = desc.patch_fields(n, {'D': D, 'A': 2 << 1, 'Dt': Dt, 'At': At})
				code = b''
				code += desc.to_bytes(n)
				run_test(code, RANDOM_INITIAL_STATE)

def test_ffs():
	initial_state = [
		[0x90384f9b, 0x60f38a27, 0x06d347a7, 0xbcbb809c, 0xabb815ca, 0xce36a5d6, 0x05a836bb, 0x0829174a, 0x226abd8e, 0xe0cfe7da, 0xb3951896, 0x288932c9, 0x21e8f9c0, 0xd03ef2fc, 0xea3f5e6a, 0x720cc3a5, 0xb7b38620, 0x78f33846, 0xf81706c3, 0x64cc700e, 0x34193026, 0x7a4eca20, 0x31c41fa5, 0x40ad8608, 0xeff60911, 0xae940dfc, 0x6b27e2a1, 0x41e30434, 0x519265d9, 0xdcaeba8a, 0x5d85c7e0, 0x9abe6bb2],
		[0x90072532, 0xc0abec7a, 0x10f82c3c, 0xd1976ce0, 0xa34b1fa1, 0x0b36cc6e, 0xf27c9241, 0x9f0b9eee, 0x35ed4cf6, 0xa9a5a5f5, 0x58249c3b, 0x49444952, 0xcdd312db, 0x8cdf6dcb, 0xeba911bc, 0x82899c0c, 0xc73e6ce4, 0xb0ecfb51, 0x69b3b862, 0x836551d9, 0x64870451, 0x01711baa, 0x5c3c0dac, 0x761bc043, 0x338a0fe9, 0xf0093278, 0x3359b425, 0x93673a15, 0x69f048b4, 0xe7ac5f1c, 0x51d64884, 0xd86d0523],
		[0, 1, 2, 0x89d1346e, 0x8550cff6, 0x43eb4557, 0xa491733c, 0xd2e4eeb7, 0x7e5c4d3c, 0x1ae6c83c, 0xe008794d, 0x2cd0b430, 0x202e4548, 0xec9d978b, 0x2c1a0205, 0x094dc9af, 0x0b6c7a68, 0xeda74740, 0x5a9daf3f, 0xb8702602, 0x32ebd408, 0x2f8e96d3, 0xd9b8acbd, 0xd34c857d, 0x5b43ecee, 0x645f195d, 0x1b1c362f, 0x169e073a, 0x38b9873c, 0x7cc6482b, 0x72659f90, 0x327b3e3a],
		[0x12a37385, 0xae245c4e, 0x425fd3a9, 0xf6cf1f7c, 0xad5dc3ed, 0x8a2614db, 0x0293fd10, 0x32d5b69f, 0xac007dca, 0xe944a723, 0xc3db8f63, 0x4064a265, 0x07144789, 0x369e71bd, 0x55d1d20f, 0x884f1c82, 0xbf7b7298, 0x088dc44e, 0x7fa663f3, 0x74ec2f93, 0xba560031, 0x9633ad23, 0x4d3e7422, 0xea35c243, 0x08cbdb0a, 0x24196a7c, 0x556e295f, 0x75348ccf, 0x654f74bf, 0xc5b1648b, 0x71abaac1, 0x8ad4d76c],
		[0xb381e0b9, 0x9dd50cde, 0x60ee9f42, 0x7d8c3062, 0xfb16add6, 0xf6ec65ac, 0x014226bf, 0xfe89cdc2, 0x922988f3, 0x74996a87, 0x5b03bdcc, 0x786a3b16, 0x3f9b0268, 0xe8c1f024, 0x09c74c24, 0x3964e3cb, 0x89211bc8, 0x680b4184, 0x93e2f320, 0x97b82f86, 0xc72e4296, 0x315d8ed0, 0x49ce223e, 0xec4d0ba2, 0x7fd019ea, 0x9965cb6e, 0x0e58fe52, 0x1d692839, 0x7d29c12a, 0xb3209888, 0x7a2b260a, 0x57941327],
		[0xb80a0170, 0x73c4708f, 0x26c32578, 0x0c3faf4a, 0xfae79c53, 0x2c827f6b, 0x136aaa54, 0x88b09345, 0xae17ef58, 0x75e60068, 0xa1045c64, 0xd56274ff, 0xaa672a81, 0x398a2db7, 0xcbf353ab, 0x13fda9c8, 0x5dbb50cf, 0x825e7397, 0x4dec564e, 0xeea99c9d, 0x927909b9, 0xdef6bfd7, 0xfcdfeb69, 0x00911dd0, 0xbf776e91, 0x55d23f32, 0x54c70e14, 0xcdf2f665, 0x0ef6cbda, 0xbe691a7e, 0xcdca10c0, 0xc8a38a84],
		[0xe15360d4, 0xb818c6c0, 0x2c1468d2, 0x29561ee4, 0x76104eb1, 0xce035d3c, 0x999a54d6, 0x27d6f4b0, 0x1adc45b5, 0x5d0da72c, 0x5f68eff3, 0x75fef63d, 0xe6a77ae9, 0xbc4c69ce, 0xab42af19, 0x1fae7813, 0xf8a3c2ca, 0x0c748879, 0xf38b6432, 0x4cc9e6b0, 0x63bfca87, 0x89cad78f, 0x59fc6cbf, 0x0d301561, 0x661d0897, 0x062fb2c2, 0xda2f8725, 0x751701fc, 0x851d84e3, 0xeb312e5d, 0xe381628a, 0x39360174],
		[0x8993daef, 0xa13aa4e4, 0x42415833, 0x44d8451f, 0x68eaeeb0, 0x285ff866, 0x0a976bc1, 0xdfbc0f57, 0xabf29785, 0x4c9ddf17, 0xfdd8b8bb, 0xcd15ec3e, 0xc007ec69, 0xba14bf2b, 0x5847b51a, 0x596f041f, 0x0af218fd, 0x6e1e75b4, 0x001639cb, 0x51c0c43a, 0x4c309c03, 0x96a2e740, 0xaba775f9, 0xd62150ac, 0x61d48cd9, 0x718a31b7, 0x9ba190b4, 0x5376eea9, 0x9a8c75a8, 0x70ab10dd, 0x72165574, 0xde330fa0],
	]

	n = applegpu.opcode_to_number(bytes.fromhex('be0d960e0000'))
	desc = applegpu.get_instruction_descriptor(n)

	for Dt in [0, 2]:
		d_ors = [0,1]
		for d_or in d_ors:
			D = (6 << 1) | d_or
			for At in [0, 1, 9, 0xD]:
				n = desc.patch_fields(n, {'D': D, 'A': 2 << 1, 'Dt': Dt, 'At': At})
				code = b''
				code += desc.to_bytes(n)
				run_test(code, initial_state)

def test_uniforms():
	# barely a test, but oh well

	# threads_per_grid constants at u20,u21,u22
	code = b''.join(assemble.assemble_line('mov r%d, u%d' % (i, i+20)) for i in range(3))
	run_test(code, RANDOM_INITIAL_STATE)

	code = b''.join(assemble.assemble_line('mov r%dh, u%dl' % (i, i+20)) for i in range(3))
	run_test(code, RANDOM_INITIAL_STATE)

def test_sr32():
	run_test(assemble.assemble_line('get_sr 0, r0, sr32'), RANDOM_INITIAL_STATE)

def test_memory():
	# TODO: document what each of these are testing
	extra_data = bytes(range(256))*2

	tests = [1, 0x3F, 0x40, 0x41, 0x1F << 6, (0x1F << 6) | 1, (0x1E << 6) | 1, (0x1E << 6) | 0x3F]
	tests += [i << 22 for i in [1, 0x1F, 0x20, 0x21, 0x1F << 5, (0x1F << 5) | 1, (0x1E << 5) | 1, (0x1E << 5) | 0x1F]]
	float_tests = struct.pack('<' + 'I' * len(tests), *tests)

	for j in range(16):
		for i in [13]:
			for shift in range(4):
				for out in ['r3_r4_r5_r6', 'r3l_r3h_r4l_r4h']:
					code = b''
					# buffer address
					code += assemble.assemble_line('mov r0, u0')
					code += assemble.assemble_line('mov r1, u1')
					code += assemble.assemble_line('mov_imm r2, 512')
					code += assemble.assemble_line('iadd r0_r1, r0_r1, r2')

					code += assemble.assemble_line('get_sr r2, sr80')

					code += assemble.assemble_line('device_load 1, 0, 0, 4, 0, ' + str(i) + ', '+str(j)+', '+out+', r0_r1, r2, unsigned, lsl ' + str(shift))

					code += assemble.assemble_line('wait 0')

					code += assemble.assemble_line('mov_imm r0, 0')
					code += assemble.assemble_line('mov_imm r1, 0')

					device_memory = applegpu.AddressSpace()

					device_memory.map(TEST_ADDRESS, 1024)
					for x, v in enumerate(extra_data):
						device_memory.set_byte(TEST_ADDRESS + 512 + x, v)

					run_test(code, RANDOM_INITIAL_STATE, device_memory=device_memory, extra_data=extra_data)

	for j in range(16):
		for i in [12]:
			for shift in range(4):
				for out in ['r3_r4_r5_r6']: # TODO: 'r3l_r3h_r4l_r4h' doesn't work yet
					code = b''
					code += assemble.assemble_line('mov r0, u0')
					code += assemble.assemble_line('mov r1, u1')
					code += assemble.assemble_line('mov_imm r2, 512')
					code += assemble.assemble_line('iadd r0_r1, r0_r1, r2')

					code += assemble.assemble_line('get_sr r2, sr80')

					code += assemble.assemble_line('device_load 1, 0, 0, 4, 0, ' + str(i) + ', '+str(j)+', '+out+', r0_r1, r2, unsigned, lsl ' + str(shift))

					code += assemble.assemble_line('wait 0')

					code += assemble.assemble_line('mov_imm r0, 0')
					code += assemble.assemble_line('mov_imm r1, 0')

					device_memory = applegpu.AddressSpace()

					device_memory.map(TEST_ADDRESS, 1024)
					for x, v in enumerate(extra_data):
						device_memory.set_byte(TEST_ADDRESS + 512 + x, v)

					run_test(code, RANDOM_INITIAL_STATE, device_memory=device_memory, extra_data=extra_data)

					device_memory = applegpu.AddressSpace()

					device_memory.map(TEST_ADDRESS, 1024)
					for x, v in enumerate(float_tests):
						device_memory.set_byte(TEST_ADDRESS + 512 + x, v)
					run_test(code, RANDOM_INITIAL_STATE, device_memory=device_memory, extra_data=float_tests)

	for j in range(16):
		for i in [8, 10]: # srgb
			for shift in range(4):
				for out in ['r3_r4_r5_r6', 'r3l_r3h_r4l_r4h']:
					code = b''
					# buffer address
					code += assemble.assemble_line('mov r0, u0')
					code += assemble.assemble_line('mov r1, u1')
					code += assemble.assemble_line('mov_imm r2, 512')
					code += assemble.assemble_line('iadd r0_r1, r0_r1, r2')

					code += assemble.assemble_line('get_sr r2, sr80')

					code += assemble.assemble_line('device_load 1, 0, 0, 4, 0, ' + str(i) + ', '+str(j)+', '+out+', r0_r1, r2, unsigned, lsl ' + str(shift))

					code += assemble.assemble_line('wait 0')

					code += assemble.assemble_line('mov_imm r0, 0')
					code += assemble.assemble_line('mov_imm r1, 0')

					device_memory = applegpu.AddressSpace()

					device_memory.map(TEST_ADDRESS, 1024)
					for x, v in enumerate(extra_data):
						device_memory.set_byte(TEST_ADDRESS + 512 + x, v)

					run_test(code, RANDOM_INITIAL_STATE, device_memory=device_memory, extra_data=extra_data)

	for i in [8]:
		code = b''
		# buffer address
		code += assemble.assemble_line('mov r0, u0')
		code += assemble.assemble_line('mov r1, u1')
		code += assemble.assemble_line('mov_imm r2, 513')
		code += assemble.assemble_line('iadd r0_r1, r0_r1, r2')

		code += assemble.assemble_line('get_sr r2, sr80')

		code += assemble.assemble_line('device_load 1, 0, 0, 4, 0, ' + str(i) + ', 15, r3_r4_r5_r6, r0_r1, r2, unsigned, lsl 1')
		code += assemble.assemble_line('wait 0')

		code += assemble.assemble_line('mov_imm r0, 0')
		code += assemble.assemble_line('mov_imm r1, 0')
		device_memory = applegpu.AddressSpace()

		device_memory.map(TEST_ADDRESS, 1024)
		for i, v in enumerate(extra_data):
			device_memory.set_byte(TEST_ADDRESS + 512 + i, v)

		run_test(code, RANDOM_INITIAL_STATE, device_memory=device_memory, extra_data=extra_data)

	# test for how mask works
	for i in range(16):
		code = b''
		# buffer address
		code += assemble.assemble_line('mov r0, u0')
		code += assemble.assemble_line('mov r1, u1')
		code += assemble.assemble_line('mov_imm r2, 512')
		code += assemble.assemble_line('iadd r0_r1, r0_r1, r2')

		code += assemble.assemble_line('get_sr r4, sr80')

		code += assemble.assemble_line('device_load 1, 0, 0, 4, 0, i8, '+str(i)+', r3_r4, r0_r1, r4, unsigned, lsl 0')

		code += assemble.assemble_line('wait 0')


		code += assemble.assemble_line('mov_imm r0, 0')
		code += assemble.assemble_line('mov_imm r1, 0')
		device_memory = applegpu.AddressSpace()

		device_memory.map(TEST_ADDRESS, 1024)
		for i, v in enumerate(extra_data):
			device_memory.set_byte(TEST_ADDRESS + 512 + i, v)

		run_test(code, RANDOM_INITIAL_STATE, device_memory=device_memory, extra_data=extra_data)

	# basic types
	for i in range(8):
		code = b''
		# buffer address
		code += assemble.assemble_line('mov r1, u0')
		code += assemble.assemble_line('mov r2, u1')
		code += assemble.assemble_line('mov_imm r3, ' + str(512)) # + 7 + 115))
		code += assemble.assemble_line('iadd r1_r2, r1_r2, r3')

		code += assemble.assemble_line('get_sr r3, sr80')

		code += assemble.assemble_line('device_load 1, 0, 0, 4, 0, ' + str(i) + ', 3, r5_r6, r1_r2, r3, unsigned, lsl 0')
		if i != 2:
			code += assemble.assemble_line('device_load 1, 0, 0, 4, 0, ' + str(i) + ', 3, r0l_r0h, r1_r2, r3, unsigned, lsl 0')

		code += assemble.assemble_line('wait 0')


		code += assemble.assemble_line('mov_imm r1, 0')
		code += assemble.assemble_line('mov_imm r2, 0')
		device_memory = applegpu.AddressSpace()

		device_memory.map(TEST_ADDRESS, 1024)
		for i, v in enumerate(extra_data):
			device_memory.set_byte(TEST_ADDRESS + 512 + i, v)

		run_test(code, RANDOM_INITIAL_STATE, device_memory=device_memory, extra_data=extra_data)


def main():
	# TODO: fix memory tests
	#print('test_memory()')
	#test_memory()

	print('test_uniforms()')
	test_uniforms()

	print('test_sr32()')
	test_sr32()

	print('test_bitop()')
	test_bitop()
	
	print('test_add()')
	test_add()
	
	print('test_madd()')
	test_madd()
	
	# print('test_fmadd()')
	# test_fmadd()
	
	# print('test_fadd()')
	# test_fadd()
	
	# print('test_fmul()')
	# test_fmul()
	
	# print('test_fmadd16()')
	# test_fmadd16()
	
	# print('test_shift()')
	# test_shift()
	
	# print('test_exec_ops()')
	# test_exec_ops()
	
	# print('test_simd_shuffle_down()')
	# test_simd_shuffle_down()
	
	# print('test_simd_shuffle()')
	# test_simd_shuffle()
	
	# print('test_icmp_ballot()')
	# test_icmp_ballot()
	
	# print('test_fcmpsel()')
	# test_fcmpsel()
	
	# print('test_popcount()')
	# test_popcount()
	
	# print('test_bitrev()')
	# test_bitrev()
	
	# print('test_ffs()')
	# test_ffs()


main()

tmpdir.cleanup()
