import os
import sys
import applegpu

VERBOSE = False
STOP_ON_STOP = True

instruction_lengths = {
	0x0f04:  4, # else
	0x0f05:  4, # if
	0x0704:  6, # barrier
	0x5701:  6, # jmp_exec_none
	0x0f06:  6, # pop_exec
	0x1704:  8, # unpack unorm/snorm
	0x2705:  8, # popcount
	0x2706:  8, # unpack rgb10a2 rg11b10f rgb9e5
	0xa707:  8, # convert
	0xa704:  8, # bitrev
	0xa705:  8, # ffs
	0x5706:  8, # store vertex output
	0x3700:  8, # quad_and
	0x3701:  8, # quad_xor
	0x3702:  8, # quad_smin
	0x3703:  8, # quad_umin
	0x3706:  8, # quad_fadd
	0xb700:  8, # quad_or
	0xb701:  8, # quad_iadd
	0xb702:  8, # quad_smax
	0xb703:  8, # quad_umax
	0xb706:  8, # quad_fmul
	0xb705:  8, # quad_fmin
	0xb707:  8, # quad_fmax
	0x3f00:  8, # simd_and
	0x3f01:  8, # simd_xor
	0x3f02:  8, # simd_smin
	0x3f03:  8, # simd_umin
	0x3f06:  8, # simd_fadd
	0xbf00:  8, # simd_or
	0xbf01:  8, # simd_iadd
	0xbf02:  8, # simd_smax
	0xbf03:  8, # simd_umax
	0xbf06:  8, # simd_fmul
	0xbf05:  8, # simd_fmin
	0xbf07:  8, # simd_fmax
	0x0f01: 10, # jmp_exec_none?
	0x2707: 10, # unknown, appears in round implementation
	0x2f00: 10, # floor/ceil/trunc/rint
	0x2f02: 10, # log2
	0x2f03: 10, # sin_pt_1???
	0x9f01: 10, # iadd
	0x9704: 10, # pack unorm/snorm
	0x1f01: 10, # isub
	0xa701: 10, # asr
	0xa703: 10, # asrh
	0xaf00: 10, # rcp
	0xaf02: 10, # exp2
	0x4700: 10, # quad_shuffle
	0x4701: 10, # quad_shuffle_up
	0xc701: 10, # quad_shuffle_down
	0xc700: 10, # quad_shuffle_xor
	0x4704: 10, # simd_shuffle
	0x4705: 10, # simd_shuffle_up
	0xc705: 10, # simd_shuffle_down
	0xc704: 10, # simd_shuffle_xor
	0x1702: 10, # quad_ballot
	0x1707: 10, # simd_ballot
	0x2700: 12, # bfi
	0x2701: 12, # extr
	0x2702: 12, # ???
	0x9f00: 12, # imadd
	0x1f00: 12, # imsub
	0xa700: 12, # bfeil
	0x9703: 12, # quad_ballot?
	0x9707: 12, # simd_ballot?
	0x6f00: 12, # ???
	0x1705: 12, # ???
	0x8f04: 14, # while + jmp_exec_any?
	0x6700: 14, # device_load
	0x6702: 14, # threadgroup_load
	0xa706: 14, # pack rgb10a2 rg11b10f rgb9e5
	0xe700: 14, # device_store
	0xe702: 14, # theadgroup_store
	0xd700: 16, # image_store
}

def instruction_length(code, offset):
	op = code[offset] & 7
	size = 0
	if op == 0 or op == 1: # fadd/fmul/fmadd
		if (code[offset + 2] & 4) == 0:
			size = 4
		# TODO: How does it actually figure this out?
		elif (code[offset + 2] & 3) == 1 and (code[offset + 4] & 2) == 2:
			size = 12 # fmul is special???
		else:
			size = 6 + 2 * (code[offset + 4] & 3)
	elif op == 2: # cmpsel
		if (code[offset + 2] & 1) == 0:
			size = 6
		elif (code[offset + 4] & 2) == 0:
			size = 8
		elif (code[offset + 4] & 1) == 0:
			size = 10
		else:
			size = 14
	elif op == 3: # bitop
		if (code[offset + 2] & 6) != 6:
			size = 4
		else:
			size = 10
	elif op == 4: # mov_imm, get_sr
		if (code[offset + 1] & 0x80) == 0:
			size = 2 # mov_imm 7bit
		elif (code[offset + 2] & 0x3) == 0:
			size = 4
		elif (code[offset + 2] & 0x3) < 3:
			size = 8
		else:
			size = 10
	elif op == 5: # texture load / sample
		size = 14
	elif op == 6:
		if code[offset + 1] == 0:
			size = 4 # stop
		else:
			size = 8 # wait
	elif op == 7:
		op = (code[offset] << 8) | (code[offset + 1] & 0xf)

	if op in instruction_lengths:
		size = instruction_lengths[op]
	return size

def hex(code):
	return " ".join(f"{b:02X}" for b in code)

def disassemble(code, code_offset = 0):
	p = 0
	end = False
	skipping = False
	while p < len(code) and not end:
		length = instruction_length(code, p)
		if not length:
			print("Unrecognized opcode")
			while p < len(code):
				print(hex(code[p:p+16]))
				p += 16
			break
		n = applegpu.opcode_to_number(code[p:p+length])
		if not skipping and (n & 0xFFFFffff) == 0:
			print()
			skipping = True
		if skipping:
			if (n & 0xFFFF) == 0:
				p += 2
				continue
			else:
				skipping = False
		for o in applegpu.instruction_descriptors:
			if o.matches(n):
				mnem = o.decode_mnem(n)
				if o.decode_size(n) != length:
					print(f"Length mismatch (expected {length}, got {o.decode_size(n)})")
				asm = str(o.disassemble(n, pc = p + code_offset))
				if VERBOSE:
					asm = asm.ljust(60) + '\t'
					fields = '[' + ', '.join('%s=%r' % i for i in o.decode_fields(n)) + ']'
					rem = o.decode_remainder(n)
					if rem:
						fields = fields.ljust(85) + ' ' + str(rem)
					asm += fields
				print('%4x:' % (p + code_offset), hex(code[p:p+length]).ljust(42), asm)
				if mnem == 'stop':
					if STOP_ON_STOP:
						end = True
				break
		else:
			print('%4x:' % (p + code_offset), hex(code[p:p+length]).ljust(42), '<disassembly failed>')

		assert length >= 2 and length % 2 == 0
		p += length

if __name__ == '__main__':
	if len(sys.argv) > 1:
		f = open(sys.argv[1], 'rb')
		if len(sys.argv) > 2:
			f.seek(int(sys.argv[2], 0))
		code = f.read()
		disassemble(code)
	else:
		print('usage: python3 disassemble.py [filename] [offset]')
		exit(1)
