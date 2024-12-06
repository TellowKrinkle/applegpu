import os
import sys
import applegpu

VERBOSE = False
STOP_ON_STOP = True

instruction_lengths = {
	0x40f:  4, # else
	0x50f:  4, # if
	0x407:  6, # barrier
	0x157:  6, # jmp_exec_none
	0x60f:  6, # pop_exec
	0x417:  8, # unpack unorm/snorm
	0x527:  8, # popcount
	0x627:  8, # unpack rgb10a2 rg11b10f rgb9e5
	0x7a7:  8, # convert
	0x4a7:  8, # bitrev
	0x5a7:  8, # ffs
	0x657:  8, # store vertex output
	0x037:  8, # quad_and
	0x137:  8, # quad_xor
	0x237:  8, # quad_smin
	0x337:  8, # quad_umin
	0x637:  8, # quad_fadd
	0x0b7:  8, # quad_or
	0x1b7:  8, # quad_iadd
	0x2b7:  8, # quad_smax
	0x3b7:  8, # quad_umax
	0x6b7:  8, # quad_fmul
	0x5b7:  8, # quad_fmin
	0x7b7:  8, # quad_fmax
	0x03f:  8, # simd_and
	0x13f:  8, # simd_xor
	0x23f:  8, # simd_smin
	0x33f:  8, # simd_umin
	0x63f:  8, # simd_fadd
	0x0bf:  8, # simd_or
	0x1bf:  8, # simd_iadd
	0x2bf:  8, # simd_smax
	0x3bf:  8, # simd_umax
	0x6bf:  8, # simd_fmul
	0x5bf:  8, # simd_fmin
	0x7bf:  8, # simd_fmax
	0x10f: 10, # jmp_exec_none?
	0x727: 10, # unknown, appears in round implementation
	0x02f: 10, # floor/ceil/trunc/rint
	0x22f: 10, # log2
	0x32f: 10, # sin_pt_1???
	0x19f: 10, # iadd
	0x497: 10, # pack unorm/snorm
	0x11f: 10, # isub
	0x327: 10, # ???
	0x427: 10, # ???
	0x1a7: 10, # asr
	0x3a7: 10, # asrh
	0x0af: 10, # rcp
	0x2af: 10, # exp2
	0x047: 10, # quad_shuffle
	0x147: 10, # quad_shuffle_up
	0x1c7: 10, # quad_shuffle_down
	0x0c7: 10, # quad_shuffle_xor
	0x447: 10, # simd_shuffle
	0x547: 10, # simd_shuffle_up
	0x5c7: 10, # simd_shuffle_down
	0x4c7: 10, # simd_shuffle_xor
	0x217: 10, # quad_ballot
	0x717: 10, # simd_ballot
	0x027: 12, # bfi
	0x127: 12, # extr
	0x227: 12, # shlhi
	0x09f: 12, # imadd
	0x01f: 12, # imsub
	0x0a7: 12, # bfeil
	0x2a7: 12, # shrhi
	0x397: 12, # quad_ballot?
	0x797: 12, # simd_ballot?
	0x06f: 12, # ???
	0x517: 12, # ???
	0x48f: 14, # while + jmp_exec_any?
	0x067: 14, # device_load
	0x267: 14, # threadgroup_load
	0x6a7: 14, # pack rgb10a2 rg11b10f rgb9e5
	0x0e7: 14, # device_store
	0x2e7: 14, # theadgroup_store
	0x0d7: 16, # image_store
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
		op = code[offset] | ((code[offset + 1] & 0xf) << 8)

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
