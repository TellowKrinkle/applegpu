import os
import sys
import applegpu

VERBOSE = False
STOP_ON_STOP = True

def disassemble(code, code_offset = 0):
	p = 0
	end = False
	skipping = False
	while p < len(code) and not end:
		n = applegpu.opcode_to_number(code[p:])
		if not skipping and (n & 0xFFFFffff) == 0:
			print()
			skipping = True
		if skipping:
			if (n & 0xFFFF) == 0:
				p += 2
				continue
			else:
				skipping = False
		length = 2
		for o in applegpu.instruction_descriptors:
			if o.matches(n):
				mnem = o.decode_mnem(n)
				length = o.decode_size(n)
				asm = str(o.disassemble(n, pc = p + code_offset))
				if VERBOSE:
					asm = asm.ljust(60) + '\t'
					fields = '[' + ', '.join('%s=%r' % i for i in o.decode_fields(n)) + ']'
					rem = o.decode_remainder(n)
					if rem:
						fields = fields.ljust(85) + ' ' + str(rem)
					asm += fields
				print('%4x:' % (p + code_offset), code[p:p+length].hex().ljust(20), asm)
				if mnem == 'stop':
					if STOP_ON_STOP:
						end = True
				break
		else:
			print('%4x:' % (p + code_offset), code[p:p+2].hex().ljust(20), '<disassembly failed>')

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
