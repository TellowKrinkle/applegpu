import argparse
import assemble
import os
import struct
import tempfile
import hwtestbed

def gen_runner(code, setup=b'', unroll=1, loops=1000000):
	setup += bytes.fromhex('62000000') # mov_imm r0l, 0
	setup += b'\x62\x7d' + struct.pack('<I', loops) # mov_imm r31, loops
	code = b'\x68\x00' + code * unroll
	code += bytes.fromhex(
		'8e7dfe1a00000000' # isub       r31.cache, r31.discard, 1
		'52957e020000'     # while_icmp r0l, nseq, r31, 0, 2
	)
	jmp_len = len(code)
	code += b'\x00\xc0' + struct.pack('<i', -jmp_len) # jmp_exec_any begin
	code += bytes.fromhex(
		'521600000000' # pop_exec r0l, 2
		'8800'         # stop
	)
	return setup + code

def gpu_mhz(name):
	if 'M1 Pro' in name or 'M1 Max' in name or 'M1 Ultra' in name:
		return 1296
	if 'M1' in name:
		return 1278
	# TODO
	return 0

parser = argparse.ArgumentParser(description='Benchmark AGX shader asm')
parser.add_argument('instructions', nargs='*', help='Shader instructions')
parser.add_argument('-i', '--input', help='Path to input file')
parser.add_argument('-t', '--tempdir', help='Directory to place temporary files')
parser.add_argument('-b', '--binary', action='store_true', help='Treat code as binary instead of assembly')
parser.add_argument('--tgsm', type=int, default=0x100, help='Set the amount of threadgroup shared memory')
parser.add_argument('--setup', nargs='*', help='Initialization shader instructions')
parser.add_argument('--setup-file', help='File containing initialization shader instructions')
parser.add_argument('--threadgroups', type=int, default=1, help='The number of threadgroups to run')
parser.add_argument('--loops', type=int, default=100000, help='The number of times to loop the shader when benchmarking')
parser.add_argument('--unroll', type=int, default=0, help='The number of times to duplicate the shader code when benchmarking')
args = parser.parse_args()
if (args.input and args.instructions) or (args.setup and args.setup_file):
	parser.error('Please supply instructions with an input file or on the command line but not both')
elif not args.input and not args.instructions:
	parser.error('An input file or instructions are required')
if args.input:
	code = assemble.assemble_file(args.input, args.binary)
else:
	if args.binary:
		code = bytes.fromhex(' '.join(args.instructions))
	else:
		code = assemble.assemble_multiline(' '.join(args.instructions).split(';'))
if args.setup:
	if args.binary:
		setup = bytes.fromhex(' '.join(args.setup))
	else:
		setup = assemble.assemble_multiline(' '.join(args.setup).split(';'))
elif args.setup_file:
	setup = assemble.assemble_file(args.setup_file, args.binary)
else:
	setup = b''

with tempfile.TemporaryDirectory(dir=args.tempdir) as tempdir:
	testbed = hwtestbed.HWTestBed(os.path.join(tempdir, 'compute.metallib'))
	space = testbed.get_code_space() - len(gen_runner(b'')) - len(setup)
	hz = gpu_mhz(testbed.get_name()) * 1000000
	unit = ' cycles' if hz else 'ns'
	units_per_second = hz if hz else 1000000000
	def run_test(unroll, loops, runs=4, tg_size=[(1, 1, 1)], num_tg=[(1, 1, 1)]):
		req = hwtestbed.HWTestBedBenchmarkRequest(
			shader=gen_runner(code, setup=setup, unroll=unroll, loops=loops),
			tgsm=args.tgsm,
			tg_size=tg_size,
			num_tg=num_tg,
			run_count=runs
		)
		return testbed.run(req)
	loops = args.loops
	if args.unroll:
		base_unroll = args.unroll
	else:
		unroll1 = min(run_test(1, loops).times)
		unroll2 = min(run_test(2, loops).times)
		if unroll2 > unroll1 * 1.5 or len(code) * 6 > space:
			base_unroll = 1
		elif unroll2 > unroll1 * 1.25 or len(code) * 12 > space:
			base_unroll = 2
		elif unroll2 > unroll1 * 1.125 or len(code) * 24 > space:
			base_unroll = 4
		else:
			base_unroll = 8
	sizes = [1, 16, 32, 64, 128, 256, 512, 1024]
	base  = run_test(base_unroll * 2,     loops, runs=8, tg_size=[(x, 1, 1) for x in sizes], num_tg=[(args.threadgroups, 1, 1)] * len(sizes))
	extra = run_test(base_unroll * 3, loops, runs=8, tg_size=[(x, 1, 1) for x in sizes], num_tg=[(args.threadgroups, 1, 1)] * len(sizes))
	tg_string = f"x{args.threadgroups}" if args.threadgroups > 1 else ""
	def get_stats(i):
		base_time = min(base.group(i)) * units_per_second / loops
		extra_time = min(extra.group(i)) * units_per_second / loops
		per_run = (extra_time - base_time) / base_unroll
		return (base_time, extra_time, per_run)
	pad = [0, 0, 0]
	for i in range(len(sizes)):
		stats = get_stats(i)
		for j in range(len(stats)):
			pad[j] = max(pad[j], len(f'{stats[j]:.2f}'))
	for i in range(len(sizes)):
		(base_time, extra_time, per_run) = get_stats(i)
		print(f'{sizes[i]:4d}{tg_string} threads: {per_run:{pad[2]}.2f}{unit} (({extra_time:{pad[1]}.2f} - {base_time:{pad[0]}.2f}) / {base_unroll}) per run')
