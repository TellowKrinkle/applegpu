#include "runner.h"

#include <array>
#include <cstring>
#include <Metal/Metal.h>

#if !__has_feature(objc_arc)
	#error Please compile with -fobjc-arc
#endif

static MTLSize MTLSizeMakeFromArray(const uint32_t(&array)[3]) {
	return MTLSizeMake(array[0], array[1], array[2]);
}

static void reportError(std::string* err, const char* action, NSError* nserr = nullptr) {
	if (!err)
		return;
	if (nserr)
		*err = [[NSString stringWithFormat:@"%s: %@", action, [nserr localizedDescription]] UTF8String];
	else
		*err = action;
}

// Enough time to spin to get the GPU into high power state
static constexpr double TARGET_SPIN_TIME = 1.0 / 30;

class MacRunner : public Runner {
	dispatch_data_t metallib;
	id<MTLDevice> dev;
	id<MTLComputePipelineState> spin_pipeline;
	id<MTLCommandQueue> queue;
	id<MTLFence> fence;
	id<MTLBuffer> spin_buffer = nullptr;
	uint32_t spin_count = 2000000; // ~> about 120ms on M1 GPU, 60ms on M3 GPU
public:
	MacRunner(id<MTLDevice> dev, dispatch_data_t metallib, id<MTLComputePipelineState> spin_pipeline)
		: metallib(metallib), dev(dev), spin_pipeline(spin_pipeline)
	{
		queue = [dev newCommandQueue];
		fence = [dev newFence];
	}
	void make_spin_buffer(id<MTLCommandBuffer> cb) {
		if (spin_buffer)
			return;
		spin_buffer = [dev newBufferWithLength:4 options:MTLResourceStorageModePrivate];
		id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
		[blit fillBuffer:spin_buffer range:NSMakeRange(0, 4) value:0];
		[blit endEncoding];
	}
	Buffer create_buffer(size_t size, std::string* err) override {
		@autoreleasepool {
			id<MTLBuffer> buf = [dev newBufferWithLength:size options:MTLResourceStorageModeShared];
			Buffer res;
			res.cpu_pointer = [buf contents];
			res.gpu_handle = (__bridge_retained void*)buf;
			res.size = size;
			return res;
		}
	}
	void destroy_buffer(const Buffer &buffer) override {
		(void)(__bridge_transfer id)buffer.gpu_handle;
	}
	Shader* create_compute_shader_from_file(const char* filename, std::string* error) override {
		return reinterpret_cast<Shader*>(strdup(filename));
	}
	Shader* create_compute_shader(void* data, size_t size, std::string* err) override {
		if (err)
			*err = "Compute shader from data is unsupported on macOS";
		return nullptr;
	}
	void destroy_shader(Shader *shader) override {
		free(shader);
	}

	id<MTLComputePipelineState> late_create_compute_shader(const char* path, const char** description, NSError** err) {
		// All our shaders look like the same shader from Metal's perspective, since we're hackily modiying binary archives to make them.
		// MTLLibraries cache their pipelines, so we need a new MTLLibrary for every run.
		MTLBinaryArchiveDescriptor* adesc = [MTLBinaryArchiveDescriptor new];
		MTLComputePipelineDescriptor* pdesc;
		id<MTLLibrary> lib = [dev newLibraryWithData:metallib error:err];
		if (!lib) { *description = "Failed to create metallib"; return nullptr; }
		id<MTLFunction> func = [lib newFunctionWithName:@"add_arrays"];
		if (!func) { *description = "Function missing from metallib"; return nullptr; }
		[adesc setUrl:[NSURL fileURLWithPath:[NSString stringWithCString:path encoding:NSUTF8StringEncoding]]];
		id<MTLBinaryArchive> archive = [dev newBinaryArchiveWithDescriptor:adesc error:err];
		if (!archive) { *description = "Failed to create binary archive"; return nullptr; }
		pdesc = [MTLComputePipelineDescriptor new];
		[pdesc setBinaryArchives:@[archive]];
		[pdesc setComputeFunction:func];
		id<MTLComputePipelineState> pipeline;
		pipeline = [dev newComputePipelineStateWithDescriptor:pdesc
		                                              options:MTLPipelineOptionFailOnBinaryArchiveMiss
		                                           reflection:nil
		                                                error:err];
		if (!pipeline) { *description = "Failed to create pipeline"; return nullptr; }
		return pipeline;
	}

	static bool handle_completed_cb(id<MTLCommandBuffer> cb, ComputeOutput& output, std::string* error)
	{
		[cb waitUntilCompleted];
		if ([cb status] == MTLCommandBufferStatusError) {
			reportError(error, "Command buffer failed", [cb error]);
			return false;
		}
		output.nanoseconds_elapsed = static_cast<uint64_t>(([cb GPUEndTime] - [cb GPUStartTime]) * 1000000000ull);
		return true;
	}

	bool run_compute_shader(ComputeRun& run, std::string* error) override {
		@autoreleasepool {
			NSError* nserr = nullptr;
			const char* action = nullptr;
			if (!run.shader) {
				reportError(error, "No shader specified");
				return false;
			}
			id<MTLComputePipelineState> pipe = late_create_compute_shader(reinterpret_cast<const char*>(run.shader), &action, &nserr);
			if (!pipe) {
				reportError(error, action, nserr);
				return false;
			}
			id<MTLCommandBuffer> spin_cb = nullptr;
			if (run.force_high_clocks) {
				spin_cb = [queue commandBuffer];
				make_spin_buffer(spin_cb);
				id<MTLComputeCommandEncoder> enc = [spin_cb computeCommandEncoder];
				[enc setBuffer:spin_buffer offset:0 atIndex:0];
				[enc setBytes:&spin_count length:sizeof(spin_count) atIndex:1];
				[enc setComputePipelineState:spin_pipeline];
				[enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
				[enc updateFence:fence];
				[enc endEncoding];
				[spin_cb commit];
			}
			std::array<id<MTLCommandBuffer>, 4> prev_cb = {};
			size_t current_invocation = 0;
			for (current_invocation = 0; current_invocation < run.num_invocations; current_invocation++) {
				const ComputeInput& input = run.inputs[current_invocation];
				id<MTLCommandBuffer> cb = [queue commandBuffer];
				id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
				[enc setComputePipelineState:pipe];
				if (run.threadgroup_memory_size)
					[enc setThreadgroupMemoryLength:run.threadgroup_memory_size atIndex:0];
				for (size_t i = 0; i < run.num_buffers; i++) {
					if (run.buffers[i].gpu_handle) {
						[enc setBuffer:(__bridge id<MTLBuffer>)run.buffers[i].gpu_handle offset:0 atIndex:i];
					}
				}
				[enc dispatchThreadgroups:MTLSizeMakeFromArray(input.threadgroups_per_grid)
				    threadsPerThreadgroup:MTLSizeMakeFromArray(input.threads_per_threadgroup)];
				if (current_invocation > 0 || run.force_high_clocks)
					[enc waitForFence:fence];
				if (current_invocation + 1 < run.num_invocations)
					[enc updateFence:fence];
				[enc endEncoding];
				[cb commit];
				if (!cb || !enc) {
					reportError(error, "Failed to create command buffer and encoder", nullptr);
					break;
				}
				uint32_t idx = current_invocation % prev_cb.size();
				if (prev_cb[idx]) {
					size_t prev_idx = current_invocation - prev_cb.size();
					if (!handle_completed_cb(prev_cb[idx], run.outputs[prev_idx], error))
						break;
				}
				prev_cb[idx] = cb;
			}
			if (current_invocation == run.num_invocations) {
				current_invocation -= std::min(prev_cb.size(), current_invocation);
				for (; current_invocation < run.num_invocations; current_invocation++) {
					if (!handle_completed_cb(prev_cb[current_invocation % prev_cb.size()], run.outputs[current_invocation], error))
						break;
				}
			}
			if (current_invocation != run.num_invocations) {
				// Something failed, wait for everything to finish
				for (id<MTLCommandBuffer> cb : prev_cb)
					[cb waitUntilCompleted];
			}
			// Update spin time to speed up future runs (initial estimite is fairly conservative)
			if (spin_cb && [spin_cb status] == MTLCommandBufferStatusCompleted) {
				double time = [spin_cb GPUEndTime] - [spin_cb GPUStartTime];
				double adjustment = TARGET_SPIN_TIME / time;
				double new_count = ceil(spin_count * adjustment);
				new_count = std::max(20000.0, new_count);
				new_count = std::min(10000000.0, new_count);
				spin_count = static_cast<uint32_t>(new_count);
			}
		}
		return true;
	}

	std::string get_device_name() override {
		@autoreleasepool {
			return [[dev name] UTF8String];
		}
	}
};

static const char* spin_shader = R"(
kernel void spin(device uint* data [[buffer(0)]], constant uint& count [[buffer(1)]]) {
	uint value = data[0];
	for (uint i = count; i; i--)
		value = data[value];
	data[0] = value;
}
)";

Runner* Runner::make(std::string* err) {
	@autoreleasepool {
		NSURL* url = [NSURL fileURLWithPath:[[[NSProcessInfo processInfo] arguments] objectAtIndex:0]];
		url = [url URLByDeletingLastPathComponent];
		NSData* nsmetallib = [NSData dataWithContentsOfURL:[url URLByAppendingPathComponent:@"compute.metallib"]];
		if (!nsmetallib) {
			reportError(err, "Failed to get shader metallib");
			return nullptr;
		}
		dispatch_data_t metallib = dispatch_data_create([nsmetallib bytes], [nsmetallib length], nullptr, DISPATCH_DATA_DESTRUCTOR_DEFAULT);

		NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
		if ([devices count] < 1) {
			reportError(err, "No metal devices available");
			return nullptr;
		}

		id<MTLDevice> dev = [devices objectAtIndexedSubscript:0];
		NSError* nserr = nullptr;
		id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithCString:spin_shader encoding:NSUTF8StringEncoding]
		                                       options:nil
		                                         error:&nserr];
		if (nserr) {
			reportError(err, "Failed to compile warm-up shader", nserr);
			return nullptr;
		}

		id<MTLComputePipelineState> spin = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"spin"] error:&nserr];
		if (nserr) {
			reportError(err, "Failed to make pipeline for warm-up shader", nserr);
			return nullptr;
		}

		return new MacRunner(dev, metallib, spin);
	}
}
