#ifndef PROTOCOL_H
#define PROTOCOL_H

enum HWTestBedRequest {
	/// `BEGIN_COMPUTE` begins a new compute request.  Clears all previously set items
	HW_TEST_BED_REQUEST_BEGIN_COMPUTE           = 1,
	/// `SET_COMPUTE_SHADER_DATA, U32 Size, U8... Data` sets the given compute shader data
	HW_TEST_BED_REQUEST_SET_COMPUTE_SHADER_DATA = 2,
	/// `SET_COMPUTE_SHADER_FILE, U32 Size, U8... Data` loads a compute shader from the given file and sets it as data
	HW_TEST_BED_REQUEST_SET_COMPUTE_SHADER_FILE = 3,
	/// `BUFFER_DATA, U32 Index, U32 Size, U8... Data`
	HW_TEST_BED_REQUEST_SET_BUFFER_DATA         = 4,
	/// `BUFFER_RESULT, U32 Index, U32 Size`
	/// If a `BUFFER_DATA` request has come in for the same index, size will be ignored and the existing buffer will be used for the request
	HW_TEST_BED_REQUEST_SET_BUFFER_RESULT       = 5,
	/// `EXECUTE_COMPUTE, U32 ThreadgroupsPerGrid[3], U32 ThreadsPerThreadgroup[3]`, clears all previously set items
	HW_TEST_BED_REQUEST_EXECUTE_COMPUTE         = 6,
	// `SET_COMPUTE_TGSM, U32 Size` sets the amount of threadgroup shared memory per threadgroup
	HW_TEST_BED_REQUEST_SET_COMPUTE_TGSM        = 7,
	/// `EXECUTE_COMPUTE_BENCH, U32 Count, (U32 ThreadgroupsPerGrid[3], U32 ThreadsPerThreadgroup[3])...`
	/// Like `EXECUTE_COMPUTE` but ensures the GPU is at its maximum clock speed (which may require a time-consuming warm-up run), and supports submitting multiple runs to do in succession (no overlap).
	/// If successful, `Count` `TIME` responses will be sent (one per run in order), along with one set of `RESPONSE_BUFFER_DATA` per `BUFFER_RESULT` request
	/// Note that buffers will *not* be reset between runs, and result buffers will contain the contents of all runs having run one after another on the same buffer
	HW_TEST_BED_REQUEST_EXECUTE_COMPUTE_BENCH   = 8,
	/// `GET_GPU_NAME` Gets the name of the GPU being used
	HW_TEST_BED_REQUEST_GET_GPU_NAME            = 9,
};

enum HWTestBedResponse {
	/// `BEGIN` (Begins a multi-command response)
	HW_TEST_BED_RESPONSE_BEGIN = 1,
	/// `END` Indicates the end of a response list started with `BEGIN`
	HW_TEST_BED_RESPONSE_END   = 2,
	/// `ERROR, U32 Size, U8... Data` (string) error message, indicating the request failed
	HW_TEST_BED_RESPONSE_ERROR = 3,
	/// `TIME, U64 Time` Indicates the amount of GPU time the request took to run
	HW_TEST_BED_RESPONSE_TIME  = 4,
	/// `BUFFER_DATA, U32 Index, U32 Size, U8... Data` Data from a buffer requested in the request
	HW_TEST_BED_RESPONSE_BUFFER_DATA = 5,
	/// `STRING, U32 Size, U8... Data` string response to one of the following requests: `GET_GPU_NAME`
	/// Single command, will not be wrapped in a `BEGIN`/`END`
	HW_TEST_BED_RESPONSE_STRING = 6,
};

#endif
