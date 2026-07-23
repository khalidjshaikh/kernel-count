package main

/*
#cgo LDFLAGS: -L. -lOpenCL
#define CL_TARGET_OPENCL_VERSION 120
#include "CL/cl.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

static const char* ocl_errstr(cl_int err) {
    switch (err) {
        case CL_SUCCESS: return "CL_SUCCESS";
        case CL_DEVICE_NOT_FOUND: return "CL_DEVICE_NOT_FOUND";
        case CL_DEVICE_NOT_AVAILABLE: return "CL_DEVICE_NOT_AVAILABLE";
        case CL_COMPILER_NOT_AVAILABLE: return "CL_COMPILER_NOT_AVAILABLE";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case CL_OUT_OF_RESOURCES: return "CL_OUT_OF_RESOURCES";
        case CL_OUT_OF_HOST_MEMORY: return "CL_OUT_OF_HOST_MEMORY";
        case CL_BUILD_PROGRAM_FAILURE: return "CL_BUILD_PROGRAM_FAILURE";
        case CL_INVALID_ARG_VALUE: return "CL_INVALID_ARG_VALUE";
        case CL_INVALID_MEM_OBJECT: return "CL_INVALID_MEM_OBJECT";
        case CL_INVALID_KERNEL_ARGS: return "CL_INVALID_KERNEL_ARGS";
        case CL_INVALID_WORK_GROUP_SIZE: return "CL_INVALID_WORK_GROUP_SIZE";
        case CL_INVALID_WORK_ITEM_SIZE: return "CL_INVALID_WORK_ITEM_SIZE";
        default: return "UNKNOWN";
    }
}

static char* get_dev_str(cl_device_id dev, cl_device_info param) {
    size_t sz;
    cl_int err = clGetDeviceInfo(dev, param, 0, NULL, &sz);
    if (err != CL_SUCCESS) return NULL;
    char* buf = (char*)malloc(sz);
    if (!buf) return NULL;
    err = clGetDeviceInfo(dev, param, sz, buf, NULL);
    if (err != CL_SUCCESS) { free(buf); return NULL; }
    return buf;
}

static cl_ulong get_dev_ulong(cl_device_id dev, cl_device_info param) {
    cl_ulong val = 0;
    clGetDeviceInfo(dev, param, sizeof(val), &val, NULL);
    return val;
}
*/
import "C"
import (
	"fmt"
	"os"
	"unsafe"
)

const kernelSource = `
__kernel void count(__global float *output, ulong iterations) {
    float id = (float)get_global_id(0);
    float sum = id;
    for (ulong i = 0; i < iterations; i++) {
        float fi = (float)i;
        sum += fi * 7.0f;
        sum -= fi * 3.0f;
        sum = sum * 0.5f + 1.0f / (fi + 1.0f);
    }
    output[(int)id] = sum;
}
`

func check(err C.cl_int, msg string) {
	if err != C.CL_SUCCESS {
		fmt.Fprintf(os.Stderr, "ERROR: %s: %s (%d)\n", msg, C.GoString(C.ocl_errstr(err)), int(err))
		os.Exit(1)
	}
}

func main() {
	fmt.Println("=== OpenCL GPU Counting Benchmark ===")

	// --- Platform ---
	var platform C.cl_platform_id
	var numPlatforms C.cl_uint
	err := C.clGetPlatformIDs(1, &platform, &numPlatforms)
	check(err, "clGetPlatformIDs")
	fmt.Printf("Platforms: %d\n", int(numPlatforms))

	// --- GPU Device ---
	var device C.cl_device_id
	var numDevices C.cl_uint
	err = C.clGetDeviceIDs(platform, C.CL_DEVICE_TYPE_GPU, 1, &device, &numDevices)
	if err != C.CL_SUCCESS || numDevices == 0 {
		fmt.Println("No GPU, trying CPU...")
		err = C.clGetDeviceIDs(platform, C.CL_DEVICE_TYPE_CPU, 1, &device, &numDevices)
		check(err, "clGetDeviceIDs")
	}

	devName := C.get_dev_str(device, C.CL_DEVICE_NAME)
	vendor := C.get_dev_str(device, C.CL_DEVICE_VENDOR)
	fmt.Printf("Device: %s — %s\n", C.GoString(devName), C.GoString(vendor))
	C.free(unsafe.Pointer(devName))
	C.free(unsafe.Pointer(vendor))
	fmt.Printf("Compute Units: %d\n", int64(C.get_dev_ulong(device, C.CL_DEVICE_MAX_COMPUTE_UNITS)))
	fmt.Printf("Max Freq: %d MHz\n", int64(C.get_dev_ulong(device, C.CL_DEVICE_MAX_CLOCK_FREQUENCY)))
	fmt.Printf("Max WG Size: %d\n", int64(C.get_dev_ulong(device, C.CL_DEVICE_MAX_WORK_GROUP_SIZE)))
	fmt.Println()

	// --- Context & Queue ---
	context := C.clCreateContext(nil, 1, &device, nil, nil, &err)
	check(err, "clCreateContext")
	defer C.clReleaseContext(context)

	queue := C.clCreateCommandQueue(context, device, C.CL_QUEUE_PROFILING_ENABLE, &err)
	check(err, "clCreateCommandQueue")
	defer C.clReleaseCommandQueue(queue)

	// --- Program ---
	src := C.CString(kernelSource)
	srcLen := C.size_t(len(kernelSource))
	program := C.clCreateProgramWithSource(context, 1, &src, &srcLen, &err)
	C.free(unsafe.Pointer(src))
	check(err, "clCreateProgramWithSource")
	defer C.clReleaseProgram(program)

	opts := C.CString("-cl-fast-relaxed-math -cl-mad-enable -Werror")
	err = C.clBuildProgram(program, 1, &device, opts, nil, nil)
	C.free(unsafe.Pointer(opts))
	if err != C.CL_SUCCESS {
		var logSize C.size_t
		C.clGetProgramBuildInfo(program, device, C.CL_PROGRAM_BUILD_LOG, 0, nil, &logSize)
		log := C.malloc(logSize)
		C.clGetProgramBuildInfo(program, device, C.CL_PROGRAM_BUILD_LOG, logSize, log, nil)
		fmt.Fprintf(os.Stderr, "Build log:\n%s\n", C.GoString((*C.char)(log)))
		C.free(log)
		check(err, "clBuildProgram")
	}

	kernel := C.clCreateKernel(program, C.CString("count"), &err)
	check(err, "clCreateKernel")
	defer C.clReleaseKernel(kernel)

	// --- Benchmark ---
	iterations := uint64(1 << 10) // 1024 per work-item
	globalSizes := []int{1 << 12, 1 << 14, 1 << 16, 1 << 18, 1 << 20}
	maxGlobal := 1 << 20
	outBuf := C.clCreateBuffer(context, C.CL_MEM_WRITE_ONLY, C.size_t(maxGlobal*4), nil, &err)
	check(err, "clCreateBuffer")
	defer C.clReleaseMemObject(outBuf)

	iterArg := C.cl_ulong(iterations)
	err = C.clSetKernelArg(kernel, 1, C.size_t(unsafe.Sizeof(iterArg)), unsafe.Pointer(&iterArg))
	check(err, "clSetKernelArg(iter)")

	fmt.Printf("Iterations per work-item: %d\n", iterations)
	fmt.Println()
	fmt.Printf("%-12s %-12s %-14s  %-12s  %s\n", "GlobalSz", "LocalSz", "Time(ms)", "Ops", "Throughput")
	fmt.Println("--------------------------------------------------------------------------------")

	best := uint64(0)
	bestCfg := ""
	var bestTime uint64

	for _, gs := range globalSizes {
		lsList := []int{64, 128, 256, 512, 1024}
		if gs < 64 {
			lsList = []int{gs}
		}
		for _, ls := range lsList {
			if ls > gs || gs%ls != 0 {
				continue
			}
			// Set output arg
			err = C.clSetKernelArg(kernel, 0, C.size_t(unsafe.Sizeof(outBuf)), unsafe.Pointer(&outBuf))
			check(err, "clSetKernelArg(out)")

			gw := C.size_t(gs)
			lw := C.size_t(ls)

			// Warmup
			var wev C.cl_event
			C.clEnqueueNDRangeKernel(queue, kernel, 1, nil, &gw, &lw, 0, nil, &wev)
			C.clFinish(queue)
			C.clReleaseEvent(wev)

			// Timed run
			var ev C.cl_event
			err = C.clEnqueueNDRangeKernel(queue, kernel, 1, nil, &gw, &lw, 0, nil, &ev)
			check(err, "clEnqueueNDRangeKernel")
			C.clFinish(queue)

			var startT, endT C.cl_ulong
			C.clGetEventProfilingInfo(ev, C.CL_PROFILING_COMMAND_START,
				C.size_t(unsafe.Sizeof(startT)), unsafe.Pointer(&startT), nil)
			C.clGetEventProfilingInfo(ev, C.CL_PROFILING_COMMAND_END,
				C.size_t(unsafe.Sizeof(endT)), unsafe.Pointer(&endT), nil)
			durNs := uint64(endT - startT)
			C.clReleaseEvent(ev)

			ops := uint64(gs) * iterations
			throughput := uint64(float64(ops) / (float64(durNs) / 1e9))

			fmt.Printf("%-12d %-12d %-14.3f %-12d %s/s\n",
				gs, ls, float64(durNs)/1e6, ops, formatCount(throughput))

			if throughput > best {
				best = throughput
				bestCfg = fmt.Sprintf("global=%d local=%d", gs, ls)
				bestTime = durNs
			}
		}
	}

	fmt.Println("--------------------------------------------------------------------------------")
	fmt.Printf("Best: %s/s  (%s, %.3f ms)\n", formatCount(best), bestCfg, float64(bestTime)/1e6)

	// --- Sustained ---
	fmt.Println()
	fmt.Println("=== Sustained (10 runs, best config) ===")
	for _, gs := range globalSizes {
		for _, ls := range []int{64, 128, 256, 512, 1024} {
			if ls > gs || gs%ls != 0 {
				continue
			}
			cfg := fmt.Sprintf("global=%d local=%d", gs, ls)
			if cfg != bestCfg {
				continue
			}
			gw := C.size_t(gs)
			lw := C.size_t(ls)
			var totalNs uint64
			const runs = 10
			for r := 0; r < runs; r++ {
				var ev C.cl_event
				C.clEnqueueNDRangeKernel(queue, kernel, 1, nil, &gw, &lw, 0, nil, &ev)
				C.clFinish(queue)
				var st, et C.cl_ulong
				C.clGetEventProfilingInfo(ev, C.CL_PROFILING_COMMAND_START,
					C.size_t(unsafe.Sizeof(st)), unsafe.Pointer(&st), nil)
				C.clGetEventProfilingInfo(ev, C.CL_PROFILING_COMMAND_END,
					C.size_t(unsafe.Sizeof(et)), unsafe.Pointer(&et), nil)
				totalNs += uint64(et - st)
				C.clReleaseEvent(ev)
			}
			avgNs := totalNs / runs
			ops := uint64(gs) * iterations
			avgTp := uint64(float64(ops) / (float64(avgNs) / 1e9))
			fmt.Printf("  Avg: %s/s  (%.3f ms per run)\n", formatCount(avgTp), float64(avgNs)/1e6)
		}
	}

	fmt.Println()
	fmt.Println("=== Done ===")
}

func formatCount(n uint64) string {
	switch {
	case n >= 1_000_000_000_000:
		return fmt.Sprintf("%.2f T", float64(n)/1e12)
	case n >= 1_000_000_000:
		return fmt.Sprintf("%.2f G", float64(n)/1e9)
	case n >= 1_000_000:
		return fmt.Sprintf("%.2f M", float64(n)/1e6)
	case n >= 1_000:
		return fmt.Sprintf("%.2f K", float64(n)/1e3)
	default:
		return fmt.Sprintf("%d", n)
	}
}