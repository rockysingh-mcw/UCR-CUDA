#include <iostream>
#include <fstream>
#include <cstdlib> // atoi, EXIT_FAILURE
#include <cuda_runtime.h>

#include "include/cuda_def.cuh"
#include "include/cdtw.cuh"

#define EUCLIDEAN (true)
#define MANHATTAN (false)

int main(int argc, char *argv[])
{
    if (argc != 6)
    {
        std::cout << "call " << argv[0]
                  << " query.bin subject.bin M N P" << std::endl;
        return 1;
    }

    // select device and reset (if intended)
    cudaSetDevice(0);
    CUERR
    cudaDeviceReset();
    CUERR

    float *zlower = NULL, *zupper = NULL, *zquery = NULL, *subject = NULL;
    float *Subject = NULL, *AvgS = NULL, *StdS = NULL;

    int M = atoi(argv[3]);
    int N = atoi(argv[4]);
    int W = M * (atoi(argv[5]) * 0.01);

    std::cout << "\n= info =====================================" << std::endl;
    std::cout << "|Query| = " << M << "\t"
              << "|Subject| = " << N << "\t"
              << "window = " << W << std::endl;

    // host side pinned memory
    cudaMallocHost(&zlower, sizeof(float) * M);
    CUERR
    cudaMallocHost(&zupper, sizeof(float) * M);
    CUERR
    cudaMallocHost(&zquery, sizeof(float) * M);
    CUERR
    cudaMallocHost(&subject, sizeof(float) * N);
    CUERR

    // device side memory
    cudaMalloc(&Subject, sizeof(float) * N);
    CUERR
    cudaMalloc(&AvgS, sizeof(float) * (N - M + 1));
    CUERR
    cudaMalloc(&StdS, sizeof(float) * (N - M + 1));
    CUERR

    // timer events
    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    std::cout << "\n= loading data =============================" << std::endl;

    cudaEventRecord(start, 0);

    // read query from file
    std::ifstream qfile(argv[1], std::ios::binary | std::ios::in);
    if (!qfile)
    {
        std::cerr << "Failed to open query file: " << argv[1] << std::endl;
        return EXIT_FAILURE;
    }
    qfile.read(reinterpret_cast<char *>(zquery), sizeof(float) * M);

    // read subject from file
    std::ifstream sfile(argv[2], std::ios::binary | std::ios::in);
    if (!sfile)
    {
        std::cerr << "Failed to open subject file: " << argv[2] << std::endl;
        return EXIT_FAILURE;
    }
    sfile.read(reinterpret_cast<char *>(subject), sizeof(float) * N);

    // z-normalize query and envelope (host functions, assumed implemented)
    znormalize(zquery, M);
    envelope(zquery, W, zlower, zupper, M);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    std::cout << "Milliseconds to load data: " << time << std::endl;

    cudaEventRecord(start, 0);

    // copy subject to gpu
    cudaMemcpy(Subject, subject, sizeof(float) * N, cudaMemcpyHostToDevice);
    CUERR

    // copy query and associated envelopes to constant memory (device constants)
    cudaMemcpyToSymbol(::Czlower, zlower, sizeof(float) * M);
    CUERR
    cudaMemcpyToSymbol(::Czupper, zupper, sizeof(float) * M);
    CUERR
    cudaMemcpyToSymbol(::Czquery, zquery, sizeof(float) * M);
    CUERR

    // --- create texture object for Subject on device (modern API) ---
    cudaTextureObject_t texObj = 0;
    {
        cudaResourceDesc resDesc = {};
        resDesc.resType = cudaResourceTypeLinear;
        resDesc.res.linear.devPtr = Subject;
        resDesc.res.linear.sizeInBytes = static_cast<size_t>(N) * sizeof(float);
        resDesc.res.linear.desc = cudaCreateChannelDesc<float>();

        cudaTextureDesc texDesc = {};
        texDesc.readMode = cudaReadModeElementType;

        cudaError_t cerr = cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);
        if (cerr != cudaSuccess)
        {
            std::cerr << "cudaCreateTextureObject failed: " << cudaGetErrorString(cerr) << std::endl;
            // proceed without texture object or exit depending on your design
            texObj = 0;
        }
    }

    // calculate windowed average and standard deviation of Subject
    avg_std<double>(Subject, AvgS, StdS, M, N);
    std::cout << "\n= pruning ==================================" << std::endl;

    // If prune_cdtw is a host function that launches kernels, and if it needs the texture
    // object, you must update its signature to accept texObj. Here we call as-is.
    prune_cdtw<EUCLIDEAN>(Subject, AvgS, StdS, M, N, W);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    // destroy texture object if created
    if (texObj)
        cudaDestroyTextureObject(texObj);

    std::cout << "Milliseconds to find best match: " << time << std::endl;

    // cleanup
    cudaFree(Subject);
    cudaFree(AvgS);
    cudaFree(StdS);

    cudaFreeHost(zlower);
    cudaFreeHost(zupper);
    cudaFreeHost(zquery);
    cudaFreeHost(subject);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
