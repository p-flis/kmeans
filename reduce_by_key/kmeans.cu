#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <cuda_runtime.h>

#include <helper_cuda.h>
#include <helper_functions.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>


#define K (42*2)
#define N (256*444)
#define TBP (256)

#define MVEC (sizeof(float)*N) //memory for vectors
#define MMEANS (sizeof(float)*K) //memory for means
#define MINVEC (sizeof(int)*N) //memory for vector indexes
#define GMMEANSF (sizeof(float)*K*N) //memory for data matrix (for calculating new centroids)
#define GMMEANSINT (sizeof(int)*K*N) //memory for data matrix (for calculating new centroids)

using namespace thrust::placeholders;

void runTest(int argc, char **argv);

//calculates distance in 3D on CPU
float h_distance(float x1, float x2, float y1, float y2, float z1, float z2)
{
    float x = x1 - x2;
    float y = y1 - y2;
    float z = z1 - z2;
    return x * x + y * y + z * z;
}

//calculates distance in 3D on GPU
__device__ float distance(float x1, float x2, float y1, float y2, float z1, float z2)
{
    float x = x1 - x2;
    float y = y1 - y2;
    float z = z1 - z2;
    return x * x + y * y + z * z;
}

//kernel which finds the closest centroid for every input vector
__global__ void kmeans(float* g_inputX, float* g_inputY, float* g_inputZ, 
                           float* g_meansX, float* g_meansY, float* g_meansZ, 
                           float* g_tmX, float* g_tmY, float* g_tmZ, int* g_tmC)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N)
        return;

    float best = INFINITY;
    int besti = -1;

    float x = g_inputX[idx];
    float y = g_inputY[idx];
    float z = g_inputZ[idx];

    for (int i = 0; i < K; i++)
    {
        float mx = g_meansX[i];
        float my = g_meansY[i];
        float mz = g_meansZ[i];

        float d = distance(x, mx, y, my, z, mz);
        if (d < best)
        {
            best = d;
            besti = i;
        }
    }
   g_tmX[N * besti + idx] = x;
   g_tmY[N * besti + idx] = y;
   g_tmZ[N * besti + idx] = z;
   g_tmC[N * besti + idx] = 1;
}

//kernel which assigns recalculated centroids on GPU
__global__ void assigner(float* g_meansX, float* g_meansY, float* g_meansZ, 
                        float* g_tmX, float* g_tmY, float* g_tmZ, int* g_tmC)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= K)
        return;

    float x = g_tmX[idx];
    float y = g_tmY[idx];
    float z = g_tmZ[idx];
    int c = g_tmC[idx];
    
    if(c == 0)
    {
        return;
    }
    
    g_meansX[idx] = x/c;
    g_meansY[idx] = y/c;
    g_meansZ[idx] = z/c;
}


int main(int argc, char **argv)
{
    runTest(argc, argv);
}

//function that returns random float
float rf()
{
    return rand() / (float) RAND_MAX;
}

//reduce by key function (segmented reduction)
template <class T>
void rbk(thrust::device_vector<T>& pointerIn, thrust::device_vector<T>& pointerOut)
{
    thrust::reduce_by_key(thrust::device, 
                              thrust::make_transform_iterator(thrust::counting_iterator<int>(0), _1/N),
                              thrust::make_transform_iterator(thrust::counting_iterator<int>(N*K), _1/N), 
                              pointerIn.begin(), 
                              thrust::discard_iterator<int>(), 
                              pointerOut.begin());
}

//initialization of centroids
void initCentroids(float* meansX, float* meansY, float* meansZ)
{
    for (int i = 0; i < K; i++)
    {
        meansX[i] = rf();
        meansY[i] = rf();
        meansZ[i] = rf();
    }
}

//writes vectors to a text file
void writeData(float* px, float* py, float* pz, int * pc, const char* filename){
    std::ofstream myfile;
    myfile.open(filename);
    for(int i = 0; i < K; i++)
    {
        for(int j = 0; j < N; j++)
        {
            int tmpind = N * i + j;
            if(pc[tmpind] == 1)
            {
                myfile << px[tmpind] << " " << py[tmpind] << " " << pz[tmpind] << " " << i << std::endl;

            }
        }
    }
    myfile.close();
}

//writes centroids to a text file
void writeCentroids(float* px, float* py, float* pz, const char* filename){
    std::ofstream myfile;
    myfile.open(filename);
    for(int i = 0; i < K; i++){
        myfile << px[i] << " " << py[i] << " " << pz[i] << " "  << i << std::endl;
    }
    myfile.close();
}

void runTest(int argc, char **argv)
{
    srand(1);

    // mallocing memory on CPU
    float* inputX = (float*) malloc(MVEC);
    float* inputY = (float*) malloc(MVEC);
    float* inputZ = (float*) malloc(MVEC);
    
    //randomizing input vectors
    for (int i = 0; i < N; i++)
    {
        inputX[i] = rf();
        inputY[i] = rf();
        inputZ[i] = rf();
    }

    float* meansX = (float*) malloc(MMEANS);
    float* meansY = (float*) malloc(MMEANS);
    float* meansZ = (float*) malloc(MMEANS);
    
    float* prev_meansX = (float*) malloc(MMEANS);
    float* prev_meansY = (float*) malloc(MMEANS);
    float* prev_meansZ = (float*) malloc(MMEANS);

    //initialization of centroids
    initCentroids(meansX, meansY, meansZ);

    //prev_means are for calculating difference between iterations
    for (int i = 0; i < K; i++)
    {
        prev_meansX[i] = meansX[i];
        prev_meansY[i] = meansY[i];
        prev_meansZ[i] = meansZ[i];
    }

    float* d_inputX, *d_inputY, *d_inputZ, 
    *d_meansX, *d_meansY, *d_meansZ;

    //mallocing memory on GPU
    checkCudaErrors(cudaMalloc((void **) &d_inputX, MVEC));
    checkCudaErrors(cudaMalloc((void **) &d_inputY, MVEC));
    checkCudaErrors(cudaMalloc((void **) &d_inputZ, MVEC));

    checkCudaErrors(cudaMalloc((void **) &d_meansX, MMEANS));
    checkCudaErrors(cudaMalloc((void **) &d_meansY, MMEANS));
    checkCudaErrors(cudaMalloc((void **) &d_meansZ, MMEANS));

    thrust::device_vector<float> d_tmX(K*N), d_tmY(K*N), d_tmZ(K*N);
    thrust::device_vector<int> d_tmC(K*N);
   
    checkCudaErrors(cudaMemcpy(d_inputX, inputX, MVEC, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_inputY, inputY, MVEC, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_inputZ, inputZ, MVEC, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemcpy(d_meansX, meansX, MMEANS, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_meansY, meansY, MMEANS, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_meansZ, meansZ, MMEANS, cudaMemcpyHostToDevice));

    dim3 grid(N/TBP+1, 1, 1);
    dim3 threads(TBP, 1, 1);

    dim3 gridAssigner(K/TBP+1, 1, 1);
    dim3 threadsAssigner(TBP, 1, 1);

    thrust::device_vector<float> sumX(K);
    thrust::device_vector<float> sumY(K);
    thrust::device_vector<float> sumZ(K);
    thrust::device_vector<int> sumC(K);
    
    float diff = INFINITY;
    int max_iterations = 100;
    float epsilon = 0.00001f;
    
    //main loop
    for (int i = 0; i < max_iterations && diff >= epsilon; i++)
    {
        cudaDeviceSynchronize();
    
        thrust::fill(d_tmX.begin(), d_tmX.end(), 0.0f);
        thrust::fill(d_tmY.begin(), d_tmY.end(), 0.0f);
        thrust::fill(d_tmZ.begin(), d_tmZ.end(), 0.0f);
        thrust::fill(d_tmC.begin(), d_tmC.end(), 0);
        
        cudaDeviceSynchronize();
        kmeans<<<grid, threads>>>(d_inputX, d_inputY, d_inputZ,
                                      d_meansX, d_meansY, d_meansZ,
                                      thrust::raw_pointer_cast(&d_tmX[0]), 
                                      thrust::raw_pointer_cast(&d_tmY[0]), 
                                      thrust::raw_pointer_cast(&d_tmZ[0]), 
                                      thrust::raw_pointer_cast(&d_tmC[0]));

        cudaDeviceSynchronize();
 
        //reduce by key operation
        rbk<float>(d_tmX, sumX);
        rbk<float>(d_tmY, sumY);
        rbk<float>(d_tmZ, sumZ);
        rbk<int>(d_tmC, sumC);
        
        cudaDeviceSynchronize();
        getLastCudaError("Kernel execution failed");
        assigner<<<gridAssigner, threadsAssigner>>>(d_meansX, d_meansY, d_meansZ,
                                    thrust::raw_pointer_cast(&sumX[0]), 
                                    thrust::raw_pointer_cast(&sumY[0]), 
                                    thrust::raw_pointer_cast(&sumZ[0]), 
                                    thrust::raw_pointer_cast(&sumC[0]));
        
        cudaDeviceSynchronize();
        checkCudaErrors(cudaMemcpy(meansX, d_meansX, MMEANS, cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(meansY, d_meansY, MMEANS, cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(meansZ, d_meansZ, MMEANS, cudaMemcpyDeviceToHost));

        cudaDeviceSynchronize();

        //calculating norm
        diff = 0;
        for(int j = 0; j < K; j++)
        { 
            diff += h_distance(
                meansX[j], prev_meansX[j],
                meansY[j], prev_meansY[j],
                meansZ[j], prev_meansZ[j]
            );
            prev_meansX[j] = meansX[j];
            prev_meansY[j] = meansY[j];
            prev_meansZ[j] = meansZ[j]; 
        }
        diff/=K;
        //printing result
        printf("iteration: %2d:  %.10f \n", i, diff);
        cudaDeviceSynchronize();
    }
    
    thrust::host_vector<float> tmX(d_tmX);
    thrust::host_vector<float> tmY(d_tmY);
    thrust::host_vector<float> tmZ(d_tmZ);
    thrust::host_vector<int> tmC(d_tmC);

    //writing result to a file
    writeData(thrust::raw_pointer_cast(&tmX[0]), 
    thrust::raw_pointer_cast(&tmY[0]), 
    thrust::raw_pointer_cast(&tmZ[0]), 
    thrust::raw_pointer_cast(&tmC[0]), 
    "outputData.txt");

    writeCentroids(meansX, meansY, meansZ, "outputCentroids.txt");


    //freeing memory
    free(inputX);
    free(inputY);
    free(inputZ);

    free(meansX);
    free(meansY);
    free(meansZ);

    free(prev_meansX);
    free(prev_meansY);
    free(prev_meansZ);

    checkCudaErrors(cudaFree(d_inputX));
    checkCudaErrors(cudaFree(d_inputY));
    checkCudaErrors(cudaFree(d_inputZ));

    checkCudaErrors(cudaFree(d_meansX));
    checkCudaErrors(cudaFree(d_meansY));
    checkCudaErrors(cudaFree(d_meansZ));
}
