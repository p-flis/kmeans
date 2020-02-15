#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <cuda_runtime.h>

#include <helper_cuda.h>
#include <helper_functions.h>

#define K 51
#define N (256*100)
#define TBP 256

#define MVEC (sizeof(float)*N) //memory for vectors
#define MMEANS (sizeof(float)*K) //memory for means
#define MINVEC (sizeof(int)*N) //memory for vector indexes

void runTest(int argc, char **argv);

//calculates distance in 3D on CPU
float h_distance(float x1, float x2, float y1, float y2, float z1, float z2)
{
    float x = x1-x2;
    float y = y1-y2;
    float z = z1-z2;
    return x*x+y*y+z*z;
}

//calculates distance in 3D on GPU
__device__ float distance(float x1, float x2, float y1, float y2, float z1, float z2)
{
    float x = x1-x2;
    float y = y1-y2;
    float z = z1-z2;
    return x*x+y*y+z*z;
}

//kernel which finds the closest centroid for every input vector (gather-scatter)
__global__ void
kmeans(float* g_inputX, float* g_inputY, float* g_inputZ, float* g_meansX, float* g_meansY, float* g_meansZ, int* g_means)
{
    const int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if(idx >= N) return;

    float best = INFINITY;
    float besti = -1;

    float x = g_inputX[idx];
    float y = g_inputY[idx];
    float z = g_inputZ[idx];

    for(int i = 0; i < K; i++)
    {
        float mx = g_meansX[i];
        float my = g_meansY[i];
        float mz = g_meansZ[i];

        float d = distance(x, mx, y, my, z, mz);
        if(d < best)
        {
            best = d;
            besti = i;
        }
    }

    g_means[idx] = besti;
}

//kernel that performs naive reduction
__global__ void
reducerKernel(float* g_inputX, float* g_inputY, float* g_inputZ, float* g_meansX, float* g_meansY, float* g_meansZ, int* g_means, float* g_c)
{
    const int idx = blockIdx.x*blockDim.x + threadIdx.x;
    const int idt = threadIdx.x;


    if(idx >= N) return;


    __shared__ float s_data[4*TBP];

    s_data[idt] = g_inputX[idx];

    s_data[TBP+idt] = g_inputY[idx];

    s_data[2*TBP+idt] = g_inputZ[idx];

    s_data[3*TBP+idt] = g_means[idx];

    __syncthreads();


    if(idt == 0)
    {
        float meansX[K] = {0};
        float meansY[K] = {0};
        float meansZ[K] = {0};

        for(int a = 0; a < K; a++)
        {
            meansX[a] = meansY[a] = meansZ[a] = 0;
        }
        int clu[K] = {0};


        for(int i = 0; i < blockDim.x; i++)
        {
            int k = (int)s_data[3*TBP+i];
            meansX[k] += s_data[i];
            meansY[k] += s_data[TBP+i];
            meansZ[k] += s_data[2*TBP+i];
            clu[k]++;
        }
        for(int i = 0; i < K; i++)
        {
            atomicAdd(g_meansX+i, meansX[i]);
            atomicAdd(g_meansY+i, meansY[i]);
            atomicAdd(g_meansZ+i, meansZ[i]);
            atomicAdd(g_c+i, clu[i]);
        }
    }

    __syncthreads();

}


//kernel which assigns recalculated centroids on GPU
__global__ void
assigner(float* g_inputX, float* g_inputY, float* g_inputZ, float* g_meansX, float* g_meansY, float* g_meansZ, int* g_means, float* g_c)
{
    const int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if(idx < K)
    {
        g_meansX[idx] /= g_c[idx];
        g_meansY[idx] /= g_c[idx];
        g_meansZ[idx] /= g_c[idx];
    }
}



int
main(int argc, char **argv)
{
    runTest(argc, argv);
}

//function that returns random float
float rf()
{
    return rand()/(float)RAND_MAX;
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
void writeData(float* px, float* py, float* pz, int* centroids, const char* filename){
    std::ofstream myfile;
    myfile.open(filename);
    for(int i = 0; i < N; i++){
        myfile << px[i] << " " << py[i] << " " << pz[i] << " "  << centroids[i] << std::endl;
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

void
runTest(int argc, char **argv)
{
    srand(2);

    // mallocing memory on CPU
    float* inputX = (float*)malloc(MVEC);
    float* inputY = (float*)malloc(MVEC);
    float* inputZ = (float*)malloc(MVEC);

    //randomizing input vectors
    for(int i = 0; i < N; i++)
    {
        inputX[i] = rf();
        inputY[i] = rf();
        inputZ[i] = rf();
    }

    float* meansX = (float*)malloc(MMEANS);
    float* meansY = (float*)malloc(MMEANS);
    float* meansZ = (float*)malloc(MMEANS);
  
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

    int* means = (int*)malloc(MINVEC);

    float* d_inputX, *d_inputY, *d_inputZ, *d_meansX, *d_meansY, *d_meansZ;

    int* d_means;
    float* c_c;

    //mallocing memory on GPU
    checkCudaErrors(cudaMalloc((void **) &d_inputX, MVEC));
    checkCudaErrors(cudaMalloc((void **) &d_inputY, MVEC));
    checkCudaErrors(cudaMalloc((void **) &d_inputZ, MVEC));

    checkCudaErrors(cudaMalloc((void **) &d_meansX, MMEANS));
    checkCudaErrors(cudaMalloc((void **) &d_meansY, MMEANS));
    checkCudaErrors(cudaMalloc((void **) &d_meansZ, MMEANS));

    checkCudaErrors(cudaMalloc((void **) &d_means, MINVEC));

    checkCudaErrors(cudaMemcpy(d_inputX, inputX, MVEC, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_inputY, inputY, MVEC, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_inputZ, inputZ, MVEC, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemcpy(d_meansX, meansX, MMEANS, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_meansY, meansY, MMEANS, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_meansZ, meansZ, MMEANS, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc((void **) &c_c, MMEANS));

    dim3  grid(N/TBP+1, 1, 1);
    dim3  threads(TBP, 1, 1);

    dim3 gridAssigner(K/TBP+1, 1, 1);
    dim3 threadsAssigner(TBP, 1, 1);

    float diff = INFINITY;
    int max_iterations = 100;
    float epsilon = 0.00001f;

    //main loop
    for (int i = 0; i < max_iterations && diff >= epsilon; i++)
    {
        cudaDeviceSynchronize();
        kmeans<<<grid, threads>>>(d_inputX, d_inputY, d_inputZ, d_meansX, d_meansY, d_meansZ, d_means);

        cudaDeviceSynchronize();

        cudaMemset(d_meansX, 0, MMEANS);
        cudaMemset(d_meansY, 0, MMEANS);
        cudaMemset(d_meansZ, 0, MMEANS);
        cudaMemset(c_c, 0, MMEANS);
        
        cudaDeviceSynchronize();
        reducerKernel<<<grid, threads, 4*TBP*sizeof(float)>>>(d_inputX, d_inputY, d_inputZ, d_meansX, d_meansY, d_meansZ, d_means, c_c);

        cudaDeviceSynchronize();
        assigner<<<gridAssigner, threadsAssigner, 4*TBP*sizeof(float)>>>(d_inputX, d_inputY, d_inputZ, d_meansX, d_meansY, d_meansZ, d_means, c_c);

        cudaDeviceSynchronize();
        checkCudaErrors(cudaMemcpy(means, d_means, MINVEC, cudaMemcpyDeviceToHost));

        checkCudaErrors(cudaMemcpy(meansX, d_meansX, MMEANS, cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(meansY, d_meansY, MMEANS, cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(meansZ, d_meansZ, MMEANS, cudaMemcpyDeviceToHost));

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
    

    }

    //writing result to a file
    writeData(inputX, inputY, inputZ, means, "outputData.txt");
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
    free(means);

    checkCudaErrors(cudaFree(d_inputX));
    checkCudaErrors(cudaFree(d_inputY));
    checkCudaErrors(cudaFree(d_inputZ));

    checkCudaErrors(cudaFree(d_meansX));
    checkCudaErrors(cudaFree(d_meansY));
    checkCudaErrors(cudaFree(d_meansZ));

    checkCudaErrors(cudaFree(d_means));
    checkCudaErrors(cudaFree(c_c));
}
