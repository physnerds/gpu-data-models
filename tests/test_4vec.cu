#include <stdlib.h>
#include <vector>
#include <iostream>
#include <random>
#include <chrono>

#include "TCudaVector.h"

 struct FourVector{
  double _px,_py,_pz,_E;
    int _index;
    
};


//we might need to come back on this later on...
__device__ FourVector *fv;

std::vector<FourVector>GenerateRandomTrack(int _size,double mass){
    std::vector<FourVector>track_container;
    std::mt19937_64 rng;
    // initialize the random number generator with time-dependent seed
    uint64_t timeSeed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::seed_seq ss{uint32_t(timeSeed & 0xffffffff), uint32_t(timeSeed>>32)};
    rng.seed(ss);
    // initialize a uniform distribution between 0 and 1
    std::uniform_real_distribution<double> unif(0, 1);
    // ready to generate random numbers
    
    for (int i = 0; i < _size; i++)
    {
        FourVector _temp_track;
        double rand1 = unif(rng);
        double rand2= unif(rng);
        double rand3 = unif(rng);
        
        _temp_track._px = rand1;
        _temp_track._py = rand2;
        _temp_track._pz = rand3;
        _temp_track._E = sqrt(rand1*rand1+rand2*rand2+rand3*rand3+mass*mass);
        _temp_track._index = i;
        track_container.push_back(_temp_track);

    }
     return track_container;     
}

std::vector<double4>GenerateRandomTracks(int _size,double mass){
    std::vector<double4>track_container;
    std::mt19937_64 rng;
    // initialize the random number generator with time-dependent seed
    uint64_t timeSeed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::seed_seq ss{uint32_t(timeSeed & 0xffffffff), uint32_t(timeSeed>>32)};
    rng.seed(ss);
    // initialize a uniform distribution between 0 and 1
    std::uniform_real_distribution<double> unif(0, 1);
    // ready to generate random numbers
    
    for (int i = 0; i < _size; i++)
    {
        double4 _temp_track;
        double rand1 = unif(rng);
        double rand2= unif(rng);
        double rand3 = unif(rng);
        
        _temp_track.x = rand1;
        _temp_track.y = rand2;
        _temp_track.z = rand3;
        _temp_track.w = sqrt(rand1*rand1+rand2*rand2+rand3*rand3+mass*mass);
        track_container.push_back(_temp_track);

    }
     return track_container;     
}


__global__ void testVector(FourVector* in, FourVector* out,int* index_ptr,int _size){
 
    for(int id= blockIdx.x * blockDim.x + threadIdx.x; 
        id<_size;
        id +=blockDim.x * gridDim.x){
        index_ptr[id] = id;
        printf("Device count %d %d \n",id,index_ptr[id]);
       // (out[id])._px;
        out[id]._px = in[id]._px*2.;
        out[id]._py = in[id]._px*2.;
        out[id]._pz = in[id]._pz*2 ;
        out[id]._E = in[id]._E;
        out[id]._index = in[id]._index;
    }
}

__global__ void testVector3(double4* in, double4* out,int* index_ptr,int _size){
 
    for(int id= blockIdx.x * blockDim.x + threadIdx.x; 
        id<_size;
        id +=blockDim.x * gridDim.x){
        index_ptr[id] = id;
       // printf("Device count %d %d \n",id,index_ptr[id]);
       // (out[id])._px;
         out[id].x = in[id].x*2.;
         out[id].y = in[id].x*2.;
         out[id].z = in[id].z*2 ;
         out[id].w = in[id].w;

    }
}

__global__ void testVector2(int* index_ptr,int _size){
        for(int id= blockIdx.x * blockDim.x + threadIdx.x; 
        id<_size;
        id +=blockDim.x * gridDim.x){
        index_ptr[id] = id;
        printf("Device count %d %d \n",id,index_ptr[id]);
        }
}

int RunCudaVectorTypes(){
    
    int numThreads = 1;
    dim3 threadsPerBlock( 1024, 1, 1);
    
    dim3 numberofBlocks( (numThreads + threadsPerBlock.x-1)/threadsPerBlock.x,1,1);   
    
    auto trajectory = GenerateRandomTracks(100,134.34);
    std::cout<<"size of the structure "<<sizeof(double4)<<std::endl;    
    //can I use this into a CudaVector
    //we can offload it into the Cuda friendly stuff..
    CudaVector<double4> trial(trajectory);
    std::cout<<trial[0].x<<std::endl;
    auto dev_trial = trial.ReturnDeviceVector();
    size_t f_size = sizeof(double4);//(4*sizeof(double)+sizeof(int));
    double4* ptr_trial;
    cudaMallocManaged((void**)&ptr_trial,100*f_size);
    ptr_trial = thrust::raw_pointer_cast( dev_trial.data() );
    
    
    double4 *ptr_out;
    cudaError_t ret = cudaMallocManaged((void**)&ptr_out, 100*f_size);
    
    if (ret != cudaSuccess) {
    std::cout << cudaGetErrorString(ret) << std::endl;
    return 1;}
    
    int *index_ptr;
    cudaMallocManaged(&index_ptr,100*sizeof(int));
       
    testVector3<<<1,100>>>(ptr_trial,ptr_out,index_ptr,100);   
    
   // testVector2<<<1,100>>>(index_ptr,100); 
    double4 *host_out = new double4[100];
    
    int *index_out = new int[100];
    
    cudaMemcpy(host_out,ptr_out,100*f_size,cudaMemcpyDeviceToHost);
    cudaMemcpy(index_out,index_ptr,100*sizeof(int),cudaMemcpyDeviceToHost);
    
        
    for(int i = 0;i<10;i++){
        std::cout<<index_out[i]<<std::endl;
        std::cout<<host_out[i].x<<std::endl;
        //std::cout<<host_out[i]._px<<" "<<host_out[i]._index<<" "<<trial[i]._px<<
     //   " "<<index_out[i]<<std::endl;   
    }
    free(host_out);
    cudaFree(ptr_out);
    return 0;
}

int RunCPPStructTypes(){
    
    int numThreads = 1;
    dim3 threadsPerBlock( 1024, 1, 1);
    
    dim3 numberofBlocks( (numThreads + threadsPerBlock.x-1)/threadsPerBlock.x,1,1);   
    
    auto trajectory = GenerateRandomTrack(100,134.34);
    std::cout<<"size of the structure "<<sizeof(FourVector)<<std::endl;    
    //can I use this into a CudaVector
    //we can offload it into the Cuda friendly stuff..
    CudaVector<FourVector> trial(trajectory);
    std::cout<<trial[0]._px<<std::endl;
    auto dev_trial = trial.ReturnDeviceVector();
    size_t f_size = sizeof(FourVector);//(4*sizeof(double)+sizeof(int));
    FourVector* ptr_trial;
    cudaMallocManaged((void**)&ptr_trial,100*f_size);
    ptr_trial = thrust::raw_pointer_cast( dev_trial.data() );
    
    
    FourVector *ptr_out;
    cudaError_t ret = cudaMallocManaged((void**)&ptr_out, 100*f_size);
    
    if (ret != cudaSuccess) {
    std::cout << cudaGetErrorString(ret) << std::endl;
    return 1;}
    
    int *index_ptr;
    cudaMallocManaged(&index_ptr,100*sizeof(int));
       
    testVector<<<1,100>>>(ptr_trial,ptr_out,index_ptr,100);   
    
   // testVector2<<<1,100>>>(index_ptr,100); 
    FourVector *host_out = new FourVector[100];
    
    int *index_out = new int[100];
    
    cudaMemcpy(host_out,ptr_out,100*f_size,cudaMemcpyDeviceToHost);
    cudaMemcpy(index_out,index_ptr,100*sizeof(int),cudaMemcpyDeviceToHost);
    
        
    for(int i = 0;i<10;i++){
        std::cout<<index_out[i]<<std::endl;
        std::cout<<host_out[i]._px<<" "<<host_out[i]._index<<std::endl;
  
    }
    free(host_out);
    cudaFree(ptr_out);
    return 0;
}

int main(int argc,char* argv[]){
    
    //RunCudaVectorTypes();
    RunCPPStructTypes();
}
