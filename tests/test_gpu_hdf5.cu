#ifndef H5_WRITE_CHARS_H
#define H5_WRITE_CHARS_H


#include "time.h"
#include <ctime>

#include <string>
#include <cstddef>
#include "TFile.h"
#include "TROOT.h"
#include "TTree.h"
#include "TClass.h"
#include "TBranch.h"
#include "TString.h"
#include "gpu_kernels.h"
#include "serialize_dataprods.h"
#include "rootUtilities.h"
//#include "CudaVector.h"
#include "TCudaVector.h"

//thrust libraries
#include <thrust/reduce.h>
#include<thrust/execution_policy.h>
#include<thrust/random.h>
#include<thrust/device_vector.h>
#include<thrust/host_vector.h>

#define DSIZE 5
// error checking macro
#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

template<typename T>
struct arrayStruct{
    int nentries=0;
    T arr1;
    T arr2;
            
};

template<typename T>
__global__ void testVector(T* A,T* B , T* C, int* idx_C)
{
   int id =  (blockIdx.x * blockDim.x) + threadIdx.x;
    C[id] = A[id] + B[id];
    idx_C[id] = id;
   
}
    


__host__ static __inline__ float rand_01()
{
    return ((float)rand()/RAND_MAX);
}


int test_rand(){
   thrust::host_vector<float> h_1(DSIZE);

  thrust::generate(h_1.begin(), h_1.end(), rand_01);
  std::cout<< "Values generated: " << std::endl;
  for (unsigned i=0; i<DSIZE; i++)
    std::cout<< h_1[i] << " : ";
  std::cout<<std::endl;
return 0;
}   
/*
int test_tcuda_struct(){
      int numThreads = 1;
    dim3 threadsPerBlock( 1024, 1, 1);
    
    dim3 numberofBlocks( (numThreads + threadsPerBlock.x-1)/threadsPerBlock.x,1,1);  
    
    std::vector<std::string>b_names = {"phi","Q2"};
    std::string _filename = "../rfiles/For_TMVA_DATA_minervame5A6A6B_kludged.root";
    
    std::string tree_name = "ccqe_data";
    
    auto _file = new TFile(_filename.c_str(),"READ");
    auto _tree = (TTree*)_file->Get(tree_name.c_str());    
    auto tot_entries = _tree->GetEntriesFast();
    tot_entries = 100;
    std::map<std::string,std::vector<double>> map_vals = FillWithVals<double>(_tree, b_names,tot_entries);
    std::cout<<"Total Entries is "<<tot_entries<<std::endl;
    
    std::vector<double>phi = map_vals["phi"];
    std::vector<double>q2 = map_vals["q2"];
       
    //std::vector<arrayStruct<double>>my_vec;
    Vector<arrayStruct<double>>test_vector(phi.size());
    return 1;
}
*/
int test_tcuda_vector(){
    clock_t t0,t1;
    double t1sum=0;

    int numThreads = 1;
    dim3 threadsPerBlock( 1024, 1, 1);
    
    dim3 numberofBlocks( (numThreads + threadsPerBlock.x-1)/threadsPerBlock.x,1,1);    

    int dims = 1;
    int num_threads = 100;//nentries;//nentries/2;
    int s_val = 0;
    
    std::vector<std::string>b_names = {"phi","Q2"};
    std::string _filename = "../rfiles/For_TMVA_DATA_minervame5A6A6B_kludged.root";
    std::string tree_name = "ccqe_data";
    auto _file = new TFile(_filename.c_str(),"READ");
    auto _tree = (TTree*)_file->Get(tree_name.c_str());    
    auto tot_entries = _tree->GetEntriesFast();
    tot_entries = 100;
    std::map<std::string,std::vector<double>> map_vals = FillWithVals<double>(_tree, b_names,tot_entries);
    std::cout<<"Total Entries is "<<tot_entries<<std::endl;
    std::vector<double>phi = map_vals["phi"];
    CudaVector<double> cuda_phi(phi);
    
    std::vector<double>q2 = map_vals["Q2"];
    CudaVector<double> cuda_q2(q2);
   
    auto dev_cuda_q2 = cuda_q2.ReturnDeviceVector();
   
    auto dev_cuda_phi = cuda_phi.ReturnDeviceVector();
    
   //cannot internalize this as well....
    // double* ptr_phi =cuda_phi.GetDeviceArrayPointer(dev_cuda_phi);
    double* ptr_phi = thrust::raw_pointer_cast( &dev_cuda_phi[0] );
    
    //cannot internalize this part for some reason
    //double* ptr_q2 =cuda_q2.GetDeviceArrayPointer(dev_cuda_q2);   
    double* ptr_q2 = thrust::raw_pointer_cast( &dev_cuda_q2[0] );//cuda_q2.GetDeviceArrayPointer();
    
    double* ptr_out;
    int* ptr_int;
    
    cudaMalloc(&ptr_out, tot_entries*sizeof(double)); 
    cudaMalloc(&ptr_int, tot_entries*sizeof(int)); 
       
    testVector<<<threadsPerBlock,numberofBlocks>>>(ptr_phi,ptr_q2, ptr_out,ptr_int);
    int st = 0;
 //  VecAdd<<<dims,num_threads>>>(ptr_phi,ptr_q2,ptr_out,st);
    double *val_c = new double[tot_entries];
    int *val_int = new int[tot_entries];
     
    cudaMemcpy(val_c, ptr_out, tot_entries*sizeof(double), cudaMemcpyDeviceToHost); 
    cudaMemcpy(val_int,ptr_int,tot_entries*sizeof(int),cudaMemcpyDeviceToHost);
        
    for(int i = 0;i<q2.size();i++){
        std::cout<<q2[i]<<" "<<phi[i]<<" "<<val_c[i]<<" "<<
            q2[i]+phi[i]<<" "<<val_int[i]<<std::endl;        
}
    return 1;
    
}


int test_cuda_vector(){
    clock_t t0,t1;
    double t1sum=0;

    std::vector<std::string>b_names = {"phi","Q2"};
    std::string _filename = "../rfiles/For_TMVA_DATA_minervame5A6A6B_kludged.root";
    std::string tree_name = "ccqe_data";
    auto _file = new TFile(_filename.c_str(),"READ");
    auto _tree = (TTree*)_file->Get(tree_name.c_str());    
    auto tot_entries = _tree->GetEntriesFast();
    tot_entries = 100;
    std::map<std::string,std::vector<double>> map_vals = FillWithVals<double>(_tree, b_names,tot_entries);

    std::vector<double>phi = map_vals["phi"];
    Vector<double>cuda_phi(phi);
    
    std::vector<double>Q2 = map_vals["Q2"];
    Vector<double>cuda_q2(Q2);
    
    //test the copy of the vector...
    Vector<double>device_phi = cuda_phi.Host_Device_Cpy();
    Vector<double>device_q2 =  cuda_q2.Host_Device_Cpy();
    
    Vector<double>total_val(Q2.size());
    total_val.AllocateDeviceMemory();
    std::cout<<"size of cuda_phi "<<cuda_phi.GetSize()<<" "<<device_phi.GetSize()<<std::endl;

    int dims = 1;
    int num_threads = 1;  
   

    CudaVectorAdd(dims,num_threads,device_phi,device_q2,total_val);

    cudaDeviceSynchronize();    
    
   // Vector<double> host_total_val = total_val.Device_Host_Cpy();
      Vector<double>host_total_val(100);
    std::cout<<"Size of host_total_val "<<host_total_val.GetSize()<<std::endl;
    for(int i=0;i<host_total_val.GetSize();i++)std::cout<<host_total_val[i]<<" "<<cuda_q2[i]<<" "<<cuda_phi[i]<<std::endl;
    
    return 1;
    
}

int test_gpu_data_vec(){
    clock_t t0,t1;
    double t1sum=0;

    std::vector<std::string>b_names = {"phi","Q2"};
    std::string _filename = "../rfiles/For_TMVA_DATA_minervame5A6A6B_kludged.root";
    std::string tree_name = "ccqe_data";
    auto _file = new TFile(_filename.c_str(),"READ");
    auto _tree = (TTree*)_file->Get(tree_name.c_str());    
    auto tot_entries = _tree->GetEntriesFast();
    tot_entries = 1;
    std::map<std::string,std::vector<double>> map_vals = FillWithVals<double>(_tree, b_names,tot_entries);

        
    double *array_add_cuda = new double[tot_entries];
    double *array_phi_cuda = new double[tot_entries];
    double *array_q2_cuda  = new double[tot_entries];    
    
    std::vector<double>phi = map_vals["phi"];
    Vector<double>cuda_phi(phi.size()); //This needs to go away...need to write a function.
    Vector<double>temp(phi.size()); 
    std::cout<<" HERE "<< phi.size()<<std::endl;
    
    std::vector<double>q2 = map_vals["Q2"];
    Vector<double>cuda_q2(q2.size());
    
    for(int i=0;i<cuda_q2.GetSize();++i){
        cuda_q2.set(i,q2[i]);
        cuda_phi.set(i,phi[i]);
        array_phi_cuda[i] = phi[i];
        array_q2_cuda[i] = q2[i];
        array_add_cuda[i] = 0.0;
        temp.set(i,0.0);
    }
    

    double *device_add_cuda = new double[tot_entries];
    double *device_phi_cuda = new double[tot_entries];
    double *device_q2_cuda  = new double[tot_entries];      
    
    
     t0 = clock();
     Vector<double>*cuda_add_vector,*device_cuda_q2,*device_cuda_phi,*host_add_vector;

   //  cudaMalloc((void**)&cuda_add_vector,q2.size()*sizeof(double));
    
     cudaMalloc(&device_add_cuda, tot_entries*sizeof(double));
     cudaMalloc(&device_phi_cuda, tot_entries*sizeof(double));
     cudaMalloc(&device_q2_cuda, tot_entries*sizeof(double));
    
   //  cudaMalloc((void**)&device_cuda_q2,q2.size()*sizeof(double));
   //  cudaMalloc((void**)&device_cuda_phi,q2.size()*sizeof(double));

    int nentries = phi.size();
    std::cout<<"Entries "<<nentries<<std::endl;
  //  cudaMemcpy(device_add_cuda,array_phi_cuda,nentries*sizeof(double),cudaMemcpyHostToDevice);
  //  cudaMemcpy(device_phi_cuda,array_phi_cuda,nentries*sizeof(double),cudaMemcpyHostToDevice);
  //  cudaMemcpy(device_q2_cuda,array_q2_cuda,nentries*sizeof(double),cudaMemcpyHostToDevice);    
  //  cudaMemcpy(device_cuda_q2,&cuda_q2[0], nentries*sizeof(double), cudaMemcpyHostToDevice);   
  //  cudaMemcpy(device_cuda_phi,&cuda_phi[0], nentries*sizeof(double), cudaMemcpyHostToDevice);
  //  cudaMemcpy(cuda_add_vector,&temp[0], nentries*sizeof(double), cudaMemcpyHostToDevice);
    
    std::cout<<"Copied memory to the host device "<<std::endl;
    
    int dims = 1;
    int num_threads = 100;  
   
    ArrayCudaVectorAdd<<<dims,num_threads>>>(device_phi_cuda,device_q2_cuda,device_add_cuda,nentries);
    
    std::cout<<"Completed the calculation "<<std::endl;
    //now copy back...
    double *host_vector = new double[nentries]; 
    std::fill(host_vector, host_vector+nentries, 0.0);
    
    cudaMemcpy(host_vector, device_add_cuda, cuda_phi.GetSize()*sizeof(double), cudaMemcpyDeviceToHost);
    t1 = clock();
    t1sum =  ((double)(t1-t0))/CLOCKS_PER_SEC;
    
    std::cout<<"ELEMENTS "<<phi[0]<<"  "<<q2[0]<<" "<<std::endl;
    std::cout<<"Test End sum "<<host_vector[nentries-1]<<" "<<phi[nentries-1]+q2[nentries-1]<<std::endl;
    std::cout<<"Test Start sum "<<host_vector[0]<<" "<<phi[0]+q2[0]<<std::endl;
    cudaFree(host_vector);
    
    std::cout<<" Total RunTime "<<t1sum<<std::endl;
    //for(int i=0;i<nentries;i++)std::cout<<host_vector[i]<<std::endl;
    
    return 1;
    
}

int test_gpu_data_array(){
    clock_t t0,t1;
    double t1sum=0;
    t0 = clock();
  
    std::cout<<"HERE"<<std::endl;
    std::string _filename = "../rfiles/For_TMVA_DATA_minervame5A6A6B_kludged_small_stats.root";
    std::string tree_name = "ccqe_data";
    auto _file = new TFile(_filename.c_str(),"READ");
    auto _tree = (TTree*)_file->Get(tree_name.c_str());
    auto l = _tree->GetListOfBranches();
    
    auto branch_names = return_dsetnames(l);
    std::cout<<" HERE 2 "<<std::endl;
    for(auto name:branch_names)std::cout<<"Branch "<<name<<std::endl;
    int nentries = _tree->GetEntriesFast();
    nentries = 100;
    auto tot_branches = l->GetEntriesFast();
    std::cout<<"nentries "<<nentries<<" tot branch "<<tot_branches<<std::endl;
    double *val_a, *val_b, *val_c, *d_A, *d_B, *d_C;
    
    val_a = new double[nentries];
    val_b = new double[nentries];
    val_c = new double[nentries];
    for(int i=0;i<nentries;i++){
        _tree->GetEntry(i);
        for(Long64_t jentry=0;jentry<tot_branches;++jentry){
            auto b = dynamic_cast<TBranch*>((*l)[jentry]);
            
            auto leaves = b->GetListOfLeaves();
            auto leaf = dynamic_cast<TLeaf*>((*leaves)[0]);
            std::string b_name = b->GetName();
            if(b_name=="Q2")val_a[i] = leaf->GetValue(0);
            if(b_name == "recoil_E")val_b[i] = leaf->GetValue(0);
           // std::cout<<"Reading for "<<b_name<<" "<<i<<std::endl;
        }
        val_c[i] = 0.0;
    }

   // cudaCheckErrors("cudaMalloc failure");
    std::cout<<"Finished reading the branches "<<std::endl;
    cudaMalloc(&d_A, nentries*sizeof(double));
    //cudaCheckErrors("cudaMalloc failure");
    cudaMalloc(&d_B, nentries*sizeof(double));  
    //cudaCheckErrors("cudaMalloc failure");
    cudaMalloc(&d_C, nentries*sizeof(double));
    std::cout<<"Allocating the memories "<<std::endl;
    
    
    cudaMemcpy(d_A, val_a, nentries*sizeof(double), cudaMemcpyHostToDevice);    
    cudaMemcpy(d_B, val_b, nentries*sizeof(double), cudaMemcpyHostToDevice);    
    cudaCheckErrors("cudaMalloc failure");
    //this is a 1D addition....so probably we can just allocate the number of threads accurdingly.
    //1D and lets say 2 entries per thread.
    std::cout<<"Finished allocating the memories"<<std::endl;
    int dims = 1;
    int num_threads = 100;//nentries;//nentries/2;
    int s_val = 0;
    VecAdd<<<dims,num_threads>>>(d_A,d_B,d_C,nentries,s_val);

    std::cout<<"Finished doing the calculation "<<std::endl;
    cudaCheckErrors("cudaMalloc failure");
    cudaMemcpy(val_c, d_C, nentries*sizeof(double), cudaMemcpyDeviceToHost); 
     cudaCheckErrors("cudaMalloc failure");
    //check the cosistency....
    for(int i=0;i<nentries;i++)
        std::cout<<val_a[i]+val_b[i]<<" "<<val_c[i]<<" "<<i<<std::endl;
    t1 = clock();
    t1sum = ((double)(t1-t0))/CLOCKS_PER_SEC;
    std::cout<<"End of the code Time taken "<<t1sum<<std::endl;
    return 1;
 
}



#endif

int main(int argc, char* argv[]){
 //   test_rand();
 // test_gpu_data_vec();
 // test_gpu_data_array();
  test_tcuda_vector();
  return 1;

}

