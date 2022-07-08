
#include "Memory.cuh"
#include "OffloadIntoHost.h"
#include "ReadRawDataHDF5.h"
#include "TCudaVector.h"
#include "Info.h"

#include<tuple>
#include<experimental/tuple>
#include<type_traits>
//Added for some testing purpose only....
#include "tbb/task_group.h"
#include "tbb/global_control.h"
#include "tbb/task_arena.h"
#include "DeviceFunctions.h"


/*
the unique_ptr should have the ordering as such....
1. size of arrays
2. number of arrays
3. arrays
4. leftover arguments.
*/


class Test_Functor{
 public:
    HOST_AND_DEVICE
    void operator()(uint32_t* array1,uint32_t* array2,uint32_t* array_out,int _index, bool add){
        if(add){
            array_out[_index] = 
            array1[_index]+array2[_index];
        }
        else{   
         if(array1[_index]>array2[_index]){   
         array_out[_index] = 
             array1[_index]-array2[_index] ; 
         }
            else{
         array_out[_index] = 
             array1[_index]-array2[_index] ;                
            }
        }
    }
};

class TestFunc2{
 public:
    HOST_AND_DEVICE
    void operator()(uint32_t* array1,uint32_t* array_out,int _size){
        for(int id = 0;id<_size;
            id++){
        array_out[id] = 
        array1[id];
        }
    }
};

namespace Task{
    
template<class FUNCTOR, typename... ARGS>
__global__ void deviceKernel(std::size_t arraySizes, ARGS... args){

       const std::size_t i = blockIdx.x*blockDim.x+threadIdx.x;
       if(i>=arraySizes ) return;
       
       FUNCTOR()(args...);
     // FUNCTOR(args...)();
       return;

}

template<class FUNCTOR,class... ARGS>
int deviceExecute(cudaStream_t stream,std::size_t arraySizes,std::size_t array_numbers, ARGS... args){
   int nThreadsPerBlock = 1024;
   
   for(int i=0;i<Info::instance().nDevices();++i){
while(nThreadsPerBlock>Info::instance().maxThreadsPerBlock()[i]){
   nThreadsPerBlock/=2;
   
  }
  }
  if(arraySizes<nThreadsPerBlock){
    nThreadsPerBlock = arraySizes;
  }
  const int nBlocks = ((arraySizes + nThreadsPerBlock-1)/nThreadsPerBlock);

  deviceKernel<FUNCTOR><<<nBlocks, nThreadsPerBlock,0,stream>>>(arraySizes,args...);
   
   return 0;
  
}

/*
template<typename Tuple,std::size_t... I>
void test_execute(Tuple const& tuple, std::index_sequence<I...>){
  execute(std::get<I>(tuple)...);

}
*/

template<class FUNCTOR, typename... ARGS>    
class MyTask{
   static_assert(sizeof... (ARGS)>=0,
       "At least one argument needed ");
   
   public:
     MyTask(cudaStream_t &stream, tbb::task_arena &arena, std::size_t arraySizes,std::size_t arrayno,  ARGS... args );
     
     bool execute(ARGS... args);
     
     bool CopyDeviceToHost();
     
     private:
      tbb::task_arena m_arena;
      std::size_t m_size;
      std::size_t m_arrano;
      std::tuple<ARGS...>m_tuple;
      std::tuple<ARGS... >m_objects; 
      cudaStream_t m_stream;
     // ARGS... m_args;

};

template<class FUNCTOR, typename... ARGS>
std::unique_ptr<MyTask<FUNCTOR,ARGS...>>make_Task(cudaStream_t &stream, tbb::task_arena &arena, size_t arraySizes, std::size_t arrano, ARGS... args ){

    return std::make_unique<MyTask<FUNCTOR,ARGS...>>
    (stream, arena,arraySizes, arrano, args...);

}

template<class FUNCTOR, typename... ARGS>
MyTask<FUNCTOR,ARGS...>::MyTask(cudaStream_t &stream,tbb::task_arena &arena, std::size_t arraySizes, std::size_t arrano, ARGS... args):
    m_stream(std::move(stream)),m_arena(std::move(arena)),m_arrano(arrano),m_size(arraySizes),m_tuple(args...){}
    
template<class FUNCTOR, typename... ARGS>
bool MyTask<FUNCTOR,ARGS...>::execute(ARGS... args){

   if(m_stream==nullptr){
      auto fun = FUNCTOR();
      m_arena.execute([&fun,args...](){
      tbb::task_group group;
      group.run([&](){

      fun(args...);

        });
        group.wait();
      });

   }
   else{
    size_t args_size = sizeof...(args);
    
    deviceExecute<FUNCTOR>(m_stream,m_size,m_arrano,args...);
   }
   

}
    
}

int main(){
    std::string f_name = "dune_file.h5";
    int ch_id = 2500;
    std::string apa_name = "APA01";
    auto handler = OffloadIntoHost(f_name);
    auto apa_id = handler.GetAPAHandle(apa_name);
    
    ReadRawDataHDF5::DuneRawDataHeader c_header[1];
    handler.ReturnDataHeader(apa_id,ch_id,c_header);
    CudaVector<ReadRawDataHDF5::DuneRawDataHeader>cu_header =
        handler.OffloadHeaderIntoCudaArray(c_header);
    
    auto dat_size = c_header[0].Nadc_;
    
    CudaVector<uint32_t> cu_data = handler.OffloadDataIntoCudaArray(apa_id, c_header[0].chan_, c_header[0].Nadc_);
    
    auto dev_vector = cu_data.ReturnDeviceVector();
    
    auto host_vector = cu_data.ReturnHostVector();
    size_t u_size = sizeof(uint32_t);    
    
    cudaStream_t stream = nullptr;
    
    std::cout<<"nDevices "<<Info::instance().nDevices()<<std::endl;
    
    if(Info::instance().nDevices()>0){
       CUDA_EXP_CHECK(cudaStreamCreate(&stream));   
    }
    
    tbb::task_arena arena(1);
    tbb::task_group tg;  
    
    uint32_t* arr1;
    
    arr1 =
        thrust::raw_pointer_cast(dev_vector.data());
    
    uint32_t* arr2;
        arr2 = thrust::raw_pointer_cast(dev_vector.data());
    
    
    uint32_t* arr_out;// = new uint32_t[u_size];
    
    if(stream!=nullptr)
       arr_out =  Device::AllocateMemory<uint32_t>(u_size);
    else arr_out = new uint32_t[u_size];

   // uint32_t *pt1 = Device::ReturnArray<uint32_t>(stream,cu_data);
  //  uint32_t *pt2 = Device::ReturnArray<uint32_t>(stream,cu_data);
    /*
    uint32_t* pt1 = thrust::raw_pointer_cast(dev_vector.data());
    uint32_t* pt2 = thrust::raw_pointer_cast(dev_vector.data());
    auto task = Task::make_Task<Test_Functor>
(stream,arena,dat_size,3,pt1,pt2,arr_out,5,true);
    
    task->execute(pt1,pt2,arr_out,5,true);
        
    //uint32_t *host_out;
    uint32_t* host_out = Device::CopyArrayFromDevice(stream,arr_out,dat_size);
    */
    
    cudaEvent_t start,stop;
    float time;
   uint32_t* pt1 = thrust::raw_pointer_cast(dev_vector.data());
    auto task = Task::make_Task<TestFunc2>(stream,arena,dat_size,2,pt1,arr_out,dat_size);
    
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
   
    cudaEventRecord(start,0);
    task->execute(pt1,arr_out,dat_size);

    
    cudaEventRecord(stop,0);
    
    cudaEventElapsedTime(&time,start,stop);
    uint32_t* host_out = Device::CopyArrayFromDevice(stream,arr_out,dat_size);
    for(int i=0;i<dat_size;i++){
   std::cout<<host_out[i]<<" "<<host_vector[i]<<std::endl;
        
    }
    std::cout<<"Time execution "<<time<<std::endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 1;
}
