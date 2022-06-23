#include "KernelStatus.h"
#include "TaskArena.h"
//#include "OffloadKernelTasks.h"
#include "OffloadIntoHost.h"
//#include "ArrayKernelTask.cuh"
#include "Memory.cuh"
#include "ReadRawDataHDF5.h"
#include "TCudaVector.h"
#include "Info.h"
#include "DeviceFunctions.h"
#include<tuple>
#include<type_traits>
//Added for some testing purpose only....
#include "tbb/task_group.h"
#include "tbb/global_control.h"
#include "tbb/task_arena.h"





namespace Task2{

//declare some 
template<class FUNCTOR, typename... ARGS>
__global__ void deviceKernel(std::size_t arraySizes, ARGS... args){

       const std::size_t i = blockIdx.x*blockDim.x+threadIdx.x;
       if(i>=arraySizes ) return;
       
       FUNCTOR()(args...);
     // FUNCTOR(args...)();
       return;
     

}

template<class FUNCTOR, class... ARGS>
int deviceExecute(cudaStream_t stream, std::size_t arraySizes, ARGS... args){
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
    
template<class FUNCTOR, typename... ARGS>    
class MyTask{
   static_assert(sizeof... (ARGS)>=0,
       "At least one argument needed ");
   
   public:
     MyTask(cudaStream_t &stream, tbb::task_arena &arena, std::size_t arraySizes, ARGS... args );
     
     bool execute(ARGS... args);
     
     bool CopyDeviceToHost();
     
     private:
      tbb::task_arena m_arena;
      std::size_t m_size;
      std::tuple<ARGS...>m_tuple;
      std::tuple<ARGS... >m_objects; 
      cudaStream_t m_stream;
};
    
template<class FUNCTOR, typename... ARGS>
std::unique_ptr<MyTask<FUNCTOR,ARGS...>>make_Task(cudaStream_t &stream, tbb::task_arena &arena,
                                                  size_t arraySizes,ARGS... args){
 return std::make_unique<MyTask<FUNCTOR,ARGS...>>(stream,arena,arraySizes,args...);   
}
    
template<class FUNCTOR, typename... ARGS>
MyTask<FUNCTOR,ARGS...>::MyTask(cudaStream_t &stream, tbb::task_arena &arena, std::size_t arraySizes, ARGS... args):
    m_stream(std::move(stream)),m_arena(std::move(arena)),m_size(arraySizes),m_tuple(args...){}
                                                                 
template<class FUNCTOR, typename... ARGS>
bool MyTask<FUNCTOR,ARGS...>::execute(ARGS... args){

 //  FUNCTOR fun(std::get<0>(m_tuple),std::get<1>(m_tuple));
   //FUNCTOR fun(args...);
   auto fun = FUNCTOR();
   if(m_stream==nullptr){
   m_arena.execute([&fun,args...](){
     tbb::task_group group;
     group.run([&](){
      //  fun.RunArray();
        fun(args...);
    });
     group.wait();
   });

   }
   else{
  deviceExecute<FUNCTOR>(m_stream,m_size,args...); 
   
   }
}


}


namespace Devices {
    
//declare some variables here:::
   template< typename T >
   using device_array = std::unique_ptr< T, details::DeviceArrayDeleter >;

   /// Function creating a primitive array in CUDA device memory
   template< typename T >
   device_array< T > make_device_array( std::size_t size );    
    
template<class FUNCTOR, typename... ARGS>
__global__ void deviceKernel(std::size_t arraySizes, ARGS... args){

       const std::size_t i = blockIdx.x*blockDim.x+threadIdx.x;
       if(i>=arraySizes ) return;
       
       FUNCTOR()(i, args...);
       return;
     

}

template<class FUNCTOR, class... ARGS>
int deviceExecute(cudaStream_t stream, std::size_t arraySizes, ARGS... args){
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



template<bool IsArray,std::size_t Index, typename... ARGS>
class MakeDeviceObjectImpl;

//This is for the creating device array
template<std::size_t Index, typename... ARGS>
class MakeDeviceObjectImpl<true,Index,ARGS...>{

public:
   //array variable type in this index
   typedef typename std::tuple_element<Index,std::tuple<ARGS...>>::type ArrayVariable_t;
   
   typedef typename std::remove_pointer<ArrayVariable_t>::type Variable_t;
   device_array<Variable_t>operator()(std::size_t arraySizes)const{
      return make_device_array<Variable_t>(arraySizes);
   }

};
  //now making device variables for non-array types.
template<std::size_t Index, typename... ARGS>   
 class MakeDeviceObjectImpl<false,Index,ARGS...>{
   public:
     typedef typename std::tuple_element<Index,std::tuple<ARGS... >>::type Variable_t;
     
     static_assert(std::is_trivial<Variable_t>::value==true, "Recieved a non-trivial type");
     
     Variable_t operator()(std::size_t )const{return Variable_t();}
     
 };
    
   //Now the actual function that prepares device variables
    template<std::size_t Index, typename... ARGS>
    class MakeDeviceTuple{
        static_assert(Index<sizeof... (ARGS),"Invalid index received");
    public:
        typedef typename std::tuple_element<Index,std::tuple<ARGS...>>::type Variable_t;
        
        auto operator()(std::size_t arraySizes)const{
        //previous element of the tuple
        auto prev = MakeDeviceTuple<Index-1,ARGS...>()(arraySizes);
         
         //current element of the tuple
        auto current = std::make_tuple(MakeDeviceObjectImpl<std::is_pointer<Variable_t>::value,Index,ARGS...>()
                       (arraySizes));
                       
        return std::tuple_cat(std::move(prev),std::move(current));
        }
        
    };
   
    template<typename T>
    struct DeviceVariableImpl{
       static_assert(std::is_trivial<T>::value==true,
                     "Arrays must have trivial types elements");
    
        typedef T type;
        
    };
    
    template<typename T>
    struct DeviceVariableImpl<T* >{
       static_assert(std::is_trivial<T>::value==true,
                     "Arrays must have trivial type elements");
        
        typedef typename ::device_array<T> type;
    };
    

    /*   
    template<typename... ARGS >
    struct TaskDeviceVariables{
      typedef typename 
      std::tuple<typename ::DeviceVariableImpl<ARGS>::type... > type;
    
    };
*/
    
}


class Functor3 {
public: 
   HOST_AND_DEVICE
   void operator()( std::size_t i, uint32_t* array1 ) {

      array1[ i ] *= 12;

   }
}; // class Functor3



__global__ void testvector(uint32_t* in,uint32_t* out, int _size){
    for(int id = blockIdx.x*blockDim.x+threadIdx.x;
        id<_size;
        id+= blockDim.x*gridDim.x){
        
        out[id] = in[id]+10;
        
        }

}


int main(){
    
    std::string f_name = "dune_file.h5";
    int ch_id = 2500;
    std::string apa_name = "APA01";
    auto handler = OffloadIntoHost(f_name);
    auto apa_id = handler.GetAPAHandle(apa_name);
    //now get the header info 
    ReadRawDataHDF5::DuneRawDataHeader c_header[1];
    handler.ReturnDataHeader( apa_id,ch_id, c_header);
    CudaVector<ReadRawDataHDF5::DuneRawDataHeader> cu_header=
        handler.OffloadHeaderIntoCudaArray(c_header);
    
    auto dat_size = c_header[0].Nadc_;
    CudaVector<uint32_t> cu_data = handler.OffloadDataIntoCudaArray(apa_id, c_header[0].chan_, c_header[0].Nadc_);
    
    auto dev_vector = cu_data.ReturnDeviceVector();
    size_t u_size = sizeof(uint32_t);
    
    uint32_t* ptr_trial;
    cudaMallocManaged((void**)&ptr_trial,dat_size*u_size);
    ptr_trial = thrust::raw_pointer_cast(dev_vector.data());

    
  //  uint32_t *ptr_out;
    cudaError_t ret;
    
    // ret = cudaMallocManaged((void**)&ptr_out,dat_size*u_size);
    uint32_t*  ptr_out = Device::AllocateMemory<uint32_t>(dat_size);
     if (ret != cudaSuccess) {
    std::cout << cudaGetErrorString(ret) << std::endl;
    return 1;
     }
     

    cudaStream_t stream = nullptr;
    
    if(Info::instance().nDevices()>0){
       CUDA_EXP_CHECK(cudaStreamCreate(&stream));   
    }
    
    testvector<<<1,dat_size>>>(ptr_trial,ptr_out,dat_size);
   uint32_t *host_out =
    Device::CopyArrayFromDevice(stream,ptr_out,dat_size);

    for(int i = 0;i<dat_size;i++)std::cout<<host_out[i]<<" "<<cu_data[i]<<std::endl;    
    
    return 1;   
    
}