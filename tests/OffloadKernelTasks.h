#ifndef OFFLOADKERNELTASKS_H
#define OFFLOADKERNELTASKS_H

#include "TaskPtr.h"
#include "KernelStatus.h"

#ifndef __CUDACC__
typedef void* cudaStream_t;
#endif 


//I might have to make it separate or just remove it otherwise.
class IKernelTask{
   
   public: 
      virtual ~IKernelTask() = default;
      
      enum KernelExecMode {
      
      //kernel was executed synchronously on the cpu
      Synchronous = 0,
      
      //kernel was executed asynchronously on the cuda device
      Asynchronous = 1
      
      };
      
      virtual bool execute (cudaStream_t stream)=0;
      
      virtual bool finished (bool code, KernelExecMode mode) = 0;


};


template<class FUNCTOR, typename... ARGS>
class OffloadKernelTasks : public IKernelTask {
    static_assert (sizeof... (ARGS)>0,"At least one functor argument must be provided");
     //forward declare some classes.
    class TaskDeleter;

    
    public:
    //Constructor for non-blocking execution
      OffloadKernelTasks(TaskPtr_t postExecTask,
                          std::size_t arraySizs, ARGS... args );
     
     //Constructor for blocking execution
      OffloadKernelTasks(KernelStatus &status,
                          std::size_t arraySizes, ARGS... args);
      
      //I will go on the limb and say that it might be needed at some point.
      
     virtual bool execute(cudaStream_t stream) override;
      
     virtual bool finished(bool code, KernelExecMode mode) override;
      
    private:
      TaskPtr_t m_postExecTask;
      //Size of te arrays that we feed from kernel/functions
      std::size_t m_arraySizes;
      
      //arguments that come with the class object
      std::tuple<ARGS... >m_args;
      
      //need to make sure the task is finishing.
      KernelStatus* m_status;
      
      bool m_ranOnDevice;
          
};

#endif
