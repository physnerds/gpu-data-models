#include "TaskPtr.h"

namespace details{
 
    void TaskDeleter::operator()(tbb::task* task){
        // if null pointer, return 
        if(task==nullptr){
         return;
     }
        //Destroy the task..
        tbb::task::destroy(*task);
        
        // And then complain..
        throw std::runtime_error("Task was not properly scheduled");
        
        
    }
}

bool enqueue (TaskPtr_t postExecTask){
 
    if(!postExecTask){
     return 0;   
    }
    
    tbb:task::enqueue( *(postExecTask.release())
#if __TBB_TASK_PRIORITY
                      , tbb::priority_high
#endif
                      );
    return 1;
                    
}