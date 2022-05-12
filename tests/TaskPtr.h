#ifndef TASKPTR_H
#define TASKPTR_H

#include <tbb/task_arena.h>

#include <memory>

namespace details{
    
    class TaskDeleter{
     public:
        
        void operator()(tbb::task_arena *task);
    };
 
    
}

typedef std::unique_ptr<tbb::task,details::TaskDeleter>TaskPtr_t;

bool enqueue(TaskPtr_t postExecTask);

#endif