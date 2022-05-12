#include "OffloadKernelTasks.h"
#include "tbb/task_scheduler_init.h"
#include "TaskPtr.h"
#include "KernelStatus.h"

template<class FUNCTOR, typename... ARGS>
std::unique_ptr<OffloadKernelTasks<FUNCTOR, ARGS... >>
  make_ArrayKernelTask(TaskPtr_t postExecTask,
                     std::size_t arraySizes, ARGS... args );


template<class FUNCTOR, typename... ARGS>
std::unique_ptr<OffloadKernelTasks<FUNCTOR, ARGS... >>
    make_ArrayKernelTask(KernelStatus& status, 
                         std::size_t arraySizes, ARGS... args );



