#ifndef KERNELSTATUS_H
#define KERNELSTATUS_H

#include <mutex>

class KernelStatus{
    public:
      KernelStatus();
    
      void finished(bool status);
    
      bool wait();
    
    private:
      bool m_status;
      std::mutex m_mutex;    
    
};


#endif