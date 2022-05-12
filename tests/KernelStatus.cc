#include "KernelStatus.h"

KernelStatus::KernelStatus():m_status(1),m_mutex(){
    
    m_mutex.lock();
 
}

void KernelStatus::finished(bool status){
    
    m_status = status;
    m_mutex.unlock();
    return;
}

bool KernelStatus::wait(){
    std::lock_guard<std::mutex>lock(m_mutex);
    
    return m_status;
    
}