// c++ -std=c++14 -I../src -L../src demo_taskmanager.cpp -lngs_core

#include <ngs_core.hpp>
using namespace ngstd;


int main ()
{
  TaskManager::SetNumThreads(4);

  PajeTrace::SetMaxTracefileSize(1000*1000);
  TaskManager::SetPajeTrace(true);

  int numthreads = EnterTaskManager();

  for (size_t n = 10; n <= 10000; n *= 10)
    {
      Array<double> res(n);
      
      for (size_t i = 0; i < 5; i++)
        
        ParallelFor (n, [&res] (size_t i)
                     {
                       double val = 1+1e-5*i;
                       double prod = 1;
                       for (size_t j = 0; j < 100; j++)
                         prod *= val;
                       res[i] = prod;
                     });
    }
  
  ExitTaskManager(numthreads);
}
