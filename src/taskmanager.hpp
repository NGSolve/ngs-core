#ifndef FILE_TASKMANAGER
#define FILE_TASKMANAGER

/*********************************************************************/
/* File:   taskmanager.hpp                                           */
/* Author: M. Hochsterger, J. Schoeberl                              */
/* Date:   10. Mar. 2015                                             */
/*********************************************************************/




namespace ngstd
{

  class TaskInfo
  {
  public:
    int task_nr;
    int ntasks;

    int thread_nr;
    int nthreads;

    int node_nr;
    int nnodes;
  };

  
  class TaskManager
  {

    class NodeData
    {
    public:
      atomic<int> start_cnt;
      atomic<int> complete_cnt;
      atomic<int> participate;

      NodeData() : start_cnt(0), participate(0) { ; }
    };
    
    function<void(TaskInfo&)> func;
    volatile int ntasks;

    atomic<int> jobnr;

    atomic<int> complete[8];   // max nodes
    atomic<int> done;

    NodeData *nodedata[8];

    int num_nodes;
    
  public:
    
    TaskManager();

    void CreateJob (function<void(TaskInfo&)> afunc, 
                    int antasks = omp_get_max_threads());

    void Done()
    {
      done = true;
    }

    void Loop();
  };








  extern TaskManager * task_manager;
  
  void RunWithTaskManager (function<void()> alg);
}



#endif