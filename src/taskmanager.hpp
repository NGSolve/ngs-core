#ifndef FILE_TASKMANAGER
#define FILE_TASKMANAGER

/*********************************************************************/
/* File:   taskmanager.hpp                                           */
/* Author: M. Hochsterger, J. Schoeberl                              */
/* Date:   10. Mar. 2015                                             */
/*********************************************************************/

#include <atomic>
#include <thread>



namespace ngstd
{
  class PajeTrace;

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

  NGS_DLL_HEADER extern class TaskManager * task_manager;
  
  class TaskManager
  {
//     PajeTrace *trace;

    class NodeData
    {
    public:
      atomic<int> start_cnt;
      atomic<int> complete_cnt;
      atomic<int> participate;

      NodeData() : start_cnt(0), participate(0) { ; }
    };
    
    const function<void(TaskInfo&)> * func;
    atomic<int> ntasks;
    Exception * ex;

    int jobnr;

    atomic<int> complete[8];   // max nodes
    atomic<int> done;
    atomic<int> active_workers;

    int sleep_usecs;
    bool sleep;

    NodeData *nodedata[8];

    int num_nodes;
    int num_threads;
    static int max_threads;
#ifndef __clang__    
    static thread_local int thread_id;
#else
    static __thread int thread_id;
#endif
    
    static bool use_paje_trace;
  public:
    
    TaskManager();
    ~TaskManager();


    void StartWorkers();
    void StopWorkers();

    void SuspendWorkers(int asleep_usecs = 1000 )
      {
        sleep_usecs = asleep_usecs;
        sleep = true;
      }
    void ResumeWorkers() { sleep = false; }

    static void SetNumThreads(int amax_threads);
    static int GetMaxThreads() { return max_threads; }
    static int GetNumThreads() { return task_manager ? task_manager->num_threads : 1; }
    static int GetThreadId() { return task_manager ? task_manager->thread_id : 0; }
    int GetNumNodes() const { return num_nodes; }

    static void SetPajeTrace (bool use)  { use_paje_trace = use; }
    
    NGS_DLL_HEADER void CreateJob (const function<void(TaskInfo&)> & afunc, 
                    int antasks = task_manager->GetNumThreads());


    /*
    template <typename TFUNC>
    INLINE void ParallelFor (IntRange r, TFUNC f, int antasks = task_manager->GetNumThreads())
    {
      CreateJob 
        ([r, f] (TaskInfo & ti) 
         {
           auto myrange = r.Split (ti.task_nr, ti.ntasks);
           for (auto i : myrange) f(i);
         }, antasks);
    }
    */


    void Done() { done = true; }


    void Loop(int thread_num);
  };








  
  void RunWithTaskManager (function<void()> alg);

  // For Python context manager
  int  EnterTaskManager ();
  void ExitTaskManager (int num_threads);

  INLINE int TasksPerThread (int tpt)
  {
    return task_manager ? tpt*task_manager->GetNumThreads() : 1;
  }

  template <typename TR, typename TFUNC>
  INLINE void ParallelFor (T_Range<TR> r, TFUNC f, 
                           int antasks = task_manager ? task_manager->GetNumThreads() : 0)
  {
    if (task_manager)

      task_manager -> CreateJob 
        ([r, f] (TaskInfo & ti) 
         {
           auto myrange = r.Split (ti.task_nr, ti.ntasks);
           for (auto i : myrange) f(i);
         }, 
         antasks);

    else

      for (auto i : r) f(i);
  }


  template <typename TR, typename TFUNC>
  INLINE void ParallelForRange (T_Range<TR> r, TFUNC f, 
                                int antasks = task_manager ? task_manager->GetNumThreads() : 0)
  {
    if (task_manager)

      task_manager -> CreateJob 
        ([r, f] (TaskInfo & ti) 
         {
           auto myrange = r.Split (ti.task_nr, ti.ntasks);
           f(myrange);
         }, 
         antasks);

    else

      f(r);
  }



  
  
  
  /*
    Usage example:

    ShareLoop myloop(100);
    task_manager->CreateJob ([]()
    {
      for (int i : myloop)
        cout << "i = " << i << endl;
    });

  */
  
  class SharedLoop
  {
    atomic<int> cnt;
    IntRange r;

    
    class SharedIterator
    {
      atomic<int> & cnt;
      int myval;
      int endval;
    public:
      SharedIterator (atomic<int> & acnt, int aendval, bool begin_iterator) 
        : cnt (acnt)
      {
        endval = aendval;
        myval = begin_iterator ? cnt++ : endval;
        if (myval > endval) myval = endval;
      }
      
      SharedIterator & operator++ () 
      {
        myval = cnt++; 
        if (myval > endval) myval = endval;
        return *this; 
      }
      
      int operator* () const { return myval; }
      bool operator!= (const SharedIterator & it2) const { return myval != it2.myval; }
    };
    
    
  public:
    SharedLoop (IntRange ar) : r(ar) { cnt = r.begin(); }
    SharedIterator begin() { return SharedIterator (cnt, r.end(), true); }
    SharedIterator end()   { return SharedIterator (cnt, r.end(), false); }
  };






  class Partitioning
  {
    Array<int> part;
  public:
    Partitioning () { ; }

    template <typename T>
    Partitioning (const Array<T> & apart) { part = apart; }

    template <typename T>
    Partitioning & operator= (const Array<T> & apart) { part = apart; return *this; }



    template <typename TFUNC>
    void Calc (int n, TFUNC costs, int size = task_manager ? task_manager->GetNumThreads() : 1)
    {
      Array<size_t> prefix (n);

      size_t sum = 0;
      for (auto i : ngstd::Range(n))
        {
          sum += costs(i);
          prefix[i] = sum;
        }
      
      part.SetSize (size+1);
      part[0] = 0;

      for (int i = 1; i <= size; i++)
        part[i] = BinSearch (prefix, sum*i/size);      
    }
    
    int Size() const { return part.Size()-1; }
    T_Range<int> operator[] (int i) const { return ngstd::Range(part[i], part[i+1]); }
    T_Range<int> Range() const { return ngstd::Range(part[0], part[Size()]); }




  private:
    template <typename Tarray>
    int BinSearch(const Tarray & v, size_t i) {
      int n = v.Size();
      if (n == 0) return 0;
      
      int first = 0;
      int last = n-1;
      if(v[0]>i) return 0;
      if(v[n-1] <= i) return n;
      while(last-first>1) {
        int m = (first+last)/2;
        if(v[m]<i)
          first = m;
        else
          last = m;
      }
      return first;
    }
  };


  inline ostream & operator<< (ostream & ost, const Partitioning & part)
  {
    for (int i : Range(part.Size()))
      ost << part[i] << " ";
    return ost;
  }
  

  // tasks must be a multiple of part.size
  template <typename TFUNC>
  INLINE void ParallelFor (const Partitioning & part, TFUNC f, int tasks_per_thread = 1)
  {
    if (task_manager)
      {
        int ntasks = tasks_per_thread * task_manager->GetNumThreads();
        if (ntasks % part.Size() != 0)
          throw Exception ("tasks must be a multiple of part.size");

        task_manager -> CreateJob 
          ([&] (TaskInfo & ti) 
           {
             int tasks_per_part = ti.ntasks / part.Size();
             int mypart = ti.task_nr / tasks_per_part;
             int num_in_part = ti.task_nr % tasks_per_part;
             
             auto myrange = part[mypart].Split (num_in_part, tasks_per_part);
             for (auto i : myrange) f(i);
           }, ntasks);
      }
    else
      {
        for (auto i : part.Range())
          f(i);
      }
  }





  template <typename TFUNC>
  INLINE void ParallelForRange (const Partitioning & part, TFUNC f, int tasks_per_thread = 1)
  {
    if (task_manager)
      {
        int ntasks = tasks_per_thread * task_manager->GetNumThreads();
        if (ntasks % part.Size() != 0)
          throw Exception ("tasks must be a multiple of part.size");

        task_manager -> CreateJob 
          ([&] (TaskInfo & ti) 
           {
             int tasks_per_part = ti.ntasks / part.Size();
             int mypart = ti.task_nr / tasks_per_part;
             int num_in_part = ti.task_nr % tasks_per_part;
             
             auto myrange = part[mypart].Split (num_in_part, tasks_per_part);
             f(myrange);
           }, ntasks);
      }
    else
      {
        f(part.Range());
      }
  }





}



#endif
