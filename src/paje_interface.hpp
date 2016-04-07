#ifndef PAJE_INTERFACE_HPP_INCL__
#define PAJE_INTERFACE_HPP_INCL__

#include <limits>
#include <vector>

namespace ngstd
{
  extern NGS_DLL_HEADER class PajeTrace *trace;
  class PajeTrace
    {
      friend class TraceDisabler;

      static size_t max_tracefile_size;
      static bool trace_thread_counter;
      static bool trace_threads;

      bool tracing_enabled;
      double start_time;
      int nthreads;

    public:

      // Approximate number of events to trace. Tracing will
      // be stopped if any thread reaches this number of events
      unsigned int max_num_events_per_thread;

      static void SetTraceThreads( bool atrace_threads )
        {
          trace_threads = atrace_threads;
        }

      static void SetTraceThreadCounter( bool trace_threads )
        {
          trace_thread_counter = trace_threads;
        }

      static void SetMaxTracefileSize( size_t max_size )
        {
          max_tracefile_size = max_size;
        }

      std::string tracefile_name;

      struct Job
        {
          int job_id;
          const std::type_info *type;
          double start_time;
          double stop_time;
        };

      struct Task
        {
          int thread_id;

          int id;
          int id_type;

          int additional_value;

          double start_time;
          double stop_time;

          static constexpr int ID_NONE = -1;
          static constexpr int ID_JOB = 1;
          static constexpr int ID_TIMER = 2;
        };

      struct TimerEvent
        {
          int timer_id;
          double time;
          bool is_start;
          int thread_id;

          bool operator < (const TimerEvent & other) const { return time < other.time; }
        };

      struct ThreadLink
        {
          int thread_id;
          int key;
          double time;
          bool is_start;
          bool operator < (const ThreadLink & other) const { return time < other.time; }
        };

      std::vector<std::vector<Task> > tasks;
      std::vector<Job> jobs;
      std::vector<TimerEvent> timer_events;
      std::vector<std::vector<ThreadLink> > links;

      double GetTime()
        {
          return WallTime() - start_time;
        }

    public:
      NGS_DLL_HEADER void StopTracing();

      PajeTrace(int anthreads, std::string aname = "");
      ~PajeTrace();

      void StartTimer(int timer_id)
        {
          if(!tracing_enabled) return;
          if(timer_events.size() == max_num_events_per_thread)
            StopTracing();
          timer_events.push_back(TimerEvent{timer_id, GetTime(), true});
        }

      void StopTimer(int timer_id)
        {
          if(!tracing_enabled) return;
          if(timer_events.size() == max_num_events_per_thread)
            StopTracing();
          timer_events.push_back(TimerEvent{timer_id, GetTime(), false});
        }

      int StartTask(int thread_id, int id, int id_type = Task::ID_NONE, int additional_value = -1)
        {
          if(!tracing_enabled) return -1;
          if(!trace_threads && !trace_thread_counter) return -1;
          if(tasks[thread_id].size() == max_num_events_per_thread)
            StopTracing();
          int task_num = tasks[thread_id].size();
          tasks[thread_id].push_back( Task{thread_id, id, id_type, additional_value, GetTime(), 0.0} );
          return task_num;
        }

      void StopTask(int thread_id, int task_num)
        {
          if(!trace_threads && !trace_thread_counter) return;
          if(task_num>=0)
            tasks[thread_id][task_num].stop_time = GetTime();
        }

      void SetTask(int thread_id, int task_num, int additional_value) {
          if(!trace_threads && !trace_thread_counter) return;
          if(task_num>=0)
            tasks[thread_id][task_num].additional_value = additional_value;
      }

      void StartJob(int job_id, const std::type_info & type)
        {
          if(!tracing_enabled) return;
          if(jobs.size() == max_num_events_per_thread)
            StopTracing();
          jobs.push_back( Job{job_id, &type, GetTime(), 0.0 } );
        }

      void StopJob()
        {
          if(tracing_enabled)
            jobs.back().stop_time = GetTime();
        }

      void StartLink(int thread_id, int key)
        {
          if(!tracing_enabled) return;
          if(links[thread_id].size() == max_num_events_per_thread)
            StopTracing();
          links[thread_id].push_back( ThreadLink{thread_id, key, GetTime(), true} );
        }

      void StopLink(int thread_id, int key)
        {
          if(!tracing_enabled) return;
          if(links[thread_id].size() == max_num_events_per_thread)
            StopTracing();
          links[thread_id].push_back( ThreadLink{thread_id, key, GetTime(), false} );
        }

      void Write( std::string filename );

    };

  class TraceDisabler
    {
      bool trace_thread_counter;
      bool trace_threads;

    public:
      TraceDisabler()
        {
          trace_thread_counter = PajeTrace::trace_thread_counter;
          PajeTrace::trace_thread_counter = false;
          trace_threads = PajeTrace::trace_threads;
          PajeTrace::trace_threads = false;
        }

      ~TraceDisabler()
        {
          PajeTrace::trace_thread_counter = trace_thread_counter;
          PajeTrace::trace_threads = trace_threads;
        }
    };
}

#endif // PAJE_INTERFACE_HPP_INCL__
