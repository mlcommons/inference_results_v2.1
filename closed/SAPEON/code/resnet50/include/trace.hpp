#pragma once
#include <string>
#include <vector>
#include <mutex>
#include <atomic>
#include <thread>
namespace sapeon_runtime{
    class Trace{
        public:
            Trace(const std::string& filename);
            ~Trace();

        class Thread{
            public:
            Thread(const std::string& name);
            ~Thread();
            class Work{
                public:
                    Work(const std::string& name, const int id);
                    ~Work();
                    void Start();
                    void End();
                private:
                    const std::string name_;
                    int id_;                    
                    const std::string kCategoryName = "background";
            };             
            Work& AddWork(const std::string& name);          
            private:
            const std::string name_;
            int id_;
            std::vector<Work*> works_;
            std::mutex works_m_;
            #if defined(TRACE_ENABLE)
            #else            
            Work* dummy_work_ = nullptr;
            #endif
        };

        Thread& AddThread(const std::string& name);
        private:
            void init();
            const std::string trace_filename_;
            std::vector<Thread*> threads_;
            std::mutex threads_m_;
            std::atomic<bool> run_;
            std::thread *flush_thread_ = nullptr;
            #if defined(TRACE_ENABLE)
            #else
            Thread *dummy_thread_ = nullptr;
            #endif
    };
    extern Trace trace_log;
}