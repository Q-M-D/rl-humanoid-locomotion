/*!
 * @file PeriodicTask.cpp
 * @brief Implementation of a periodic function running in a separate thread.
 * Periodic tasks have a task manager, which measure how long they take to run.
 */


#include <sys/timerfd.h>



#include <unistd.h>
#include <cmath>

#include "PeriodicTask.h"
#include "Timer.h"
#include "Utilities_print.h"


bool set_cpu_dma_latency(int max_latency_us)
{
    int fd;

    fd = open("/dev/cpu_dma_latency", O_WRONLY);
    if (fd < 0)
    {
        perror("open /dev/cpu_dma_latency");
        return false;
    }
    if (write(fd, &max_latency_us, sizeof(max_latency_us)) !=
        sizeof(max_latency_us))
    {
        perror("write to /dev/cpu_dma_latency");
        return false;
    }
    return true;
}

/*!
 * Construct a new task within a TaskManager
 * @param taskManager : Parent task manager
 * @param period : how often to run
 * @param name : name of task
 */
PeriodicTask::PeriodicTask(PeriodicTaskManager* taskManager, float period,
               std::string name, int priority, int cpu)
        : _period(period), _name(name), _priority(priority), _cpu(cpu){
    taskManager->addTask(this);
}

/*!
 * Begin running task
 */
void PeriodicTask::start() {
    if (_running) {
        printf("[PeriodicTask] Tried to start %s but it was already running!\n",
               _name.c_str());
        return;
    }
    init();
    _running = true;
    _thread = std::thread(&PeriodicTask::loopFunction, this);
}

/*!
 * Stop running task
 */
void PeriodicTask::stop() {
    if (!_running) {
        printf("[PeriodicTask] Tried to stop %s but it wasn't running!\n",
               _name.c_str());
        return;
    }
    _running = false;
    printf("[PeriodicTask] Waiting for %s to stop...\n", _name.c_str());
    _thread.join();
    printf("[PeriodicTask] Done!\n");
    cleanup();
}

/*!
 * If max period is more than 30% over desired period, it is slow
 */
bool PeriodicTask::isSlow() {
    return _maxPeriod > _period * 1.3f || _maxRuntime > _period;
}

/*!
 * Reset max statistics
 */
void PeriodicTask::clearMax() {
    _maxPeriod = 0;
    _maxRuntime = 0;
}

/*!
 * Print the status of this task in the table format
 */
void PeriodicTask::printStatus() {
    if (!_running) return;
    if (isSlow()) {
        printf_color(PrintColor::Red, "|%-20s|%6.4f|%6.4f|%6.4f|%6.4f|%6.4f\n",
                     _name.c_str(), _lastRuntime, _maxRuntime, _period,
                     _lastPeriodTime, _maxPeriod);
    } else {
        printf("|%-20s|%6.4f|%6.4f|%6.4f|%6.4f|%6.4f\n", _name.c_str(),
               _lastRuntime, _maxRuntime, _period, _lastPeriodTime, _maxPeriod);
    }
}

/*!
 * Call the task in a timed loop.  Uses a timerfd
 */
void PeriodicTask::loopFunction() {

    sched_param sch;
    sch.sched_priority = _priority;
    if (pthread_setschedparam(pthread_self(), SCHED_FIFO, &sch)) {
        std::cout << "Failed to setschedparam: " << std::strerror(errno) << '\n';
    }

    if(not set_cpu_dma_latency(0)){
        std::cout << "Failed to cpu dma latency"<<std::endl;
    }

    auto timerFd = timerfd_create(CLOCK_MONOTONIC, 0);

    // set cpu affinity
    pthread_t thread_ = pthread_self();
    cpu_set_t cpu_set;

    CPU_ZERO(&cpu_set);
    CPU_SET(_cpu, &cpu_set);
    int s = pthread_setaffinity_np(thread_, sizeof(cpu_set_t), &cpu_set);
    if(s != 0){
        printf("pthread set affinity error\n");
    }
    s = pthread_getaffinity_np(thread_, sizeof(cpu_set_t), &cpu_set);
    if( s != 0){
        printf("pthread get affinity error\n");
    }
    for (int i = 0; i < CPU_SETSIZE; ++i) {
        if(CPU_ISSET(i, &cpu_set)){
            printf("cpu %d\n", i);
        }
    }

    int seconds = (int) _period;
    int nanoseconds = (int) (1e9 * std::fmod(_period, 1.f));

    Timer t;

    itimerspec timerSpec;
    timerSpec.it_interval.tv_sec = seconds;
    timerSpec.it_value.tv_sec = seconds;
    timerSpec.it_value.tv_nsec = nanoseconds;
    timerSpec.it_interval.tv_nsec = nanoseconds;

    timerfd_settime(timerFd, 0, &timerSpec, nullptr);

    unsigned long long missed = 0;

    printf("[PeriodicTask] Start %s (%d s, %d ns)\n", _name.c_str(), seconds,
           nanoseconds);
    while (_running) {
        _lastPeriodTime = (float) t.getSeconds();
        t.start();
        run();
        _lastRuntime = (float) t.getSeconds();

        int m = read(timerFd, &missed, sizeof(missed));
        (void) m;

        _maxPeriod = std::max(_maxPeriod, _lastPeriodTime);
        _maxRuntime = std::max(_maxRuntime, _lastRuntime);
    }
    printf("[PeriodicTask] %s has stopped!\n", _name.c_str());
}

PeriodicTaskManager::~PeriodicTaskManager() {}

/*!
 * Add a new task to a task manager
 */
void PeriodicTaskManager::addTask(PeriodicTask *task) {
    _tasks.push_back(task);
}

/*!
 * Print the status of all tasks and rest max statistics
 */
void PeriodicTaskManager::printStatus() {
    printf("\n----------------------------TASKS----------------------------\n");
    printf("|%-20s|%-6s|%-6s|%-6s|%-6s|%-6s\n", "name", "rt", "rt-max", "T-des",
           "T-act", "T-max");
    printf("-----------------------------------------------------------\n");
    for (auto &task: _tasks) {
        task->printStatus();
        task->clearMax();
    }
    printf("-------------------------------------------------------------\n\n");
}

/*!
 * Print only the slow tasks
 */
void PeriodicTaskManager::printStatusOfSlowTasks() {
    for (auto &task: _tasks) {
        if (task->isSlow()) {
            task->printStatus();
            task->clearMax();
        }
    }
}

/*!
 * Stop all tasks
 */
void PeriodicTaskManager::stopAll() {
    for (auto &task: _tasks) {
        task->stop();
    }
}
