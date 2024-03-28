#include <iostream>
#include <thread>
#include <list>
#include <mutex>
#include <queue>
#include <future>
#include <functional>
#include <condition_variable>
#include <random>
#include <cmath> 
#include <unordered_map>

template<typename T>
T fun_sin(T arg) 
{
    return std::sin(arg);
}

template<typename T>
T fun_sqrt(T arg) 
{
    return std::sqrt(arg);
}

template<typename T>
T fun_pow(T arg) 
{
    return std::pow(arg, 2.0);
}

template <typename T>
class TaskServer {
public:
    void start() 
    {
        stoken_ = false;
        server_thread_ = std::thread(&TaskServer::server_thread, this);
    }

    void stop() 
    {
        {
            std::unique_lock<std::mutex> lock(mutex_);
            stoken_ = true;
        }
        server_thread_.join();
    }

    size_t add_task(std::function<T(T)> task) 
    {
        std::unique_lock<std::mutex> lock(mutex_);
        static std::default_random_engine generator;
        static std::uniform_real_distribution<T> distribution(1.0, 10.0);
        tasks_.push({ ++next_id_, std::async(std::launch::deferred, task, distribution(generator)) });
        return tasks_.back().first;
    }

    T request_result(size_t id) 
    {
        std::unique_lock<std::mutex> lock(mutex_);
        while (results_.find(id) == results_.end()) {} 
        T result = results_[id];
        results_.erase(id);
        return result;
    }

private:
    std::mutex mutex_;
    std::thread server_thread_;
    bool stoken_ = false;
    size_t next_id_ = 1;
    std::queue<std::pair<size_t, std::future<T>>> tasks_;
    std::unordered_map<size_t, T> results_;

    void server_thread()
    {
        while (true) 
        {
            std::unique_lock<std::mutex> lock(mutex_);
            if (tasks_.empty() && stoken_) 
            {
                break;
            }
            if (!tasks_.empty()) 
            {
                auto task = std::move(tasks_.front());
                tasks_.pop();
                results_[task.first] = task.second.get();
            }
        }
    }
};

// Класс клиента задач
template <typename T>
class TaskClient {
public:
    void run_client(TaskServer<T>& server, std::function<T(T)> task) 
    {
        for (int i = 0; i < 10; ++i)
        {
            int id = server.add_task(task);
            task_ids_.push_back(id);
        }
        
    }

    std::vector<T> client_to_result(TaskServer<T>& server) 
    {
        std::vector<T> results;
        for (int id : task_ids_) 
        {
            T result = server.request_result(id);
            results.push_back(result);
        }
        return results;
    }

private:
    std::vector<int> task_ids_;
};


int main() {
    TaskServer<double> server; 
    server.start();

    TaskClient<double> client1;
    TaskClient<double> client2;
    TaskClient<double> client3;
    
    client1.run_client(server, fun_sin<double>);
    client2.run_client(server, fun_sqrt<double>);
    client3.run_client(server, fun_pow<double>);

    std::vector<double> ans_1;
    std::vector<double> ans_2;
    std::vector<double> ans_3;
    
    std::thread t1 ([&]() {ans_1 = client1.client_to_result(server);});
    std::thread t2 ([&]() {ans_2 = client2.client_to_result(server);});
    std::thread t3 ([&]() {ans_3 = client3.client_to_result(server);});
    t1.join();
    t2.join();
    t3.join();

    std::cout << "ans_1: ";
    for (double n : ans_1)
        std::cout << n << ", ";
    std::cout << "\n";

    std::cout << "ans_2: ";
    for (double n : ans_2)
        std::cout << n << ", ";
    std::cout << "\n";

    std::cout << "ans_3: ";
    for (double n : ans_3)
        std::cout << n << ", ";
    std::cout << "\n";

    server.stop();

    return 0;
}