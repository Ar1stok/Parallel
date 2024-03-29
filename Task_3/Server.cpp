#include <iostream>
#include <fstream>
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
class Server {
public:
    void start() 
    {
        stoken_ = false;
        server_thread_ = std::thread(&Server::server_thread, this);
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
        while (results_.find(id) == results_.end()) {} // Ожидание результата
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


template <typename T>
class Client {
public:
    void run_client(Server<T>& server, std::function<T(T)> task) 
    {
        for (int i = 0; i < 10000; ++i)
        {
            int id = server.add_task(task);
            task_ids_.push_back(id);
        }
        
    }

    std::vector<T> client_to_result(Server<T>& server) 
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
    Server<double> server; 
    server.start();

    auto begin = std::chrono::steady_clock::now();

    Client<double> client1;
    Client<double> client2;
    Client<double> client3;

    std::thread t1 ([&]() { client1.run_client(server, fun_sin<double>); });
    std::thread t2 ([&]() { client2.run_client(server, fun_sqrt<double>); });
    std::thread t3 ([&]() { client3.run_client(server, fun_pow<double>); });

    t1.join();
    t2.join();
    t3.join();

    std::vector<double> ans_1;
    std::vector<double> ans_2;
    std::vector<double> ans_3; 
    
    std::thread t4 ([&]() { ans_1 = client1.client_to_result(server); });
    std::thread t5 ([&]() { ans_2 = client2.client_to_result(server); });
    std::thread t6 ([&]() { ans_3 = client3.client_to_result(server); });
    
    t4.join();
    t5.join();
    t6.join();

    auto end = std::chrono::steady_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);

    std::ofstream file;
	file.open("answers.txt");
    
    file << "The time: " << elapsed_ms.count() << " ms\n" << "sin_ans:" << std::endl;
	for (double n : ans_1)
    {
        file << n << "\n";
    }
    file << std::endl;

    file << "sqrt_ans:" << std::endl;
	for (double n : ans_2)
    {
        file << n << "\n";
    }
    file << std::endl;

    file << "pow_ans:" << std::endl;
	for (double n : ans_3)
    {
        file << n << "\n";
    }
    file << std::endl;

	file.close(); 

    server.stop();

    return 0;
}