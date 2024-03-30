#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <cmath>

struct Answer {
    double x;
    double expected_value;
};

bool checkAnswer(const Answer& answer, const std::string& func_name) 
{
    if (func_name == "sin") 
    {
        return std::abs(std::sin(answer.x) - answer.expected_value) < 0.001; // Проверка для sin(x)
    } 
    else if (func_name == "sqrt") 
    {
        return std::abs(std::sqrt(answer.x) - answer.expected_value) < 0.001; // Проверка для sqrt(x)
    } 
    else if (func_name == "pow") 
    {
        return std::abs(std::pow(answer.x, 2.0) - answer.expected_value) < 0.001; // Проверка для x^2
    }
    return false;
}

int main() {
    std::ifstream file("answer.txt");
    if (!file.is_open()) {
        std::cerr << "Error opening file." << std::endl;
        return 1;
    }

    int all_ans = 0;
    int right_ans = 0;

    std::string line;
    while (std::getline(file, line)) 
    {
        std::istringstream iss(line);
        std::string func_name;
        Answer answer;

        if (iss >> func_name >> answer.x >> answer.expected_value) 
        {
            if (checkAnswer(answer, func_name)) 
            {
                right_ans++;
            }
            all_ans++;
        } 
        else 
        {
            std::cerr << "Error reading line from file." << std::endl;
        }
    }

    file.close();

    std::cout << "Count of answers: " << all_ans << std::endl;
    std::cout << "Right answers: " << right_ans << std::endl;

    return 0;
}