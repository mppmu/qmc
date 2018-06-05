#include <functional> // reference_wrapper
#include <ostream> // ostream
#include <chrono> // chrono
#include <string> // to_string

namespace integrators
{
    struct Logger : public std::reference_wrapper<std::ostream>
    {
        //        using std::reference_wrapper<std::ostream>::reference_wrapper;

        bool display_timing;
        std::chrono::steady_clock::time_point reference_time; // time logger was initialised

        template<typename T> std::ostream& operator<<(T arg) const
        {
            std::string time_string = "";
            if( display_timing )
            {
                std::chrono::steady_clock::time_point now_time = std::chrono::steady_clock::now();
                time_string = "[" + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(now_time - reference_time).count()) + " ms] ";
            }
            return this->get() << time_string << arg;
        }
        std::ostream& operator<<(std::ostream& (*arg)(std::ostream&)) const { return this->get() << arg; }

        Logger(std::ostream& stream) : reference_wrapper(std::ref(stream)), display_timing(true), reference_time(std::chrono::steady_clock::now()) {}

    };
};

