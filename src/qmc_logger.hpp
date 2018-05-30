namespace integrators
{
    struct Logger : public std::reference_wrapper<std::ostream>
    {
        using std::reference_wrapper<std::ostream>::reference_wrapper;
        template<typename T> std::ostream& operator<<(T arg) const { return this->get() << arg; }
        std::ostream& operator<<(std::ostream& (*arg)(std::ostream&)) const { return this->get() << arg; }
    };
};
