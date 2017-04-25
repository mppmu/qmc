#pragma once
#ifndef QMC_H
#define QMC_H

#include <vector>
#include <map>
#include <random> // mt19937_64
#include <functional> // function

namespace integrators {

    template <typename T>
    struct result {
        T integral;
        T error;
    };

    template <typename T, typename D, typename U = unsigned long long int, typename G = std::mt19937_64>
    class Qmc {

    protected:

        virtual void integralTransform(std::vector<D>& x, D& wgt, const U dim) const;

    private:

        std::uniform_real_distribution<D> uniformDistribution = std::uniform_real_distribution<D>(0,1);

        template <typename R>
        R mul_mod(U a, U b, U k) const;

        void initg();
        void initzn(std::vector<U>& z, U& n, const U dim) const;
        void initd(std::vector<D>& d, const U dim);
        void initr(std::vector<T>& r, const U block) const;
        void initzdrn(std::vector<U>& z, std::vector<D>& d, std::vector<T>& r, U& n, U& blocks, U& block, const U dim);
        void compute(const U blocks, const U block, const U k, const U i, const std::vector<U>& z, const std::vector<D>& d, std::vector<T>& r, const U n, const std::function<T(D[])>& func, const U dim) const;
        result<T> reduce(const std::vector<T>& r, const U n, const U block) const;

    public:

        G randomGenerator;

        U getN() const;
        U minN;
        U m;
        U blockSize;
        std::map< U, std::vector<U> > generatingVectors;

        result<T> integrate(const std::function<T(D[])>& func, const U dim);

        Qmc();

        virtual ~Qmc() {}

    };
    
};

#endif
