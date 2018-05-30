namespace integrators
{
    // TODO make 3 a compiler template argument
	template <typename D, typename U>
	struct Korobov3 {
#ifdef __CUDACC__
		__host__ __device__
#endif
		void operator()(D* x, D& wgt, const U dim) const
		{
			// Korobov r = 3
            for (U sDim = 0; sDim < dim; sDim++)
			{
				wgt *= x[sDim] * x[sDim] * x[sDim] * 140.*(1. - x[sDim])*(1. - x[sDim])*(1. - x[sDim]);
				x[sDim] = x[sDim] * x[sDim] * x[sDim] * x[sDim] * (35. + x[sDim] * (-84. + x[sDim] * (70. + x[sDim] * (-20.))));
				// loss of precision can cause x > 1., must keep in x \elem [0,1]
				if (x[sDim] > 1.)
					x[sDim] = 1.;
			}
		}
	};

    template<typename D, typename U>
    struct Trivial
    {
#ifdef __CUDACC__
        __host__ __device__
#endif
        void operator()(D* x, D& wgt, const U dim) const {}
    };
    
};
