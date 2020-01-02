// Copyright 2019 Xanadu Quantum Technologies Inc.


// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
/**
 * @file
 * Contains functions for calculating the multidimensional
 * Hermite polynomials, used for computation of batched hafnians.
 */

#pragma once
#include <stdafx.h>
#include <assert.h>

typedef unsigned long long int ullint;


/**
 * Returns the index of the one dimensional flattened vector corresponding to the multidimensional tensor
 *
 * @param pos
 * @param resolution
 *
 * @return index on flattened vector
 */
ullint vec2index(std::vector<int> &pos, int resolution) {
    int dim = pos.size();
    ullint nextCoordinate = 0;

    nextCoordinate = pos[0]-1;
    for(int ii = 0; ii < dim-1; ii++) {
        nextCoordinate = nextCoordinate*resolution + (pos[ii+1]-1);
    }

    return nextCoordinate;

}

namespace libwalrus {

/**
 * Returns the multidimensional Hermite polynomials \f$H_k^{(R)}(y)\f$.
 *
 * This implementation is based on the MATLAB code available at
 * https://github.com/clementsw/gaussian-optics
 *
 * @param R a flattened vector of size \f$n^2\f$, representing a
 *       \f$n\times n\f$ symmetric matrix.
 * @param y a flattened vector of size \f$n\f$.
 * @param resolution highest number of photons to be resolved.
 *
 */
template <typename T>
inline std::vector<T> hermite_multidimensional_cpp(std::vector<T> &R, std::vector<T> &y, int &resolution) {
    int dim = std::sqrt(static_cast<double>(R.size()));

    ullint Hdim = pow(resolution, dim);
    std::vector<T> H(Hdim, 0);


    H[0] = 1;

    std::vector<int> nextPos(dim, 1);
    std::vector<int> jumpFrom(dim, 1);
    std::vector<int> ek(dim, 0);
    std::vector<double> factors(resolution+1, 0);
    int jump = 0;

    for (ullint jj = 0; jj < Hdim-1; jj++) {

        if (jump > 0) {
            jumpFrom[jump] += 1;
            jump = 0;
        }


        for (int ii = 0; ii < dim; ii++) {
            std::vector<int> forwardStep(dim, 0);
            forwardStep[ii] = 1;

            if ( forwardStep[ii] + nextPos[ii] > resolution) {
                nextPos[ii] = 1;
                jumpFrom[ii] = 1;
                jump = ii+1;
            }
            else {
                jumpFrom[ii] = nextPos[ii];
                nextPos[ii] = nextPos[ii] + 1;
                break;
            }
        }

        for (int ii = 0; ii < dim; ii++)
            ek[ii] = nextPos[ii] - jumpFrom[ii];

        int k = 0;
        for(; k < static_cast<int>(ek.size()); k++) {
            if(ek[k]) break;
        }

        ullint nextCoordinate = vec2index(nextPos, resolution);
        ullint fromCoordinate = vec2index(jumpFrom, resolution);


        H[nextCoordinate] = H[nextCoordinate] + y[k];
        H[nextCoordinate] = H[nextCoordinate] * H[fromCoordinate];

        std::vector<int> tmpjump(dim, 0);

        for (int ii = 0; ii < dim; ii++) {
            if (jumpFrom[ii] > 1) {
                std::vector<int> prevJump(dim, 0);
                prevJump[ii] = 1;
                std::transform(jumpFrom.begin(), jumpFrom.end(), prevJump.begin(), tmpjump.begin(), std::minus<int>());
                ullint prevCoordinate = vec2index(tmpjump, resolution);
                H[nextCoordinate] = H[nextCoordinate] - (static_cast<T>(jumpFrom[ii]-1))*(R[dim*k+ii])*H[prevCoordinate];

            }
        }

    }
    return H;

}

/**
 * Returns the normalized multidimensional Hermite polynomials \f$\tilde{H}_k^{(R)}(y)\f$.
 *
 * This implementation is based on the MATLAB code available at
 * https://github.com/clementsw/gaussian-optics
 *
 * @param R a flattened vector of size \f$n^2\f$, representing a
 *       \f$n\times n\f$ symmetric matrix.
 * @param y a flattened vector of size \f$n\f$.
 * @param resolution highest number of photons to be resolved.
 *
 */
template <typename T>
inline std::vector<T> renorm_hermite_multidimensional_cpp(std::vector<T> &R, std::vector<T> &y, int &resolution) {
    int dim = std::sqrt(static_cast<double>(R.size()));

    ullint Hdim = pow(resolution, dim);
    std::vector<T> H(Hdim, 0);

    H[0] = 1;
    std::vector<double> intsqrt(resolution+1, 0);
    for (int ii = 0; ii<=resolution; ii++) {
        intsqrt[ii] = std::sqrt((static_cast<double>(ii)));
    }
    std::vector<int> nextPos(dim, 1);
    std::vector<int> jumpFrom(dim, 1);
    std::vector<int> ek(dim, 0);
    std::vector<double> factors(resolution+1, 0);
    int jump = 0;

    for (ullint jj = 0; jj < Hdim-1; jj++) {

        if (jump > 0) {
            jumpFrom[jump] += 1;
            jump = 0;
        }


        for (int ii = 0; ii < dim; ii++) {
            std::vector<int> forwardStep(dim, 0);
            forwardStep[ii] = 1;

            if ( forwardStep[ii] + nextPos[ii] > resolution) {
                nextPos[ii] = 1;
                jumpFrom[ii] = 1;
                jump = ii+1;
            }
            else {
                jumpFrom[ii] = nextPos[ii];
                nextPos[ii] = nextPos[ii] + 1;
                break;
            }
        }

        for (int ii = 0; ii < dim; ii++)
            ek[ii] = nextPos[ii] - jumpFrom[ii];

        int k = 0;
        for(; k < static_cast<int>(ek.size()); k++) {
            if(ek[k]) break;
        }

        ullint nextCoordinate = vec2index(nextPos, resolution);
        ullint fromCoordinate = vec2index(jumpFrom, resolution);

        H[nextCoordinate] = H[nextCoordinate] + y[k]/(intsqrt[nextPos[k]-1]);
        H[nextCoordinate] = H[nextCoordinate] * H[fromCoordinate];

        std::vector<int> tmpjump(dim, 0);

        for (int ii = 0; ii < dim; ii++) {
            if (jumpFrom[ii] > 1) {
                std::vector<int> prevJump(dim, 0);
                prevJump[ii] = 1;
                std::transform(jumpFrom.begin(), jumpFrom.end(), prevJump.begin(), tmpjump.begin(), std::minus<int>());
                ullint prevCoordinate = vec2index(tmpjump, resolution);
                H[nextCoordinate] = H[nextCoordinate] - (intsqrt[jumpFrom[ii]-1]/intsqrt[nextPos[k]-1])*(R[k*dim+ii])*H[prevCoordinate];

            }
        }

    }
    return H;

}


/**
 * Returns the matrix elements of an interferometer parametrized in terms of its R matrix
 *
 * @param R a flattened vector of size \f$n^2\f$, representing a
 *       \f$n\times n\f$ symmetric matrix.
 * @param resolution highest number of photons to be resolved.
 *
 */

template <typename T>
inline std::vector<T> interferometer_cpp(std::vector<T> &R, int &resolution) {
    int dim = std::sqrt(static_cast<double>(R.size()));

    ullint Hdim = pow(resolution, dim);
    std::vector<T> H(Hdim, 0);

    H[0] = 1;
    std::vector<double> intsqrt(resolution+1, 0);
    for (int ii = 0; ii<=resolution; ii++) {
        intsqrt[ii] = std::sqrt((static_cast<double>(ii)));
    }
    assert(dim % 2 == 0);
    std::vector<int> nextPos(dim, 1);
    std::vector<int> jumpFrom(dim, 1);
    std::vector<int> ek(dim, 0);
    std::vector<double> factors(resolution+1, 0);
    int jump = 0;

    for (ullint jj = 0; jj < Hdim-1; jj++) {

        if (jump > 0) {
            jumpFrom[jump] += 1;
            jump = 0;
        }


        for (int ii = 0; ii < dim; ii++) {
            std::vector<int> forwardStep(dim, 0);
            forwardStep[ii] = 1;

            if ( forwardStep[ii] + nextPos[ii] > resolution) {
                nextPos[ii] = 1;
                jumpFrom[ii] = 1;
                jump = ii+1;
            }
            else {
                jumpFrom[ii] = nextPos[ii];
                nextPos[ii] = nextPos[ii] + 1;
                break;
            }
        }

        int num_modes = dim/2;

        int bran = 0;
        for (int ii=0; ii < num_modes; ii++) {
            bran += nextPos[ii];
        }

        int ketn = 0;
        for (int ii=num_modes; ii < dim; ii++) {
            ketn += nextPos[ii];
        }
        if (bran == ketn) {
            for (int ii = 0; ii < dim; ii++)
                ek[ii] = nextPos[ii] - jumpFrom[ii];

            int k = 0;
            for(; k < static_cast<int>(ek.size()); k++) {
                if(ek[k]) break;
            }

            ullint nextCoordinate = vec2index(nextPos, resolution);
            ullint fromCoordinate = vec2index(jumpFrom, resolution);

            std::vector<int> tmpjump(dim, 0);
            int low_lim;
            int high_lim;

            if (k > num_modes) {
                low_lim = 0;
                high_lim = num_modes;
            }
            else {
                low_lim = num_modes;
                high_lim = dim;
            }

            for (int ii = low_lim; ii < high_lim; ii++) {
                if (jumpFrom[ii] > 1) {
                    std::vector<int> prevJump(dim, 0);
                    prevJump[ii] = 1;
                    std::transform(jumpFrom.begin(), jumpFrom.end(), prevJump.begin(), tmpjump.begin(), std::minus<int>());
                    ullint prevCoordinate = vec2index(tmpjump, resolution);
                    H[nextCoordinate] = H[nextCoordinate] - (intsqrt[jumpFrom[ii]-1]/intsqrt[nextPos[k]-1])*(R[k*dim+ii])*H[prevCoordinate];

                }
            }
        }
    }
    return H;

}


/**
 * Returns the matrix elements of a single mode squeezeing operation parametrized in terms of its R matrix
 *
 * @param R a flattened vector of size 4, representing a
 *       \f$2\times 2\f$ symmetric matrix.
 * @param resolution highest number of photons to be resolved.
 *
 */


template <typename T>
inline std::vector<T> squeezing_cpp(std::vector<T> &R, int &resolution) {
    int dim = std::sqrt(static_cast<double>(R.size()));

    ullint Hdim = pow(resolution, dim);
    std::vector<T> H(Hdim, 0);

    H[0] = std::sqrt(-R[1]);
    std::vector<double> intsqrt(resolution+1, 0);
    for (int ii = 0; ii<=resolution; ii++) {
        intsqrt[ii] = std::sqrt((static_cast<double>(ii)));
    }
    assert(dim == 2);
    std::vector<int> nextPos(dim, 1);
    std::vector<int> jumpFrom(dim, 1);
    std::vector<int> ek(dim, 0);
    std::vector<double> factors(resolution+1, 0);
    int jump = 0;

    for (ullint jj = 0; jj < Hdim-1; jj++) {

        if (jump > 0) {
            jumpFrom[jump] += 1;
            jump = 0;
        }


        for (int ii = 0; ii < dim; ii++) {
            std::vector<int> forwardStep(dim, 0);
            forwardStep[ii] = 1;

            if ( forwardStep[ii] + nextPos[ii] > resolution) {
                nextPos[ii] = 1;
                jumpFrom[ii] = 1;
                jump = ii+1;
            }
            else {
                jumpFrom[ii] = nextPos[ii];
                nextPos[ii] = nextPos[ii] + 1;
                break;
            }
        }

        int num_modes = dim/2;
        int bran = nextPos[0];
        int ketn = nextPos[1];
        if (bran % 2 == ketn % 2) {
            for (int ii = 0; ii < dim; ii++)
                ek[ii] = nextPos[ii] - jumpFrom[ii];

            int k = 0;
            for(; k < static_cast<int>(ek.size()); k++) {
                if(ek[k]) break;
            }

            ullint nextCoordinate = vec2index(nextPos, resolution);
            ullint fromCoordinate = vec2index(jumpFrom, resolution);

            std::vector<int> tmpjump(dim, 0);
            for (int ii = 0; ii < dim; ii++) {
                if (jumpFrom[ii] > 1) {
                    std::vector<int> prevJump(dim, 0);
                    prevJump[ii] = 1;
                    std::transform(jumpFrom.begin(), jumpFrom.end(), prevJump.begin(), tmpjump.begin(), std::minus<int>());
                    ullint prevCoordinate = vec2index(tmpjump, resolution);
                    H[nextCoordinate] = H[nextCoordinate] - (intsqrt[jumpFrom[ii]-1]/intsqrt[nextPos[k]-1])*(R[k*dim+ii])*H[prevCoordinate];

                }
            }
        }
    }
    return H;
}

/**
 * Returns the matrix elements of a displacement operation parametrized in terms of its double vector y
 *
 * @param y a flattened vector of size \f$2\f$, represeting the displacement via \f$\alpha, \alpha^*\f$
 * @param resolution highest number of photons to be resolved.
 *
 */

template <typename T>
inline std::vector<T> displacement_cpp(std::vector<T> &y, int &resolution) {
    int dim = 2;

    ullint Hdim = pow(resolution, dim);
    std::vector<T> H(Hdim, 0);

    H[0] = std::exp(0.5*y[0]*y[1]);
    std::vector<double> intsqrt(resolution+1, 0);
    for (int ii = 0; ii<=resolution; ii++) {
        intsqrt[ii] = std::sqrt((static_cast<double>(ii)));
    }
    std::vector<int> nextPos(dim, 1);
    std::vector<int> jumpFrom(dim, 1);
    std::vector<int> ek(dim, 0);
    std::vector<double> factors(resolution+1, 0);
    int jump = 0;

    for (ullint jj = 0; jj < Hdim-1; jj++) {

        if (jump > 0) {
            jumpFrom[jump] += 1;
            jump = 0;
        }


        for (int ii = 0; ii < dim; ii++) {
            std::vector<int> forwardStep(dim, 0);
            forwardStep[ii] = 1;

            if ( forwardStep[ii] + nextPos[ii] > resolution) {
                nextPos[ii] = 1;
                jumpFrom[ii] = 1;
                jump = ii+1;
            }
            else {
                jumpFrom[ii] = nextPos[ii];
                nextPos[ii] = nextPos[ii] + 1;
                break;
            }
        }

        for (int ii = 0; ii < dim; ii++)
            ek[ii] = nextPos[ii] - jumpFrom[ii];

        int k = 0;
        for(; k < static_cast<int>(ek.size()); k++) {
            if(ek[k]) break;
        }

        ullint nextCoordinate = vec2index(nextPos, resolution);
        ullint fromCoordinate = vec2index(jumpFrom, resolution);

        H[nextCoordinate] = H[nextCoordinate] + y[k]/(intsqrt[nextPos[k]-1]);
        H[nextCoordinate] = H[nextCoordinate] * H[fromCoordinate];

        std::vector<int> tmpjump(dim, 0);


        int ii = 0;
        if(k==0){
        	ii = 1;
        }

        if (jumpFrom[ii] > 1) {
            std::vector<int> prevJump(dim, 0);
            prevJump[ii] = 1;
            std::transform(jumpFrom.begin(), jumpFrom.end(), prevJump.begin(), tmpjump.begin(), std::minus<int>());
            ullint prevCoordinate = vec2index(tmpjump, resolution);
            H[nextCoordinate] = H[nextCoordinate] - (intsqrt[jumpFrom[ii]-1]/intsqrt[nextPos[k]-1])*H[prevCoordinate];

	    }

    }
    return H;

}


}



