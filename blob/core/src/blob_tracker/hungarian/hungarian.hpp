#ifndef BLOB_TRACKER_HUNGARIAN_HPP
#define BLOB_TRACKER_HUNGARIAN_HPP

#include <vector>
#include <boost/numeric/ublas/matrix.hpp>

namespace hungarian {
    constexpr static const size_t MAXN = 500;  /*!< Maximum number of objects in one partition. */

    /**
     * \brief Implementation of the Hungarian algorithm that solves the assignment
	 * problem in O(M * N^2) (minimum weight perfect matching in a bipartite graph).
     * 
     * \param g		Adjacency matrix of the bipartite graph.
	 * 				g[u][v] - edge weight between vertices u and v.
     * \return		ret[i] = -1, if no matching vertex exists in the right partition for vertex i.
     *				ret[i] = j,  if vertex i from the left partition is matched with vertex j
	 *				from the right partition. 
     */
    std::vector<ptrdiff_t>
    hungarian(const boost::numeric::ublas::matrix<float> &g);

    /**
     * \brief Implementation of the Hungarian algorithm that solves the assignment
	 * problem in O(M * N^2) (minimum weight perfect matching in a bipartite graph).
	 * 
	 * This version includes an additional constraint - some edges between vertices
	 * of left and right partitions are missing.
     * 
     * \param g		Adjacency matrix of the bipartite graph.
	 *				g[u][v] - edge weight between vertices u and v.
     * \param c		Matrix specifying existing/non-existing edges between vertex pairs.
     *				c[y][x] = true, if vertices y and x are connected by an edge.
     *				c[y][x] = false, if edge is missing (g[y][x] is ignored and can be any value).
     * \return		ret[i] = -1, if no matching vertex exists in the right partition for vertex i.
     *				ret[i] = j,  if vertex i from the left partition is matched with vertex j
	 * 				from the right partition. 
     */
    std::vector<ptrdiff_t>
    hungarian(boost::numeric::ublas::matrix<float> g,
        const boost::numeric::ublas::matrix<bool> &c);
};

#endif //BLOB_TRACKER_HUNGARIAN_HPP
