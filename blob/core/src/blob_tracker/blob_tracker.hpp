#ifndef BLOB_TRACKER_BLOB_TRACKER_HPP
#define BLOB_TRACKER_BLOB_TRACKER_HPP

#include "../blob/blob.hpp"
#include "kalman_filter/kalman_filter.hpp"
#include "hungarian/hungarian.hpp"
#include <optional>
#include <unordered_set>

namespace blobs {
	/**
	 * \brief Establishes one-to-one correspondence between objects in previous and current frames.
	 */
	class blob_tracker {
	public:
		constexpr static const size_t MAXN = hungarian::MAXN;	/*!< Maximum number of objects per frame */
		blob_tracker(bool use_prediction, bool use_ttl, int ttl);

		/*! Object tracking status */
		enum class o_status {
			BORN,	/*!< Object first appeared in current frame */
			
			ALIVE,	/*!< Object existed in previous frame and was detected again */
			
			GHOST,	/*!< Temporary lost object state:
					*   - Object was detected previously but is currently lost
					*   - Will transition to ALIVE if redetected
					*   - Will transition to DIED if TTL expires
					*   - Maintained for TTL frames using Kalman filter prediction
					*   - Enables tracking through brief occlusions */
			
			DIED	/*!< Permanent object termination:
					*   - TTL counter reached zero
					*   - Object ID is released and may be reused
					*   - Subsequent objects with same ID are new entities */
		};

		struct object {
			o_status status;
			unsigned int id;
			blob<double> blob_data;
		};

		/**
		 * \brief Establishes one-to-one correspondence between objects in consecutive frames.
		 * 
		 * \details Assigns IDs and statuses to current frame objects by solving the assignment problem:
		 *          - Matches objects to minimize total displacement between frames
		 *          - Applies movement constraints when threshold is finite:
		 *            * Objects cannot be matched if their distance exceeds move_threshold
		 *            * Accounts for physical movement limitations between frames
		 *          - Uses Hungarian algorithm for optimal assignment
		 * 
		 * \param blobs             Objects detected in current frame
		 * \param move_threshold    Maximum allowed displacement between frames (pixels):
		 *                          - Infinite: no movement constraint
		 *                          - Finite: physical movement limit
		 * \param scale_threshold   Maximum allowed radius change between frames (pixels)
		 * 
		 * \return Tracked objects with assigned IDs and statuses
		 */
		std::vector<object>
		track(const std::vector<blob<double>> &blobs, double move_threshold, double scale_threshold);

	private:
		struct internal_state {
			internal_state();
			object obj{};
			kalman_filter kf;
			int ttl;
		};
	private:
		bool m_use_prediction, m_use_ttl;
		int initial_ttl;
		std::vector<internal_state> state;
		std::unordered_set<unsigned int> id_pool;
	};
}

#endif //BLOB_TRACKER_BLOB_TRACKER_HPP
