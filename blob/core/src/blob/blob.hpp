#ifndef BLOB_TRACKER_BLOB_HPP
#define BLOB_TRACKER_BLOB_HPP

/**
 * \brief Implementation of detection and tracking algorithms.
 */
namespace blobs {
	template<typename FPT>
	struct blob {
		int y, x;
		FPT r;
	};
}

#endif //BLOB_TRACKER_BLOB_HPP
