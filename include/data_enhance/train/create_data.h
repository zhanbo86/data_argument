#ifndef data_enhance_CREATE_DATA_H_
#define data_enhance_CREATE_DATA_H_

#include "opencv2/opencv.hpp"
#include "data_enhance/config.h"

using namespace cv;
using namespace std;

/*! \namespace data_enhance
Namespace where all the C++ data_enhance functionality resides
*/
namespace data_enhance {

  // shift an image
  Mat translateImg(Mat img, int offsetx, int offsety, int bk = 0);
  // rotate an image
  Mat rotateImg(Mat source, float angle, int bk = 0);

  // crop the image
  Mat cropImg(Mat src, int x, int y, int shift, int bk = 0);

  Mat generateSyntheticImage(const Mat& image, int use_swap = 1);

} /*! \namespace data_enhance*/

#endif  // data_enhance_CREATE_DATA_H_
