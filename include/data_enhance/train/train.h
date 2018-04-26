#ifndef data_enhance_TRAIN_TRAIN_H_
#define data_enhance_TRAIN_TRAIN_H_

#include <opencv2/opencv.hpp>

namespace data_enhance {

class ITrain {
 public:
  ITrain();

  virtual ~ITrain();

  virtual void train() = 0;

  virtual void test() = 0;

 private:
  virtual cv::Ptr<cv::ml::TrainData> tdata() = 0;
};
}

#endif  // data_enhance_TRAIN_TRAIN_H_
