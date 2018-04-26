#ifndef data_enhance_TRAIN_ANNTRAIN_H_
#define data_enhance_TRAIN_ANNTRAIN_H_

#include "data_enhance/train/train.h"
#include "data_enhance/util/kv.h"
#include <memory>

namespace data_enhance {

class AnnTrain : public ITrain {
 public:
  explicit AnnTrain(const char* chars_folder, const char* xml);

  virtual void train();

  virtual void test();

  std::pair<std::string, std::string> identifyChinese(cv::Mat input);
  std::pair<std::string, std::string> identify(cv::Mat input);

 private:
  virtual cv::Ptr<cv::ml::TrainData> tdata();

  cv::Ptr<cv::ml::TrainData> sdata(size_t number_for_count = 100);

  cv::Ptr<cv::ml::ANN_MLP> ann_;
  const char* ann_xml_;
  const char* chars_folder_;
};
}

#endif  // data_enhance_TRAIN_ANNTRAIN_H_
