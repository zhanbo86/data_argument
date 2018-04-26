#include "data_enhance.h"
#include "data_enhance/util/switch.hpp"
#include "data_enhance/core/char_recognise.hpp"
#include "data_enhance/core/collect_picture.hpp"
#include "data_enhance/util/util.h"
#include "data_enhance/util/kv.h"


int main(int argc, const char* argv[])
{
  std::shared_ptr<data_enhance::Kv> kv(new data_enhance::Kv);
  kv->load("../../../src/data_enhance/resources/text/chinese_mapping");
  bool isExit = false;
  while (!isExit) {
    data_enhance::Utils::print_file_lines("../../../src/data_enhance/resources/text/main_menu");
    std::cout << kv->get("make_a_choice") << ":";

    int select = -1;
    bool isRepeat = true;
    while (isRepeat) {
      std::cin >> select;
      isRepeat = false;
      switch (select) {
        case 1:
          {
             CollectPic picture_ocr("../../../src/data_enhance/raw_img/val_2");
             picture_ocr.imgCollect();
          }
          break;
        case 2:
          std::cout << "Run \"demo ann\" for more usage." << std::endl;
          {
            data_enhance::AnnTrain ann("../../../src/data_enhance/train_set/char4_mul", "../../../src/data_enhance/train_set/ann.xml");
            ann.train();
          }
          break;
        case 3:
          {
             CharRecog ocr_recog("../../../src/data_enhance/bometTest");
             ocr_recog.charRecognise();
          }
          break;
        case 4:
          isExit = true;
          break;
        default:
          std::cout << kv->get("input_error") << ":";
          isRepeat = true;
          break;
      }
    }
  }
  return 0;
}
