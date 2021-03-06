#ifndef COLLECT_PICTURE_HPP
#define COLLECT_PICTURE_HPP

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
//#include <tesseract/baseapi.h>
#include <iostream>
#include "pre_process.hpp"
#include "data_enhance/util/util.h"
#include "data_enhance/config.h"
#include "data_enhance/core/chars_identify.h"


class CollectPic
{
public:
    CollectPic(const char* chars_folder);
    ~CollectPic();
    int imgCollect();
    int cdata(std::vector<Mat> &matVec);

private:
    const char* chars_folder_;
};

#endif  // COLLECT_PICTURE_HPP
