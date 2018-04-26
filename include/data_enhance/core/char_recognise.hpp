#ifndef CHAR_RECOGNISE_HPP
#define CHAR_RECOGNISE_HPP

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include "pre_process.hpp"
#include "data_enhance/core/chars_identify.h"
#include "data_enhance/config.h"
#include <tesseract/baseapi.h>
#include "data_enhance/util/util.h"


class CharRecog
{
public:
    CharRecog(const char* chars_folder);
    ~CharRecog();
    int charRecognise();
    Mat preprocessChar(Mat in);
    int cdata(std::vector<cv::Mat> &matVec);


    static const int CHAR_SIZE = 20;
private:
    const char* chars_folder_;
};

#endif  // CHAR_RECOGNISE_HPP
