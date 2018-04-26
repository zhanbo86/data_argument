#ifndef data_enhance_CONFIG_H_
#define data_enhance_CONFIG_H_

#define CV_VERSION_THREE_TWO

//#define DEBUG

namespace data_enhance {

  enum Color { BLUE, YELLOW, WHITE, UNKNOWN };

  enum LocateType { SOBEL, COLOR, CMSER, OTHER };

  enum CharSearchDirection { LEFT, RIGHT };

  enum
  {
    PR_MODE_UNCONSTRAINED,
    PR_MODE_CAMERPOHNE,
    PR_MODE_PARKING,
    PR_MODE_HIGHWAY
  };

  enum
  {
    PR_DETECT_SOBEL = 0x01,  /**Sobel detect type, using twice Sobel  */
    PR_DETECT_COLOR = 0x02,  /**Color detect type   */
    PR_DETECT_CMSER = 0x04,  /**Character detect type, using mser  */
  };


static const char* kDefaultAnnPath = "../../../src/data_enhance/train_set/ann.xml";


typedef enum {
  kForward = 1, // correspond to "has plate"
  kInverse = 0  // correspond to "no plate"
} SvmLabel;


static const int   kPlateResizeWidth = 136;
static const int   kPlateResizeHeight = 36;

static const int   kShowWindowWidth = 1000;
static const int   kShowWindowHeight = 800;

static const float kSvmPercentage   = 0.7f;

static const int   kCharacterInput  = 120;
static const int   kAnnInput = kCharacterInput;

static const int   kCharacterSize = 10;
static const int   kPredictSize = kCharacterSize;

static const int   kNeurons       = 40;

static const char *kChars[] = {
  "0", "1", "2",
  "3", "4", "5",
  "6", "7", "8",
  "9","A",
//    "00",/* "01", */"02","03", "04", "05","08", "11", "15","16","17",
//    /*"17",*/ "18", "19","20", "21", "26","29", "30", "31","32",
//    "36", "37", "38","39", "40", "41","42", "43", "44","45",
//    "46", "50", "51","61", "70", "71","73", "74", "80","81",
//    "90", "91", "a7",
//    "001", "011", "017","031", "041", "042","050", "117", "151","153",
//    "161", "170", "171","172", "173", "180","181", "191", "201","217",
//    "261", "290", "301","317", "318", "321","361", "371", "381","401",
//    "417", "429", "431","461", "508", "517","617", "704", "705","717",
//    "719", "724", "738","739", "744", "745","802", "817", "900","917",
//    "a74",
//    "0117", "0318", "0417","0429", "0508", "1517","1539", "1610", "1617","1704",
//    "1705", "1717", "1719","1724", "1738", "1739","1752", "1802", "1817","1917",
//    "2017", "2617", "2900","3017", "3217", "3617","3717", "3817", "4017","4290",
//    "4317", "4617", "7042","7050", "9001", "a744","a745",
  /*  10  */
//  "A", "B", "C",
//  "D", "E", "F",
//  "G", "H", /* {"I", "I"} */
//  "J", "K", "L",
//  "M", "N", /* {"O", "O"} */
//  "P", "Q", "R",
//  "S", "T", "U",
//  "V", "W", "X",
//  "Y", "Z"
//  /*  24  */
};

static const int kCharactersNumber = 10;

static bool kDebug = false;

static const int kGrayCharWidth = 20;
static const int kGrayCharHeight = 32;
  static const int kCharLBPGridX = 4;
  static const int kCharLBPGridY = 4;
  static const int kCharLBPPatterns = 16;

  static const int kCharHiddenNeurans = 64;

  static const int kCharsCountInOnePlate = 7;
  static const int kSymbolsCountInChinesePlate = 6;

  static const float kPlateMaxSymbolCount = 7.5f;
  static const int kSymbolIndex = 2;

// Disable the copy and assignment operator for this class.
#define DISABLE_ASSIGN_AND_COPY(className) \
private:\
  className& operator=(const className&); \
  className(const className&)

// Display the image.
#define SET_DEBUG(param) \
  kDebug = param

// Display the image.
#define SHOW_IMAGE(imgName, debug) \
  if (debug) { \
    namedWindow("imgName", WINDOW_AUTOSIZE); \
    moveWindow("imgName", 500, 500); \
    imshow("imgName", imgName); \
    waitKey(0); \
    destroyWindow("imgName"); \
  }

// Load model. compatitable withe 3.0, 3.1 and 3.2
#ifdef CV_VERSION_THREE_TWO
  #define LOAD_SVM_MODEL(model, path) \
    model = ml::SVM::load(path);
  #define LOAD_ANN_MODEL(model, path) \
    model = ml::ANN_MLP::load(path);
#else
  #define LOAD_SVM_MODEL(model, path) \
    model = ml::SVM::load<ml::SVM>(path);
  #define LOAD_ANN_MODEL(model, path) \
    model = ml::ANN_MLP::load<ml::ANN_MLP>(path);
#endif

}

#endif // data_enhance_CONFIG_H_
