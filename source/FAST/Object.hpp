#pragma once

#define NOMINMAX // Removes windows min and max macros
#define _USE_MATH_DEFINES
#include "FAST/Exception.hpp"
#include "FAST/Reporter.hpp"
#include <memory>

#define FAST_OBJECT(className)                                  \
    public:                                                     \
        typedef std::shared_ptr<className> pointer;               \
        static std::shared_ptr<className> New() {                       \
            std::shared_ptr<className> smartPtr(new className());   \
            smartPtr->setPtr(smartPtr);                              \
                                                                \
            return smartPtr;                                    \
        }                                                       \
        virtual std::string getNameOfClass() const {            \
            return std::string(#className);                     \
        };                                                      \
        static std::string getStaticNameOfClass() {             \
            return std::string(#className);                     \
        };                                                      \
    private:                                                    \
        void setPtr(className::pointer ptr) {                   \
            mPtr = ptr;                                         \
        }                                                       \

#define FAST_PROCESS_OBJECT(className) \
    FAST_OBJECT(className)             \
    private:                           \
        std::shared_ptr<className> getSharedPtr() { return std::dynamic_pointer_cast<className>(mPtr.lock()); }                                 \
    public:                                   \
    std::shared_ptr<className> connect(uint inputPortID, std::shared_ptr<ProcessObject> parentProcessObject, uint outputPortID = 0) { \
            return std::dynamic_pointer_cast<className>(ProcessObject::connect(inputPortID, parentProcessObject, outputPortID));                                                        \
    };                                                        \
    std::shared_ptr<className> connect(std::shared_ptr<ProcessObject> parentProcessObject, uint outputPortID = 0) {                  \
        return std::dynamic_pointer_cast<className>(ProcessObject::connect(parentProcessObject, outputPortID));                                                \
    };                                 \


// Macro for counting arguments, add more if needed: From https://stackoverflow.com/questions/2124339/c-preprocessor-va-args-number-of-arguments
#ifdef _MSC_VER // Microsoft compilers

#define FAST_GET_ARG_COUNT(...)  FAST_INTERNAL_EXPAND_ARGS_PRIVATE(FAST_INTERNAL_ARGS_AUGMENTER(__VA_ARGS__))

#define FAST_INTERNAL_ARGS_AUGMENTER(...) unused, __VA_ARGS__
#define FAST_INTERNAL_EXPAND(x) x
#define FAST_INTERNAL_EXPAND_ARGS_PRIVATE(...) FAST_INTERNAL_EXPAND(FAST_INTERNAL_GET_ARG_COUNT_PRIVATE(__VA_ARGS__, 69, 68, 67, 66, 65, 64, 63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0))
#define FAST_INTERNAL_GET_ARG_COUNT_PRIVATE(_1_, _2_, _3_, _4_, _5_, _6_, _7_, _8_, _9_, _10_, _11_, _12_, _13_, _14_, _15_, _16_, _17_, _18_, _19_, _20_, _21_, _22_, _23_, _24_, _25_, _26_, _27_, _28_, _29_, _30_, _31_, _32_, _33_, _34_, _35_, _36, _37, _38, _39, _40, _41, _42, _43, _44, _45, _46, _47, _48, _49, _50, _51, _52, _53, _54, _55, _56, _57, _58, _59, _60, _61, _62, _63, _64, _65, _66, _67, _68, _69, _70, count, ...) count

#else // Non-Microsoft compilers

#define FAST_GET_ARG_COUNT(...) FAST_INTERNAL_GET_ARG_COUNT_PRIVATE(0, ## __VA_ARGS__, 70, 69, 68, 67, 66, 65, 64, 63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0)
#define FAST_INTERNAL_GET_ARG_COUNT_PRIVATE(_0, _1_, _2_, _3_, _4_, _5_, _6_, _7_, _8_, _9_, _10_, _11_, _12_, _13_, _14_, _15_, _16_, _17_, _18_, _19_, _20_, _21_, _22_, _23_, _24_, _25_, _26_, _27_, _28_, _29_, _30_, _31_, _32_, _33_, _34_, _35_, _36, _37, _38, _39, _40, _41, _42, _43, _44, _45, _46, _47, _48, _49, _50, _51, _52, _53, _54, _55, _56, _57, _58, _59, _60, _61, _62, _63, _64, _65, _66, _67, _68, _69, _70, count, ...) count

#endif

//#define FAST_VA_NUM_ARGS(...) FAST_VA_NUM_ARGS_IMPL(__VA_ARGS__, 16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1)
//#define FAST_VA_NUM_ARGS_IMPL(_1,_2,_3,_4,_5,_6,_7,_8,_9,_10,_11,_12,_13,_14,_15,_16,N,...) N

// Macro for choosing function based on nr of arguments
#define FAST_macro_dispatcher(func, ...) \
            FAST_macro_dispatcher_(func, FAST_GET_ARG_COUNT(__VA_ARGS__))
#define FAST_macro_dispatcher_(func, nargs) \
            FAST_macro_dispatcher__(func, nargs)
#define FAST_macro_dispatcher__(func, nargs) \
            func ## nargs
#ifdef SWIG
%define FAST_CONSTRUCTOR(...) FAST_macro_dispatcher(FAST_CONSTRUCTOR, __VA_ARGS__)(__VA_ARGS__)
%enddef
#else
#define FAST_CONSTRUCTOR(...) FAST_macro_dispatcher(FAST_CONSTRUCTOR, __VA_ARGS__)(__VA_ARGS__)
#endif

// FAST constructor macros, not pretty and very repetitive, but it works.
#define FAST_CONSTRUCTOR4(className, TYPE1, NAME1, VAL1) \
    static std::shared_ptr<className> create(TYPE1 NAME1 VAL1) {\
        std::shared_ptr<className> ptr(new className(NAME1));   \
        ptr->setPtr(ptr);\
        return ptr;\
    };                                             \
    protected:\
        explicit className(TYPE1 NAME1 VAL1);                 \
    public:\


#define FAST_CONSTRUCTOR7(className, TYPE1, NAME1, VAL1, TYPE2, NAME2, VAL2) \
    static std::shared_ptr<className> create(TYPE1 NAME1 VAL1, TYPE2 NAME2 VAL2) {\
        std::shared_ptr<className> ptr(new className(NAME1, NAME2));   \
        ptr->setPtr(ptr);\
        return ptr;\
    };                                             \
    protected:\
        className(TYPE1 NAME1 VAL1, TYPE2 NAME2 VAL2);                 \
    public:\

#define FAST_CONSTRUCTOR10(className, TYPE1, NAME1, VAL1, TYPE2, NAME2, VAL2, TYPE3, NAME3, VAL3) \
    static std::shared_ptr<className> create(TYPE1 NAME1 VAL1, TYPE2 NAME2 VAL2, TYPE3 NAME3 VAL3) {\
        std::shared_ptr<className> ptr(new className(NAME1, NAME2, NAME3));   \
        ptr->setPtr(ptr);\
        return ptr;\
    };                                             \
    protected:\
        className(TYPE1 NAME1 VAL1, TYPE2 NAME2 VAL2, TYPE3 NAME3 VAL3);                 \
    public:\

#define FAST_CONSTRUCTOR13(className, TYPE1, NAME1, VAL1, TYPE2, NAME2, VAL2, TYPE3, NAME3, VAL3, TYPE4, NAME4, VAL4) \
    static std::shared_ptr<className> create(TYPE1 NAME1 VAL1, TYPE2 NAME2 VAL2, TYPE3 NAME3 VAL3, TYPE4 NAME4 VAL4) {\
        std::shared_ptr<className> ptr(new className(NAME1, NAME2, NAME3, NAME4));   \
        ptr->setPtr(ptr);\
        return ptr;\
    };                                             \
    protected:\
        className(TYPE1 NAME1 VAL1, TYPE2 NAME2 VAL2, TYPE3 NAME3 VAL3, TYPE4 NAME4 VAL4);                 \
    public:                                                                                                           \

#define FAST_CONSTRUCTOR16(className, TYPE1, NAME1, VAL1, TYPE2, NAME2, VAL2, TYPE3, NAME3, VAL3, TYPE4, NAME4, VAL4, TYPE5, NAME5, VAL5) \
    static std::shared_ptr<className> create(TYPE1 NAME1 VAL1, TYPE2 NAME2 VAL2, TYPE3 NAME3 VAL3, TYPE4 NAME4 VAL4, TYPE5 NAME5 VAL5) {\
        std::shared_ptr<className> ptr(new className(NAME1, NAME2, NAME3, NAME4, NAME5));   \
        ptr->setPtr(ptr);\
        return ptr;\
    };                                             \
    protected:\
        className(TYPE1 NAME1 VAL1, TYPE2 NAME2 VAL2, TYPE3 NAME3 VAL3, TYPE4 NAME4 VAL4, TYPE5 NAME5 VAL5);                 \
    public:

/** @namespace fast
 *  @brief FAST namespace - All FAST code lives in this namespace.
 */
namespace fast {

enum StreamingMode { STREAMING_MODE_NEWEST_FRAME_ONLY, STREAMING_MODE_STORE_ALL_FRAMES, STREAMING_MODE_PROCESS_ALL_FRAMES };

/**
 * \brief Base class for all FAST objects.
 *
 * This class has some functionality common to all FAST objects.
 */
class FAST_EXPORT  Object {
    public:
        typedef std::shared_ptr<Object> pointer;
        Object();
        virtual ~Object() {};
        static std::string getStaticNameOfClass() {
            return "Object";
        }
        Reporter& getReporter();
    protected:
        Reporter& reportError();
        Reporter& reportWarning();
        Reporter& reportInfo();
        ReporterEnd reportEnd() const;
        std::weak_ptr<Object> mPtr;
    private:
        Reporter mReporter;

};

}
