#ifdef _MSC_VER					   //用于判断是否是 vs 平台
#define CROSS_PLATFORM_HIDDEN_API  // 这行其实就是充数的，在vs中放到那里啥用没有，但是有了这个定义就可以和 gcc下的代码保持同步
#ifdef CROSS_PLATFORM_LIBRARY_EXPORTS
#define CROSS_PLATFORM_API __declspec(dllexport)
#else
#define CROSS_PLATFORM_API __declspec(dllimport)
#endif
#else																   // 说明是 gcc 或者 clang
#define CROSS_PLATFORM_API __attribute((visibility("default")))		   // 明确指示，这个函数在动态库中可见
#define CROSS_PLATFORM_HIDDEN_API __attribute((visibility("hidden")))  // 明确指示，这个函数在动态库中不可见
#endif