#ifndef PLATFORM_CBLAS_H
#define PLATFORM_CBLAS_H

// Cross-platform CBLAS header
#ifdef __APPLE__
    // macOS uses Accelerate framework
    #include <Accelerate/Accelerate.h>
#else
    // Linux uses standard cblas
    #include <cblas.h>
#endif

#endif // PLATFORM_CBLAS_H
