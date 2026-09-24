/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once

#include <assert.h>
#include <stdlib.h>
#include "backend_runtime.h"

#define CHECK_CUDA(call)                        \
    do {                                                                                                  \
        flash::runtime::Error status_ = call;                                                             \
        if (status_ != flash::runtime::success) {                                                         \
            fprintf(stderr, "GPU error (%s:%d): %s\n", __FILE__, __LINE__, flash::runtime::error_string(status_)); \
            exit(1);                                                                                      \
        }                                                                                                 \
    } while(0)

#define CHECK_CUDA_KERNEL_LAUNCH() CHECK_CUDA(flash::runtime::last_error())
