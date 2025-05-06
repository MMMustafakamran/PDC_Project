/*
 * Copyright (c) 2014-2021 Kevin Huck
 * Copyright (c) 2014-2021 University of Oregon
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#pragma once

#include <stdio.h>
#include <stdlib.h>

inline void _apex_assert(const char* expression, const char* file, int line)
{
    fprintf(stderr, "Assertion '%s' failed, file '%s' line '%d'.",
        expression, file, line);
    abort();
}

#ifdef NDEBUG
#define APEX_ASSERT(EXPRESSION) ((void)0)
#else
#define APEX_ASSERT(EXPRESSION) ((EXPRESSION) ? (void)0 : \
    _apex_assert(#EXPRESSION, __FILE__, __LINE__))
#endif
