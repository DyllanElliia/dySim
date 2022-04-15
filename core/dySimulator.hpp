/*
 * @Author: DyllanElliia
 * @Date: 2022-02-16 14:55:17
 * @LastEditors: DyllanElliia
 * @LastEditTime: 2022-04-14 17:23:24
 * @Description:
 */
#pragma once
#ifdef _dym_test_
#include "simulator/mls_mpm3.hpp"
#else
#include "simulator/mls_mpm.hpp"
#endif

#ifdef DYM_USE_MARCHING_CUBES
#include "tools/graphicAlg/marchingCube.hpp"
#endif