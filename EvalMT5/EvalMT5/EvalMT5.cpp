// EvalMT5.cpp: определяет экспортированные функции для приложения DLL.
//

#include "stdafx.h"
#include <stdio.h>
#include <c_api.h>

extern "C" __declspec(dllexport) void LoadModel(wchar_t* s);
extern "C" __declspec(dllexport) void EvalModel(double* inp, double* out);
extern "C" __declspec(dllexport) void DeInit();

void LoadModel(wchar_t* s)
{

}

void EvalModel(double* inp, double* out)
{

}

void DeInit()
{

}

