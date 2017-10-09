// NewNeuroDense.cpp: определяет экспортированные функции для приложения DLL.
//
#pragma once
#define COMPILER_MSVC
#define NOMINMAX

#include "stdafx.h"

#include <tensorflow/core/public/session.h>
#include <tensorflow/core/protobuf/meta_graph.pb.h>

using namespace std;
using namespace tensorflow;

extern "C" __declspec(dllexport) void LoadModel(wchar_t* s);
extern "C" __declspec(dllexport) void EvalModel(double* inp, double* out);
extern "C" __declspec(dllexport) void DeInit();

Session* session;
MetaGraphDef graph_def;

void LoadModel(wchar_t* s)
{
	wstring d(s);
	wstring ptg = d + L".meta";

	const string pathToGraph(ptg.begin(), ptg.end());
	const string checkpointPath(d.begin(), d.end());

	session = NewSession(SessionOptions());
	Status status;
	status = ReadBinaryProto(Env::Default(), pathToGraph, &graph_def);
	status = session->Create(graph_def.graph_def());

	Tensor checkpointPathTensor(DT_STRING, TensorShape());
	checkpointPathTensor.scalar<std::string>()() = checkpointPath;
	status = session->Run({ { graph_def.saver_def().filename_tensor_name(), checkpointPathTensor }, }, {}, { graph_def.saver_def().restore_op_name() }, nullptr);
}

void EvalModel(double* inp, double* out)
{
	vector<Tensor> outputs;
	Tensor inputTensor(DT_FLOAT, TensorShape({ 1, 45 }));
	auto m = inputTensor.flat<float>();

	for (int i = 0; i < 45; i++)
		m(i) = (float)(inp[i]);
	std::vector<std::pair<string, tensorflow::Tensor>> inputs = {
		{ "Inputs/Input", inputTensor }
	};
	session->Run(inputs, { "predictions/Classes" }, {}, &outputs);
	auto tfouts = outputs[0].flat<float>();
	for (int i = 0; i < 2; i++)
		out[i] = tfouts(i);
}

void DeInit()
{
	session->Close();
}



