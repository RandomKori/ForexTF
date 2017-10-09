// Test1.cpp: определяет точку входа для консольного приложения.
//


#pragma once
#define COMPILER_MSVC
#define NOMINMAX

#include "stdafx.h"


#include <tensorflow/core/public/session.h>
#include <tensorflow/core/protobuf/meta_graph.pb.h>

using namespace std;
using namespace tensorflow;

Session* session;
MetaGraphDef graph_def;

int main()
{
	wchar_t* s = L"D:\\ModelFORTS\\ResNetFXModel";
	wstring d(s);
	wstring ptg = d + L".meta";
	const string pathToGraph(ptg.begin(), ptg.end());
	const string checkpointPath(d.begin(), d.end());

	session = NewSession(SessionOptions());
	Status status;
	status = ReadBinaryProto(Env::Default(), pathToGraph, &graph_def);
	status = session->Create(graph_def.graph_def());
	float m[45];
	Tensor checkpointPathTensor(DT_STRING, TensorShape());
	checkpointPathTensor.scalar<std::string>()() = checkpointPath;
	status = session->Run({ { graph_def.saver_def().filename_tensor_name(), checkpointPathTensor }, }, {}, { graph_def.saver_def().restore_op_name() }, nullptr);
	vector<Tensor> outputs;
	Tensor inputTensor(DT_FLOAT, TensorShape({ 1, 45, 1 }));
	auto fl = inputTensor.tensor_data();
	fl.set(m,sizeof(float)*45);
	std::vector<std::pair<string, tensorflow::Tensor>> inputs = {
		{ "Imputs/Placeholder", inputTensor }
	};
	status=session->Run(inputs, { "Layer_out/Classes" }, {}, &outputs);
	float o[3];
	auto tfouts = outputs[0].tensor_data();
	for (int i = 0; i < 3; i++)
		o[i] = tfouts[i];
	return 0;
}

