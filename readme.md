# NeuralNetLib

[![NuGet](https://img.shields.io/nuget/v/RichTea.NeuralNetLib.svg?style=flat)](https://www.nuget.org/packages/RichTea.NeuralNetLib/)

This project implements a basic neural network and training algorithms.

## Cake Tasks
This project uses [Cake](https://cakebuild.net)!
* ./cake.ps1 -target=Clean
* ./cake.ps1 -target=Restore-Nuget-Packages
* ./cake.ps1 -target=Build
* ./cake.ps1 -target=Test

## CI

|        | Windows | Linux |
| ------ | --------|-------|
| Master | [![Build status](https://ci.appveyor.com/api/projects/status/gy1oqhk2mmi9v6qf/branch/master?svg=true)](https://ci.appveyor.com/project/RichTeaMan/neuralnet/branch/master) | [![Build status](https://travis-ci.org/RichTeaMan/NeuralNet.svg?branch=master)](https://travis-ci.org/RichTeaMan/NeuralNet) |

## Example

Neural Net Lib supports back propagation:

``` csharp
Net net = new Net(new Random(), 2, 1);

BackPropagation prop = new BackPropagation(2, 1);
DataSet _1 = new DataSet(new double[] { 0, 0 }, new double[] { 0 });    // 0 | 0 = 0
DataSet _2 = new DataSet(new double[] { 0, 1 }, new double[] { 1 });    // 0 | 1 = 1
DataSet _3 = new DataSet(new double[] { 1, 0 }, new double[] { 1 });    // 1 | 0 = 1
DataSet _4 = new DataSet(new double[] { 1, 1 }, new double[] { 0 });    // 1 | 1 = 0

prop.AddDataSet(_1);
prop.AddDataSet(_2);
prop.AddDataSet(_3);
prop.AddDataSet(_4);

int epoch = 1000;
var backPropResult = prop.Train(net, epoch);

Assert.IsTrue(backPropResult.SSE < 0.2, "LogicNetXOR SSE after {0} epochs is '{1}'", epoch, backPropResult.SSE);
```

Genetic training is also a thing:
``` csharp
int iterations = 1000;
int population = 100;

DataSet _1 = new DataSet(new double[] { 0, 0 }, new double[] { 0 });    // 0 | 0 = 0
DataSet _2 = new DataSet(new double[] { 0, 1 }, new double[] { 1 });    // 0 | 1 = 1
DataSet _3 = new DataSet(new double[] { 1, 0 }, new double[] { 1 });    // 1 | 0 = 1
DataSet _4 = new DataSet(new double[] { 1, 1 }, new double[] { 0 });    // 1 | 1 = 0

var dataSets = new[] { _1, _2, _3, _4 };

var fitnessEvaluator = new DatasetEvaluator(dataSets);
var trainer = new GeneticAlgorithmTrainer<DatasetEvaluator>(new Random(), fitnessEvaluator);

var topNet = trainer.TrainAi(_1.InputCount, _1.OutputCount, 3, population, iterations).First();

double sse = 0;
foreach (var dataSet in dataSets)
{
    double result = topNet.Calculate(dataSet.Inputs).First();
    sse += Math.Pow(dataSet.Outputs.First() - result, 2.0);
}

Assert.IsTrue(sse < 0.2, $"LogicNetXOR SSE after {iterations} iterations is '{sse}'");
```
