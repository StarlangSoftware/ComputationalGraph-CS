// See https://aka.ms/new-console-template for more information

using Math;
using ComputationalGraph;

var graph = new Graph();
// input matrix with bias.
var input = new ComputationalNode(false, '*');
var m1 = new Matrix(3, 4, -0.01, 0.01, new Random(1));
var w1 = new ComputationalNode(m1, '*');
var a1 = graph.AddEdge(input, w1);
var a1Sigmoid = graph.AddEdge(a1, FunctionType.SIGMOID);
var m2 = new Matrix(4, 5, -0.01, 0.01, new Random(2));
var w2 = new ComputationalNode(m2, '*');
var a2 = graph.AddEdge(a1Sigmoid, w2);
var output = graph.AddEdge(a2, FunctionType.SIGMOID);
var epoch = 1000;
var learningRate = 0.01;
var classList = new List<int>();
for (var i = 0; i < epoch; i++) {
    // Creating input vector.
    input.SetValue(new Matrix(1, 3, 0, 100, new Random(1)));
    graph.ForwardCalculation();
    classList.Add(4);
    graph.Backpropagation(learningRate, classList);
    classList.Clear();
}
input.SetValue(new Matrix(1, 3, 0, 100, new Random(1)));
var classLabel = graph.ForwardCalculation()[0];
Console.WriteLine(classLabel == 4);
