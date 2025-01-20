#pragma once

class LayerTests
{
public:
    static bool ExecuteTests();
private:
    static bool TestFCLLayer();
    static bool TestInputLayer();
    static bool TestCNNLayer();
    static bool TestDropLayer();
    static bool TestDropLayerBackprop();
    static bool TestMaxPoolLayer();
    static bool TestAveragePoolLayer();
    static bool TestMaxPoolLayerBackprop();
    static bool TestAveragePoolLayerBackprop();
};
