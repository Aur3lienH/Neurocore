#pragma once

class LayerTests
{
public:
    static bool ExecuteTests();
private:
    static bool TestFCLLayer();
    static bool TestInputLayer();
    static bool TestCNNLayer();
};
