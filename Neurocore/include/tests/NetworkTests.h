#pragma once


class NetworkTests
{
public:
    static bool ExecuteTests();
    static bool BasicFFNFeedForward();
    static bool BasicFFNLearn();

    static bool CNNMaxPoolTest();

    static bool DataLoaderTest();
};