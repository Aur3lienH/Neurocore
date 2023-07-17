#include <iostream>
#include <chrono>
#pragma once

namespace Tools
{
    class ProgressBar
    {
    public:
        ProgressBar(std::string name);
        ~ProgressBar();
        void InitProgress();
        void EndProgress();
        int GetConsoleWidth();
        std::string name;
        float progress;
    protected:
        void PrintProgressPart(int size);
    };

    class ClasProBar : public ProgressBar
    {
    public:
        ClasProBar(std::string name, float maxValue);
        ~ClasProBar();
        void ChangeProgress(float value);
        void EndProgress();
        void InitProgress();
    private:
        void PrintProgressBar(float newValue);
        float maxValue;
        float progress;

    };



    class NetProBar : public ProgressBar
    {
    public:
        NetProBar(std::string name, int totalBytes);
        ~NetProBar();
        void ChangeProgress(uint64_t ByteSent);
        void EndProgress();
        void InitProgress();

    private:
        std::chrono::_V2::high_resolution_clock::time_point lastTime = std::chrono::high_resolution_clock::now();
        void PrintProgressBar(float newProgress);
        uint64_t byteDiff = 0;
        uint64_t bytesToDownload;
    };

    class TrainBar : public ProgressBar
    {
    public:
        TrainBar(int totalEpochs);
        void ChangeProgress(int EpochsDone, float loss);
    private:
        void Print();
        int epochs;
        float loss;
        int totalEpochs;
        std::chrono::high_resolution_clock::time_point startTime;
    
    };


}

