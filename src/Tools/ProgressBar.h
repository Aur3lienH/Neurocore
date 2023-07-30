#include <iostream>
#include <chrono>

#pragma once

namespace Tools
{
    class ProgressBar
    {
    public:
        explicit ProgressBar(const std::string& _name);

        ~ProgressBar();

        virtual void InitProgress();

        virtual void EndProgress();

        static int GetConsoleWidth();

        std::string name;
        float progress;
    protected:
        void PrintProgressPart(int size) const;
    };

    class ClasProBar : public ProgressBar
    {
    public:
        ClasProBar(std::string name, float maxValue);

        ~ClasProBar();

        void ChangeProgress(float value);

        void EndProgress() override;

        void InitProgress() override;

    private:
        void PrintProgressBar(float newValue);

        float maxValue;
        float progress;

    };


    class NetProBar : public ProgressBar
    {
    public:
        NetProBar(const std::string& _name, int totalBytes);

        ~NetProBar();

        void ChangeProgress(uint64_t ByteSent);

        void EndProgress() override;

        void InitProgress() override;

    private:
        std::chrono::_V2::high_resolution_clock::time_point lastTime = std::chrono::high_resolution_clock::now();

        void PrintProgressBar(float newProgress);

        uint64_t byteDiff = 0;
        uint64_t bytesToDownload;
    };

    class TrainBar : public ProgressBar
    {
    public:
        explicit TrainBar(int totalEpochs);

        void ChangeProgress(int EpochsDone, float loss);

    private:
        void Print();

        int epochs;
        float loss;
        int totalEpochs;
        std::chrono::high_resolution_clock::time_point startTime;

    };


}

