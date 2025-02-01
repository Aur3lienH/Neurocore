#pragma once

#include <iostream>
#include <chrono>
#include <iomanip>
#include <cmath>
#include "tools/Unit.h"

#ifdef WIN32

#include <windows.h>

#else

#include <sys/ioctl.h>
#include <unistd.h>

#endif





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


namespace Tools
{
    ProgressBar::ProgressBar(const std::string& _name)
    {
        name = _name;

    }

    ProgressBar::~ProgressBar()
    {
        std::cout << "\n";
    }

    void ProgressBar::InitProgress()
    {
        std::cout << name << " 0%";
    }

    void ProgressBar::EndProgress()
    {
        std::cout << "\r\n";
    }

    inline int ProgressBar::GetConsoleWidth()
    {
        const char* ideTerminal = getenv("TERMINAL_EMULATOR");
        if (ideTerminal && std::string(ideTerminal).find("JetBrains") != std::string::npos) {
            // CLion typically sets COLUMNS environment variable
            const char* columns = getenv("COLUMNS");
            if (columns) {
                int width = std::atoi(columns);
                if (width > 0) {
                    return width;
                }
            }
            // If COLUMNS isn't set, CLion usually defaults to 80
            return 80;
        }
#ifdef WIN32
        CONSOLE_SCREEN_BUFFER_INFO csbi;
        GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &csbi);
        return csbi.srWindow.Right - csbi.srWindow.Left + 1;
#else
        struct winsize w{};
        ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);
        if (w.ws_col == 0)
            return 80;
        return w.ws_col;
#endif
    }

    void ProgressBar::PrintProgressPart(const int size) const
    {
        std::cout << "[";
        for (int i = 0; i < size - 2; i++)
        {
            float bar_pos = i / (float) (size - 2);
            if (bar_pos < progress)
                std::cout << "=";
            else if (bar_pos == progress)
                std::cout << ">";
            else
                std::cout << " ";
        }
        std::cout << "]";
    }


    ClasProBar::ClasProBar(std::string _name, const float _maxValue) : ProgressBar(_name)
    {
        maxValue = _maxValue;
    }

    ClasProBar::~ClasProBar()
    {
    }

    void ClasProBar::ChangeProgress(const float value)
    {
        PrintProgressBar(value);
    }

    void ClasProBar::PrintProgressBar(float newProgress)
    {
        newProgress = newProgress / maxValue;
        if ((int) (newProgress * 100) == (int) (progress * 100))
            return;
        std::cout << "\r";
        const int Width = GetConsoleWidth();
        const int nameLength = std::min((int) name.length(), Width - 10);
        const int ProgressBarWidth = Width - nameLength - 10;
        int progress = (int) newProgress * ProgressBarWidth;

        std::cout << name.substr(name.length() - nameLength, name.length()) << " : [";
        for (int i = 0; i < ProgressBarWidth; i++)
        {
            if (i < progress)
                std::cout << "=";
            else if (i == progress)
                std::cout << ">";
            else
                std::cout << " ";
        }
        std::cout << "] " << (int) (newProgress * 100) << "%";

        progress = newProgress;
        std::cout.flush();
    }

    void ClasProBar::EndProgress()
    {
        std::cout << "\n";
    }

    void ClasProBar::InitProgress()
    {
        PrintProgressBar(0);
    }


    NetProBar::NetProBar(const std::string& _name, int totalBytes) : ProgressBar(_name)
    {
        bytesToDownload = totalBytes;
        std::cout << std::fixed << std::setprecision(1);
    }

    NetProBar::~NetProBar()
    = default;

    void NetProBar::InitProgress()
    {
        PrintProgressBar(0);
    }

    void NetProBar::EndProgress()
    {
        std::cout << "\n";
    }

    //Change progress and print the progress bar and calculate the speed of transfer
    void NetProBar::ChangeProgress(uint64_t ByteSent)
    {
        PrintProgressBar(ByteSent / (float) bytesToDownload);
    }

    void NetProBar::PrintProgressBar(float newProgress)
    {
        byteDiff += 1024;
        if ((int) (newProgress * 100) == (int) (progress * 100))
            return;

        double TimeDiffS = std::chrono::duration<double, std::ratio<1>>(
                std::chrono::high_resolution_clock::now() - lastTime).count();
        double speed = byteDiff / TimeDiffS;
        lastTime = std::chrono::high_resolution_clock::now();

        std::cout << "\r";
        int Width = GetConsoleWidth();
        int nameLength = std::min((int) name.length(), Width - 10);
        int ProgressBarWidth = Width - nameLength - 18;
        int progress = (int) newProgress * ProgressBarWidth;

        std::cout << name.substr(name.length() - nameLength, name.length()) << " : [";
        for (int i = 0; i < ProgressBarWidth; i++)
        {
            if (i < progress)
                std::cout << "=";
            else if (i == progress)
                std::cout << ">";
            else
                std::cout << " ";
        }
        std::cout << "] " << (int) (newProgress * 100) << "%";

        Unit unit = Unit("B/s", speed);
        std::cout << unit;

        progress = newProgress;
        byteDiff = 0;
        std::cout.flush();
    }


    TrainBar::TrainBar(const int _totalEpochs) : ProgressBar("Train")
    {
        totalEpochs = _totalEpochs;
        startTime = std::chrono::high_resolution_clock::now();
    }

    void TrainBar::ChangeProgress(const int EpochsDone, const float _loss)
    {
        progress = static_cast<float>(EpochsDone) / static_cast<float>(totalEpochs);
        epochs = EpochsDone;
        loss = _loss;
        Print();
    }

    void TrainBar::Print()
    {
        std::cout << "\r";
        const int Width = GetConsoleWidth();
        auto now = std::chrono::high_resolution_clock::now();
        auto elapsed = now - startTime;
        auto hours = std::chrono::duration_cast<std::chrono::hours>(elapsed);
        elapsed -= hours;
        auto minutes = std::chrono::duration_cast<std::chrono::minutes>(elapsed);
        elapsed -= minutes;
        auto seconds = std::chrono::duration_cast<std::chrono::seconds>(elapsed);
        std::string beginning =
                "Train -> loss : " + std::to_string(loss) + " epoch : " + std::to_string(epochs) + " | " +
                std::to_string(hours.count()) + ":" + std::to_string(minutes.count()) + ":" +
                std::to_string(seconds.count()) + " ";
        int BarSize = std::min((unsigned long) (Width - beginning.size()), (unsigned long) 100);
        std::cout << beginning;
        int space = Width - BarSize - beginning.size();
        if (space < 0)
            return;


        PrintProgressPart(BarSize);
        std::cout.flush();
    }
}






