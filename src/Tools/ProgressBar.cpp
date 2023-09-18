#include "ProgressBar.h"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <cmath>
#include "Unit.h"

#ifdef WIN32

#include <windows.h>

#else

#include <sys/ioctl.h>
#include <unistd.h>

#endif


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

    int ProgressBar::GetConsoleWidth()
    {
#ifdef WIN32
        CONSOLE_SCREEN_BUFFER_INFO csbi;
        GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &csbi);
        return csbi.srWindow.Right - csbi.srWindow.Left + 1;
#else
        struct winsize w{};
        ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);
        return w.ws_col;
#endif
    }

    void ProgressBar::PrintProgressPart(const int size) const
    {
        int newProgress = (int) (progress * 100);
        std::cout << "[";
        for (int i = 0; i < size - 2; i++)
        {
            if (i < newProgress)
                std::cout << "=";
            else if (i == newProgress)
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
        progress = EpochsDone / (float) totalEpochs;
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
        unsigned int space = Width - BarSize - beginning.size();
        for (unsigned int i = 0; i < space; i++)
        {
            std::cout << " ";
        }

        PrintProgressPart(BarSize);
        std::cout.flush();
    }
}








    
