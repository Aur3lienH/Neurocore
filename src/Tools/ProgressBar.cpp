#include "ProgressBar.h"
#include <iostream>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <cmath>
#include <limits>
#include "Unit.h"

#ifdef WIN32

#include <windows.h>

#else
#include <sys/ioctl.h>
#include <stdio.h>
#include <unistd.h>
#endif



namespace Tools 
{
    ProgressBar::ProgressBar(std::string _name)
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
        struct winsize w;
        ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);
        return w.ws_col;
    #endif
    }

    void ProgressBar::PrintProgressPart(int size)
    {
        int newProgress = (int)(progress * 100);
        std::cout << "[";
        for (int i = 0; i < size - 2; i++)
        {
            if(i < newProgress)
                std::cout << "=";
            else if(i == newProgress)
                std::cout << ">";
            else
                std::cout << " ";
        }
        std::cout << "]";
    }



    ClasProBar::ClasProBar(std::string _name, float _maxValue) : ProgressBar(_name)
    {
        maxValue = _maxValue;
    }
    ClasProBar::~ClasProBar()
    {
    }
    void ClasProBar::ChangeProgress(float value)
    {
        PrintProgressBar(value);
    }
    void ClasProBar::PrintProgressBar(float newProgress)
    {
        newProgress = newProgress / maxValue;
        if((int)(newProgress * 100) == (int)(progress * 100))
            return;
        std::cout << "\r";
        int Width = GetConsoleWidth();
        int nameLength = std::min((int)name.length(), Width - 10);
        int ProgressBarWidth = Width - nameLength - 10;
        int progress = (int)(newProgress * ProgressBarWidth);

        std::cout << name.substr(name.length() - nameLength, name.length()) << " : [";
        for (int i = 0; i < ProgressBarWidth; i++)
        {
            if(i < progress)
                std::cout << "=";
            else if(i == progress)
                std::cout << ">";
            else
                std::cout << " ";
        }
        std::cout << "] " << (int)(newProgress *100) << "%";

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


    NetProBar::NetProBar(std::string _name, int totalBytes) : ProgressBar(_name)
    {
        bytesToDownload = totalBytes;
        std::cout << std::fixed << std::setprecision(1);
    }
    NetProBar::~NetProBar()
    {
        
    }
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
        PrintProgressBar(ByteSent / (float)bytesToDownload);
    }
    void NetProBar::PrintProgressBar(float newProgress)
    {
        byteDiff += 1024;
        if((int)(newProgress * 100) == (int)(progress * 100))
            return;

        double TimeDiffS = std::chrono::duration<double,std::ratio<1>>(std::chrono::high_resolution_clock::now() - lastTime).count();
        double speed = byteDiff / TimeDiffS;
        lastTime = std::chrono::high_resolution_clock::now();

        std::cout << "\r";
        int Width = GetConsoleWidth();
        int nameLength = std::min((int)name.length(), Width - 10);
        int ProgressBarWidth = Width - nameLength - 18;
        int progress = (int)(newProgress * ProgressBarWidth);

        std::cout << name.substr(name.length() - nameLength, name.length()) << " : [";
        for (int i = 0; i < ProgressBarWidth; i++)
        {
            if(i < progress)
                std::cout << "=";
            else if(i == progress)
                std::cout << ">";
            else
                std::cout << " ";
        }
        std::cout << "] " << (int)(newProgress *100) << "%";
        
        Unit unit = Unit("B/s", speed);
        std::cout << unit;
        
        progress = newProgress;
        byteDiff = 0;
        std::cout.flush();
    }


    TrainBar::TrainBar(int _totalEpochs) : ProgressBar("Train")
    {
        totalEpochs = _totalEpochs;
    }

    void TrainBar::ChangeProgress(int EpochsDone, float _loss)
    {
        progress = EpochsDone / (float)totalEpochs;
        epochs = EpochsDone;
        loss = _loss;
        Print();
    }

    void TrainBar::Print()
    {
        std::cout << "\r";
        int Width = GetConsoleWidth();
        std::string beginning =  "Train -> loss : " + std::to_string(loss) + " epoch : " +  std::to_string(epochs) + " ";
        int BarSize = std::min(Width - beginning.size(),(unsigned long)100);
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








    
