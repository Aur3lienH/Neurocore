//
// Created by mat on 04/09/23.
//
#include <SFML/Graphics.hpp>
#include <vector>
#include "Network.cuh"
#include "InputLayer.cuh"
#include "FCL.cuh"
#include "Tools.cuh"
#include "Mnist.cuh"
#include "Quickdraw.cuh"
#include "MaxPooling.cuh"
#include "ConvLayer.cuh"
#include "Flatten.cuh"
#include <thread>
#include "CUDA.cuh"

/*float RGB2Gray(sf::Color color)
{
    return 0.2126f * color.r + 0.7152f * color.g + 0.0722f * color.b;
}*/

sf::Color hsv(int hue, float sat, float val)
{
    hue %= 360;
    while (hue < 0) hue += 360;

    if (sat < 0.f) sat = 0.f;
    if (sat > 1.f) sat = 1.f;

    if (val < 0.f) val = 0.f;
    if (val > 1.f) val = 1.f;

    int h = hue / 60;
    float f = float(hue) / 60 - h;
    float p = val * (1.f - sat);
    float q = val * (1.f - sat * f);
    float t = val * (1.f - sat * (1 - f));

    switch (h)
    {
        default:
        case 0:
        case 6:
            return sf::Color(val * 255, t * 255, p * 255);
        case 1:
            return sf::Color(q * 255, val * 255, p * 255);
        case 2:
            return sf::Color(p * 255, val * 255, t * 255);
        case 3:
            return sf::Color(p * 255, q * 255, val * 255);
        case 4:
            return sf::Color(t * 255, p * 255, val * 255);
        case 5:
            return sf::Color(val * 255, p * 255, q * 255);
    }
}


void UpdateGuessTexts(const sf::Texture& text, Network& network, sf::Text* guessTexts)
{
    const int dataLength = 28 * 28;
    const sf::Image tmp = text.copyToImage();
    const sf::Uint8* pixels = tmp.getPixelsPtr();
    float* input = new float[dataLength];
    for (int i = 0; i < dataLength; i++)
        input[i] = pixels[i * 4] / 255.0f;
    //input[i] = RGB2Gray(sf::Color(pixels[i * 4], pixels[i * 4 + 1], pixels[i * 4 + 2]));

#if USE_GPU
    MAT* inputMat = new MAT(dataLength, 1);
    checkCUDA(cudaMemcpy(inputMat->GetData(), input, dataLength * sizeof(float), cudaMemcpyHostToDevice));
#else
    MAT* inputMat = new MAT(dataLength, 1, input);
#endif
    const MAT* output = network.FeedForward(inputMat);

    for (int i = 0; i < 10; i++)
    {
#if USE_GPU
        const float percentage = output->GetAt(i);
#else
        const float percentage = (int) (output->GetData()[i] * 100);
#endif
        guessTexts[i].setFillColor(
                hsv(static_cast<int>((1 - percentage) * 60.f), percentage, 1) /*sf::Color(percentage, 0, 0, 255)*/);
        guessTexts[i].setString(std::to_string(i) + " : " + std::to_string(percentage) + "%");
    }
}

int GUI()
{
    const int dataLength = CSVTools::CsvLength("../datasets/mnist/mnist_train.csv");

    MAT*** data = GetDataset("../datasets/mnist/mnist_train.csv", dataLength, false);

    std::cout << "Data length: " << dataLength << std::endl;

    const float scale = 1.0f / 255.0f;
    for (int i = 0; i < dataLength; i++)
        *data[i][0] *= scale;

    Network* network = new Network();
    network->AddLayer(new InputLayer(784));
    network->AddLayer(new FCL(512, new ReLU()));
    network->AddLayer(new FCL(10, new Softmax()));
    std::cout << "before compiling !\n";
    network->Compile(Opti::Adam, new CrossEntropy());
    std::cout << "compiled ! \n";
    const int trainLength = dataLength * .4f;

#if USE_GPU
    network->Learn(1, 0.01, new DataLoader(data, trainLength), 128, 1);
#else
    const int numThreads = std::thread::hardware_concurrency();
    network->Learn(1, 0.01, new DataLoader(data, trainLength), 128, numThreads);
#endif

    double trainingAccuracy = TestAccuracy(network, data, 1000);
    std::cout << "Training Accuracy : " << trainingAccuracy * 100 << "% \n";
    double testingAccuracy = TestAccuracy(network, data + trainLength, 1000);
    std::cout << "Testing Accuracy : " << testingAccuracy * 100 << "% \n";

    /*const int numDrawingsPerCategory = 10000;
    std::cout << "quickdraw 1\n";
    std::pair<int, int> dataInfo = GetDataLengthAndNumCategories("../datasets/Quickdraw", numDrawingsPerCategory);
    const int dataLength = dataInfo.first;
    const int numCategories = dataInfo.second;

    MAT*** data = GetQuickdrawDataset("../datasets/Quickdraw", dataLength, numCategories, numDrawingsPerCategory,
                                      true);
    std::cout << "loaded" << std::endl;

    auto* network = new Network();
    network->AddLayer(new InputLayer(28, 28, 1));
    network->AddLayer(new ConvLayer(new LayerShape(3, 3, 32), new ReLU()));
    network->AddLayer(new MaxPoolLayer(2, 2));
    network->AddLayer(new Flatten());
    network->AddLayer(new FCL(numCategories, new Softmax()));

    network->Compile(Opti::Adam, new CrossEntropy());

    network->PrintNetwork();
    const int trainLength = dataLength * 0.8;
    //const int testLength = dataLength - trainLength;
    auto* dataLoader = new DataLoader(data, trainLength);
    dataLoader->Shuffle();
    network->Learn(1, 0.01, dataLoader, 96, 1);

    double trainingAccuracy = TestAccuracy(network, data, 1000);
    std::cout << "Training Accuracy : " << trainingAccuracy * 100 << "% \n";
    double testingAccuracy = TestAccuracy(network, data + trainLength, 1000);
    std::cout << "Testing Accuracy : " << testingAccuracy * 100 << "% \n";*/

    sf::Font font;
    if (!font.loadFromFile("../Fonts/LemonMilk/LEMONMILK-Medium.otf"))
        return EXIT_FAILURE;

    sf::Text* texts = new sf::Text[10];
    for (int i = 0; i < 10; i++)
    {
        sf::Text* text = &texts[i];
        text->setFont(font);
        text->setCharacterSize(24);
        text->setPosition(600.f, 24.f * static_cast<float>(i));
    }

    sf::Texture texture;
    sf::RenderWindow window(sf::VideoMode(800, 600), L"SFML Drawing – C to clear, PageUp/PageDown to pick colors",
                            sf::Style::Default);
    texture.create(window.getSize().x, window.getSize().y);
    // Set a specific frame rate, since we don't want to/
    // worry about vsync or the time between drawing iterations
    window.setVerticalSyncEnabled(false);
    window.setFramerateLimit(100);

    // First we'll use a canvas to basically store our image
    sf::RenderTexture canvas;
    canvas.create(28, 28);
    canvas.clear(sf::Color::Black);

    // Next we'll need a sprite as a helper to draw our canvas
    sf::Sprite sprite;
    sprite.setTexture(canvas.getTexture(), true);
    //sprite.setScale(600 / 28.f, 600 / 28.f); // Scale it to fit the window
    sprite.setScale(600 / 28.f, 600 / 28.f);

    // Define some colors to use
    // These are all with very low alpha so we
    // can (over-)draw based on how fast we move the cursor
    /*const std::vector<sf::Color> colors = {
            sf::Color(255, 0, 0, 8),
            sf::Color(255, 255, 0, 8),
            sf::Color(0, 255, 0, 8),
            sf::Color(0, 255, 255, 8),
            sf::Color(0, 0, 255, 8),
            sf::Color(255, 0, 255, 8)
    };
     */
    const std::vector<sf::Color> colors = {
            sf::Color(255, 255, 255, 8)
    };
    // We'll need something to actually draw
    // For simplicity, I'm just drawing a circle shape
    // but you could also draw a line, rectangle, or something more complex
    const float brush_size = 2;
    sf::CircleShape brush(brush_size, 24);
    brush.setOrigin(brush_size, brush_size); // Center on the circle's center

    sf::Vector2f lastPos;
    bool isDrawing = false;
    unsigned int color = 0;

    // Apply some default color
    brush.setFillColor(colors[color]);

    int c = 0;
    while (window.isOpen())
    {
        sf::Event event{};
        while (window.pollEvent(event))
        {
            switch (event.type)
            {
                case sf::Event::Closed:
                    window.close();
                    break;
                case sf::Event::KeyPressed:
                    switch (event.key.code)
                    {
                        case sf::Keyboard::C:
                            // Clear our canvas
                            canvas.clear(sf::Color::Black);
                            canvas.display();
                            break;
                        case sf::Keyboard::PageUp:
                            // Get next color
                            color = ++color % colors.size();
                            // Apply it
                            brush.setFillColor(colors[color]);
                            break;
                        case sf::Keyboard::PageDown:
                            // Get previous color
                            color = --color % colors.size();
                            // Apply it
                            brush.setFillColor(colors[color]);
                            break;
                    }
                    break;
                case sf::Event::Resized:
                {
                    // Window got resized, update the view to the new size
                    sf::View view(window.getView());
                    const sf::Vector2f size(window.getSize().x, window.getSize().y);
                    view.setSize(size); // Set the size
                    view.setCenter(size / 2.f); // Set the center, moving our drawing to the top left
                    window.setView(view); // Apply the view
                    break;
                }
                case sf::Event::MouseButtonPressed:
                    // Only care for the left button
                    if (event.mouseButton.button == sf::Mouse::Left)
                    {
                        isDrawing = true;
                        // Store the cursor position relative to the canvas
                        lastPos = window.mapPixelToCoords({event.mouseButton.x, event.mouseButton.y});

                        // Now let's draw our brush once, so we can
                        // draw dots without actually dragging the mouse
                        brush.setPosition(lastPos);

                        // Draw our "brush"
                        canvas.draw(brush, sf::BlendNone);

                        // Finalize the texture
                        canvas.display();
                    }
                    break;
                case sf::Event::MouseButtonReleased:
                    // Only care for the left button
                    if (event.mouseButton.button == sf::Mouse::Left)
                        isDrawing = false;
                    break;
                case sf::Event::MouseMoved:
                    if (isDrawing)
                    {
                        // Calculate the cursor position relative to the canvas
                        const sf::Vector2f newPos(
                                window.mapPixelToCoords(sf::Vector2i(event.mouseMove.x, event.mouseMove.y)));

                        const sf::Vector2f np(newPos.x * 28 / 600.f, newPos.y * 28 / 600.f);
                        // I'm only using the new position here
                        // but you could also use `lastPos` to draw a
                        // line or rectangle instead
                        brush.setPosition(np);

                        // Draw our "brush"
                        canvas.draw(brush);

                        // Finalize the texture
                        canvas.display();
                        break;
                    }
            }
        }

        // Clear the window
        window.clear(sf::Color(64, 64, 64));

        // Draw our canvas
        window.draw(sprite);
        for (int i = 0; i < 10; i++)
            window.draw(texts[i]);
        // Show the window
        window.display();

        if (++c == 10)
        {
            UpdateGuessTexts(canvas.getTexture(), *network, texts);
            c = 0;
        }
    }

    delete[] texts;

    return 0;
}