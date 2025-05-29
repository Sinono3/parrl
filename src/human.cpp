#include <SFML/Graphics/Color.hpp>
#include <SFML/Graphics/RenderWindow.hpp>
#include <SFML/Graphics/Shape.hpp>
#include <SFML/Graphics/StencilMode.hpp>
#include <SFML/System/Angle.hpp>
#include <SFML/Window/Keyboard.hpp>
#include <SFML/Window/Window.hpp>
#include <SFML/Window.hpp>
#include <SFML/OpenGL.hpp>
#include <SFML/Graphics.hpp>
#include <span>
#include <box2d/box2d.h>
#include <box2d/math_functions.h>
#include <box2d/types.h>

#include "Car.hpp"
#include "CarRacing.hpp"

int main()
{
    // create the window
    sf::RenderWindow window(sf::VideoMode({WINDOW_W, WINDOW_H}), "OpenGL", sf::Style::Default, sf::State::Windowed, sf::ContextSettings(32));
    window.setFramerateLimit(60);

    CarRacing env;
    env.reset();

    // run the main loop
    bool running = true;
    while (running)
    {
        // handle events
        while (const std::optional event = window.pollEvent())
        {
            if (event->is<sf::Event::Closed>())
            {
                // end the program
                running = false;
            }
        }

        float gas = 0.0f, brake = 0.0f;
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::W)) {
            gas = 1.0f;
        }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::S)) {
            brake = 1.0f;
        }

        float steer = 0.0f;
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::A))
            steer = -1.0f;
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::D))
            steer = 1.0f;

        auto [state, step_reward, done] = env.step({steer, gas, brake});
        std::println("state=(image), reward={}, done={}", step_reward, done);

        sf::Sprite stateSprite(env.stateTexture.getTexture());
		stateSprite.setScale(sf::Vector2f((float)WINDOW_W / (float)STATE_W,
										  (float)WINDOW_H / (float)STATE_H));
		window.draw(stateSprite);
		window.display();

        if (done)
            env.reset();
	}
}
