#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include <torch/torch.h>
#pragma clang diagnostic pop

#include <SFML/Graphics/Sprite.hpp>

#include "CarRacing.hpp"
#include "PPO.hpp"
#include "util.hpp"

int main() {
    sf::RenderWindow window(sf::VideoMode({WINDOW_W, WINDOW_H}), "OpenGL", sf::Style::Default, sf::State::Windowed, sf::ContextSettings(32));
    window.setFramerateLimit(100);

	CarRacing env;
	auto state = imageToTensor(env.reset());
	
	ActorCritic policy(std::vector<int64_t>{3, 96, 96}, (int64_t)3, 0.01f);
    torch::load(policy, "checkpoint/model.pt");

	uint max_ep_len = 500;
   
    bool running = true;
	for (uint e = 1; (e <= 1000) && running; e++) {
		auto episode_reward = 0.0f;

		for (uint i = 0; i < max_ep_len; i++) {
	        // handle events
	        while (const std::optional event = window.pollEvent())
	        {
	            if (event->is<sf::Event::Closed>())
	            {
	                // end the program
	                running = false;
	            }
	        }

			auto [action, _action_logprob, _state_value] = policy->act(state);
			auto raw = action[0];
			float steer = raw[0].item<float>();
			float gas = raw[1].item<float>();
			float brake = raw[2].item<float>();
			auto [stateImage, reward, done] = env.step({steer, gas, brake});
			state = imageToTensor(stateImage);

			episode_reward += reward;

			// Show to human
	        sf::Sprite stateSprite(env.stateTexture.getTexture());
			stateSprite.setScale(sf::Vector2f((float)WINDOW_W / (float)STATE_W,
											  (float)WINDOW_H / (float)STATE_H));
			window.draw(stateSprite);
			window.display();

			if (done) {
			    auto img = env.reset();
			    state = imageToTensor(img);
			}
		}
		env.reset();

		std::println("Total episode reward: {}", episode_reward);
	}
	return 0;
}
