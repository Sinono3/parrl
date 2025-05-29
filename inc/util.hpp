#pragma once
#include <SFML/Graphics/ConvexShape.hpp>
#include <SFML/Graphics/Image.hpp>
#include <box2d/math_functions.h>
#include <span>
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include <torch/torch.h>
#pragma clang diagnostic pop

sf::ConvexShape arrayToShape(std::span<const b2Vec2> a, float scale = 1.0f);
torch::Tensor imageToTensor(const sf::Image& img);

