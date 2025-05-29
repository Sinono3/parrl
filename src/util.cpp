#include "util.hpp"

sf::ConvexShape arrayToShape(std::span<const b2Vec2> a, float scale) {
    sf::ConvexShape shape;
    shape.setPointCount(a.size());
    for (int i = 0; i < a.size(); i++)
        shape.setPoint(i, {a[i].x * scale, a[i].y * scale});

    return shape;
}
torch::Tensor imageToTensor(const sf::Image& img)  {
    auto size = img.getSize();
	auto rgbaBuf = img.getPixelsPtr();
	auto t = torch::from_blob((void *)rgbaBuf, {size.x, size.y, 4}, torch::kUInt8);
	using namespace torch::indexing;
	// Discard alpha dimension
	t = t.to(torch::kF32, false, true) / 255.0;
	t = t.index({Slice(), Slice(), Slice(0, 3)});
	t = t.permute({2, 0, 1}).unsqueeze(0);
	// Convert to float [0-1] range
	return t;
}

