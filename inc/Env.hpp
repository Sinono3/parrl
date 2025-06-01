#pragma once
template <typename Obs> struct Step {
	Obs obs;
	float reward;
	bool done;
};

template <typename Obs, typename Action> class Env {
  public:
  	// Returns initial observation
	virtual Obs reset() = 0;
	virtual Step<Obs> step(Action action) = 0;
};
