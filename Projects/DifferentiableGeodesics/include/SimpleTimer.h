#ifndef SIMPLE_TIMER_H
#define SIMPLE_TIMER_H
#include <iostream>
#include <chrono>


class SimpleTimer {
	
	
public:
	
	std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
	bool started = false;
	std::chrono::high_resolution_clock::duration elapsed = std::chrono::nanoseconds::zero();

public:
	
	SimpleTimer() {}
	
	SimpleTimer(bool start_on_construction) {
		if(start_on_construction) {
			start();
		}
	}
	
	void start()
	{
		if(started) {
			std::cerr << "Error: attempted to start running timer " << __FILE__ << ":" << __LINE__ << std::endl;
			exit(1);
		}
		else {
			start_time = std::chrono::high_resolution_clock::now();
			started = true;
		}

	}
	
	void stop()
	{
		if(started) {
			elapsed += std::chrono::high_resolution_clock::now() - start_time;
			started = false;
		}
		else {
			std::cerr << "Error: attempted to stop timer that is not running " << __FILE__ << ":" << __LINE__ << std::endl;
			exit(1);
		}
	}
	
	void reset() {
		elapsed = std::chrono::high_resolution_clock::duration::zero();
		started = false;
	}
	
	void restart() {
		reset();
		start();
	}
	
	double elapsed_sec() 
	{
		if(started) {
			return std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed + std::chrono::high_resolution_clock::now() - start_time).count() / 1.0e9;
		}
		else {
			return std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count() / 1.0e9;
		}
	}
	
};

#endif