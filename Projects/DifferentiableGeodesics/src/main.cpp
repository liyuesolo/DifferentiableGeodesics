#include "../include/App.h"
#include "../include/Scheduler.h"

#include <igl/default_num_threads.h>
int main(int argc, char** argv)
{
    int n_thread = igl::default_num_threads(1);
    
    std::string exp = "shell_tightening";
    Scheduler scheduler;
    scheduler.exp = exp;
    scheduler.base_folder = 
        "../../../Projects/DifferentiableGeodesics/";   
    scheduler.execute(/*run without GUI = */false, /*save result = */true);
    return 0;
}
