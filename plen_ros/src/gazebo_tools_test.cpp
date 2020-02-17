#include <gazebo/gazebo.hh>
#include <gazebo/gazebo_client.hh>
#include <gazebo/common/Plugin.hh>
#include <gazebo/msgs/msgs.hh>
#include <gazebo/transport/transport.hh>

static bool stop_simulation;

void rosShutdownHandler(int sig)
{
stop_simulation = true;
}

int main(int argc, char **argv)
{
gazebo::client::setup(argc,argv);

gazebo::transport::NodePtr node(new gazebo::transport::Node());
node->Init();

//Create publisher for topic ~/world_control
gazebo::transport::PublisherPtr pub = node->Advertise<gazebo::msgs::WorldControl>("~/world_control");
//Create message
gazebo::msgs::WorldControl msg_step;

stop_simulation = false;


//Publish to topic ~/world_control
pub->WaitForConnection();

//Set step to true
// msg_step.set_step(1);
msg_step.set_multi_step(100);

//Publish 1st step
pub->Publish(msg_step);
// pub->Publish(msg_step, true);

while (!stop_simulation){
    std::cout << "Running \n";
    // pub->Publish(msg_step, 100);
}

gazebo::shutdown();
return 0;
} // end main()