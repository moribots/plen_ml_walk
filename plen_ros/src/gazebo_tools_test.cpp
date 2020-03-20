#include <gazebo/gazebo.hh>
#include <gazebo/gazebo_client.hh>
#include <gazebo/common/Plugin.hh>
#include <gazebo/msgs/msgs.hh>
#include <gazebo/transport/transport.hh>
#include <ros/ros.h>
#include <std_srvs/Empty.h>
#include <functional>  // To use std::bind
#include "plen_ros/Iterate.h"


bool iterateCallback(plen_ros::Iterate::Request& req, plen_ros::Iterate::Response& res, const gazebo::transport::PublisherPtr& pub)
{
  // Create Iterate msg
  gazebo::msgs::WorldControl stepper;
  // Set multi-step to requested iterations
  stepper.set_multi_step(req.iterations);
  pub->Publish(stepper);

  res.result = true;

  return true;
}

int main(int argc, char **argv)
{
gazebo::client::setup(argc,argv);

gazebo::transport::NodePtr node(new gazebo::transport::Node());
node->Init();

ros::init(argc, argv, "gazebo_iterator"); // register the node on ROS
ros::NodeHandle nh; // get a handle to ROS

//Create publisher for topic ~/world_control
gazebo::transport::PublisherPtr pub = node->Advertise<gazebo::msgs::WorldControl>("~/world_control");
//Publish to topic ~/world_control
pub->WaitForConnection();

// Create Iterate Service Server
ros::ServiceServer iter_server = nh.advertiseService<plen_ros::Iterate::Request,
                                    plen_ros::Iterate::Response>("/iterate", std::bind(&iterateCallback,
                                    						   std::placeholders::_1, 
                                    						   std::placeholders::_2,
                                    						   pub));

//Publish 1st step

// pub->Publish(stepper, true);

while (ros::ok())
  {
  	ros::spinOnce();
  }
return 0;
} // end main()