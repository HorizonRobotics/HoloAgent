"""
LICENSE

This project as a whole is licensed under the Apache License, Version 2.0.

THIRD-PARTY LICENSES

Third-party software already included in HoloAgent is governed by the separate 
Open Source license terms under which the third-party software has been distributed.

NOTICE ON LICENSE COMPATIBILITY FOR DISTRIBUTORS

Notably, this project depends on the third-party software FAST-LIVO2 and HOVSG. 
Their default licenses restrict commercial use—separate permission from their 
original authors is required for commercial integration/redistribution.

The third-party software FAST-LIVO2 dependency (licensed under GPL-2.0-only) 
utilizes rpg_vikit-ros2 which contains components under the GPL-3.0. Please be 
aware of license compatibility when distributing a combined work.

DISCLAIMER

Users are solely responsible for ensuring compliance with all applicable license 
terms when using, modifying, or distributing the project. Project maintainers 
accept no liability for any license violations arising from such use.
"""
#include <unitree/robot/g1/loco/g1_loco_api.hpp>
#include <unitree/robot/g1/loco/g1_loco_client.hpp>
#include <iostream>
#include <fstream>
#include <unistd.h>

struct Vel
{
  float x;
  float y;
  float r;
  Vel() : x(0.0f), y(0.0f), r(0.0f) {}
  Vel(float a, float b, float c) : x(a), y(b), r(c) {}
};
int main(int argc, char * argv[])
{

  unitree::robot::ChannelFactory::Instance()->Init(0,"eth0");
  unitree::robot::g1::LocoClient client;
  client.Init();
  client.SetTimeout(10.f);

  std::ifstream velpipe("/tmp/vel_fifo", std::ios::binary);
  Vel value;
  while (true) {
      //std::cout<<"move"<<std::endl;
      //client.Move(0.2, 0.0, 0.0);
      
      velpipe.read(reinterpret_cast<char*>(&value), sizeof(Vel));
      if (velpipe.gcount() == sizeof(Vel)) {
            
            if(value.x==0.0 && value.y==0)
            {
                if(value.r>0.0 && value.r<0.3)
                {
                    value.r = 0.3;
                }
                else
                {
                    if(value.r<0.0 && value.r>-0.3)
                    {
                        value.r = -0.3;
                    }
                }
                
            }
            else
            {
                if(value.r<0.1 && value.r>0.0)
                {
                    value.r = 0.1;
                }
                else
                {
                    if(value.r>-0.1 && value.r<0.0)
                    {
                        value.r = -0.1;
                    }
                }
                
            }

	    if(value.r<-0.3 || value.r>0.3)
	    {
		    if(value.x>0.22)
		    {
		    value.x = 0.22;
		    }
	    }
            std::cout << "Read: " << value.x <<","<<value.y<<","<<value.r << std::endl;
             client.Move(value.x, value.y, value.r);
            //client.Damp();
      } else {
          //std::cout<<"no data"<<std::endl;
          usleep(10000);  // 短暂休眠
      }
      
  }
  
  return 0;
}
