#!/usr/bin/env python
#import roslib; 
#roslib.load_manifest('gazebo')

import sys
import rospy
import os
import time
 
from gazebo_msgs.msg import *
from gazebo_msgs.srv import *
from geometry_msgs.msg import Point, Pose, Quaternion, Twist, Wrench

from tf import transformations

def gms_client(command, model_name,relative_entity_name):
    if command == 1:
        rospy.wait_for_service('/gazebo/get_model_state')
        try:
            gms = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
            resp1 = gms(model_name,relative_entity_name)
            return resp1
        except rospy.ServiceException, e:
            print "Service call failed: %s"%e
    if command == 2:
        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            model_state = ModelState()
            model_state.model_name = "pioneer3at"
            model_state.pose.position.x = 0
            model_state.pose.position.y = 0
            model_state.pose.position.z = 0
            qto = transformations.quaternion_from_euler(0, 0, 0, axes='sxyz')
            model_state.pose.orientation.x = qto[0]
            model_state.pose.orientation.y = qto[1]
            model_state.pose.orientation.z = qto[2]
            model_state.pose.orientation.w = qto[3]
            model_state.twist.linear.x = 0
            model_state.twist.linear.y = 0
            model_state.twist.linear.z = 0
            model_state.twist.angular.x = 0
            model_state.twist.angular.y = 0
            model_state.twist.angular.z = 0
            gms = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            resp1 = gms(model_state)
            return resp1
        except rospy.ServiceException, e:
            print "Service call failed: %s"%e
        

def usage():
    return "%s [model_name] [relative_entity_name]"%sys.argv[0]

if __name__ == "__main__":
    if len(sys.argv) == 3:
        model_name = sys.argv[1]
        relative_entity_name = sys.argv[2]
    else:
        print usage()
        sys.exit(1)
    res = gms_client(2,model_name,relative_entity_name)
    print "mode state info: ",res
