#!/usr/bin/env python

import rospy
import time
from controller_manager_msgs.srv import SwitchController, SwitchControllerRequest
from controller_manager_msgs.srv import LoadController, UnloadController


class ControllersConnection():
    def __init__(self, namespace, controllers_list):

        rospy.logwarn("Start Init ControllersConnection")
        self.controllers_list = controllers_list
        self.controller_list = [
            "joint_state_controller", "joint_trajectory_controller"
        ]
        self.switch_service_name = '/' + namespace + '/controller_manager/switch_controller'
        self.switch_service = rospy.ServiceProxy(self.switch_service_name,
                                                 SwitchController)
        self.unload_service = rospy.ServiceProxy(
            '/' + namespace + '/controller_manager/unload_controller',
            UnloadController)
        self.load_service = rospy.ServiceProxy(
            '/' + namespace + '/controller_manager/load_controller',
            LoadController)
        rospy.logwarn("END Init ControllersConnection")
        self.namespace = namespace

    def unload_controllers(self):
        rospy.logwarn('waiting for /' + self.namespace +
                      '/controller_manager/unload_controller')
        rospy.wait_for_service('/' + self.namespace +
                               '/controller_manager/unload_controller')

        rospy.logwarn("SWITCHING OFF, THEN UNLOADING")
        for controller in self.controller_list:
            # Switch Controllers Off
            self.switch_controllers(controllers_on=[],
                                    controllers_off=controller)
            # Unload Controllers
            self.unload_service(controller)

    def load_controllers(self):
        rospy.wait_for_service('/' + self.namespace +
                               '/controller_manager/load_controller')

        rospy.logwarn("LOADING, THEN SWITCHING ON")
        for controller in self.controller_list:
            # Load Controllers
            self.load_service(controller)
            # Switch Controllers On
            self.switch_controllers(controllers_on=controller,
                                    controllers_off=[])

    def reload_controllers(self):
        self.unload_controllers()
        self.load_controllers()

    def switch_controllers(self, controllers_on, controllers_off,
                           strictness=1):
        """
        Give the controllers you want to switch on or off.
        :param controllers_on: ["name_controler_1", "name_controller2",...,"name_controller_n"]
        :param controllers_off: ["name_controler_1", "name_controller2",...,"name_controller_n"]
        :return:
        """
        rospy.wait_for_service(self.switch_service_name)

        try:
            switch_request_object = SwitchControllerRequest()
            switch_request_object.start_controllers = controllers_on
            switch_request_object.start_controllers = controllers_off
            switch_request_object.strictness = strictness
            switch_request_object.start_asap = False

            switch_result = self.switch_service(switch_request_object)
            """
            [controller_manager_msgs/SwitchController]
            int32 BEST_EFFORT=1
            int32 STRICT=2
            string[] start_controllers
            string[] stop_controllers
            int32 strictness
            ---
            bool ok
            """
            rospy.loginfo("Switch Result==>" + str(switch_result.ok))

            return switch_result.ok

        except rospy.ServiceException as e:
            print(self.switch_service_name + " service call failed")

            return None

    def reset_controllers(self):
        """
        We turn on and off the given controllers
        :param controllers_reset: ["name_controler_1", "name_controller2",...,"name_controller_n"]
        :return:
        """
        reset_result = False

        result_off_ok = self.switch_controllers(
            controllers_on=[], controllers_off=self.controllers_list)

        rospy.logdebug("Deactivated Controlers")

        if result_off_ok:
            rospy.logdebug("Activating Controlers")
            result_on_ok = self.switch_controllers(
                controllers_on=self.controllers_list, controllers_off=[])
            if result_on_ok:
                rospy.logdebug("Controllers Reseted==>" +
                               str(self.controllers_list))
                reset_result = True
            else:
                rospy.logdebug("result_on_ok==>" + str(result_on_ok))
        else:
            rospy.logdebug("result_off_ok==>" + str(result_off_ok))

        return reset_result