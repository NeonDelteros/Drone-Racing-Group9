#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from tello_msgs.srv import TelloAction
from cv_bridge import CvBridge
import time
import cv2
import numpy as np

class GatePilot(Node):
    def __init__(self):
        super().__init__('gate_pilot')
        self.bridge = CvBridge()
        
        # Land detection variables
        self.last_gate_time = time.time()  # Initialize timer
        self.landed = False

        # Existing publisher and service client
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.tello_client = self.create_client(TelloAction, '/tello_action')
        while not self.tello_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warning('Waiting for service...')
        
        # Takeoff
        req = TelloAction.Request()
        req.cmd = 'takeoff'
        self.tello_client.call_async(req)
        self.took_off = True

        # Camera subscription
        self.sub = self.create_subscription(
            Image, '/image_raw', self.cb_image, qos_profile_sensor_data)

        # Existing HSV ranges and colors
        self.hsv_ranges = {
            'green': (np.array((30,  60,  60)), np.array((90, 255, 255))),
            'blue':  (np.array((90,  50,  60)), np.array((140,255,255))),
            'red1':  (np.array((0,   60,  60)), np.array((10, 255,255))),
            'red2':  (np.array((170, 60,  60)), np.array((180,255,255))),
        }
        self.colors = {'green': (0,255,0), 'blue': (255,0,0), 'red': (0,0,255)}
        
        # Control params
        self.center_x = None
        self.center_y = None
        self.kp_yaw = 0.002
        self.kp_alt = 0.001
        self.forward_speed = 0.3  # Check if drone moves backward
        self.pass_through_frames = 0
        self.PASS_FRAMES_MAX = 25
        
        # Ascension phase variables
        self.initial_ascent_done = False
        self.initial_ascent_start = None 

    def preprocess(self, mask):
        k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
        m = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k1, iterations=2)
        return cv2.morphologyEx(m, cv2.MORPH_CLOSE, k2, iterations=2)

    def cb_image(self, msg):
        if not self.took_off or self.landed:
            return
            
        # Ascent a little at the start
        if not self.initial_ascent_done:
            if self.initial_ascent_start is None:
                self.initial_ascent_start = time.time()
            else:
                if time.time() - self.initial_ascent_start < 4.0:
                    # Send ascend command and skip processing
                    twist = Twist()
                    twist.linear.z = 0.12  #
                    self.cmd_pub.publish(twist)
                    return
                else:
                    self.initial_ascent_done = True
        
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        h, w = frame.shape[:2]
        if self.center_x is None:
            self.center_x = w // 2
            self.center_y = h // 2

        # Gate detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        masks = {}
        red_mask = None
        for name, (low, high) in self.hsv_ranges.items():
            m = cv2.inRange(hsv, low, high)
            if 'red' in name:
                red_mask = m if red_mask is None else cv2.bitwise_or(red_mask, m)
            else:
                masks[name] = self.preprocess(m)
        if red_mask is not None:
            masks['red'] = self.preprocess(red_mask)

        candidates = {}
        gate_detected = False
        for color, mask in masks.items():
            cnts, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            if hier is None:
                continue
            for i, hinfo in enumerate(hier[0]):
                if hinfo[3] >= 0:
                    area = cv2.contourArea(cnts[i])
                    if area < 1000:
                        continue
                    M = cv2.moments(cnts[i])
                    cx = int(M['m10']/M['m00']) if M['m00'] else 0
                    cy = int(M['m01']/M['m00']) if M['m00'] else 0
                    prev = candidates.get(color, (0, None, 0, 0))
                    if area > prev[0]:
                        candidates[color] = (area, cnts[i], cx, cy)
                        gate_detected = True

        twist = Twist()
        # Always move forward by default
        twist.linear.x = self.forward_speed

        if candidates:
            # Update last detection time
            self.last_gate_time = time.time()
            
            color_sel, (area, cnt, cx, cy) = max(candidates.items(), key=lambda kv: kv[1][0])
            cv2.drawContours(frame, [cnt], -1, self.colors[color_sel], 3)
            cv2.circle(frame, (cx, cy), 5, self.colors[color_sel], -1)

            err_x = cx - self.center_x
            err_y = cy - self.center_y

            twist.angular.z = -self.kp_yaw * err_x
            twist.linear.z = -self.kp_alt * err_y

            self.pass_through_frames = self.PASS_FRAMES_MAX
        elif self.pass_through_frames > 0:
            self.pass_through_frames -= 1

        # Force land if no gates detected for 60 seconds
        if not self.landed:
            if (time.time() - self.last_gate_time) > 60:
                req = TelloAction.Request()
                req.cmd = 'land'
                self.tello_client.call_async(req)
                self.landed = True
                self.get_logger().info("Landing...")
                twist = Twist()

        self.cmd_pub.publish(twist)
        cv2.imshow('Gate Pilot', frame)
        cv2.waitKey(1)

def main():
    rclpy.init()
    node = GatePilot()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()