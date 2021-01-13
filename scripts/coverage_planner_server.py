#!/usr/bin/env python

from math import *
import threading
from threading import Thread
import numpy as np
import pprint, time, copy
import pyquaternion as pyquat
import rospy, rospkg, tf, actionlib

import os, json, pdb, tempfile

import tf2_ros, tf2_geometry_msgs
from shapely.geometry import Polygon, Point
from polygon_coverage_msgs.msg import PolygonWithHolesStamped

from std_srvs.srv import Empty
from nav_msgs.srv import GetPlan
from nav_msgs.msg import OccupancyGrid
from actionlib_msgs.msg import GoalStatus, GoalID
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import PointStamped, PoseStamped, PolygonStamped

from common.list_helper import *
from common.marker_visualization import MarkerVisualization
from common.common import trapezoid_calc_path, border_calc_path
from common.common import get_tf_frame_pose, buildPoseMsg, printActionServerState
from common.common import getCostmapCellCost

from home_coverage_ros.msg import HighValueTarget
from home_coverage_ros.srv import HvtInfoGet, HvtInfoGetRequest, HvtInfoGetResponse
from home_coverage_ros.msg import CoveragePlannerAction, CoveragePlannerFeedback, CoveragePlannerResult

pp = pprint.PrettyPrinter(indent=4, width=1)
INSCRIBED_INFLATED_OBSTACLE = 253

class CoveragePlannerServer(MarkerVisualization):
	lock = threading.Lock()
	rospack = rospkg.RosPack()
	lClickPoints = []
	local_costmap = None
	global_costmap = None
	search_poly_pts = None

	running_coverage = False
	coverage_requested = False
	last_recvd_polygon_time = None

	target_tag_id = -1
	target_found = False

	count = 0
	_curGoalCnt = 0
	_force_cancel = False
	look_for_target = False

	_result = CoveragePlannerResult()
	_feedback = CoveragePlannerFeedback()
	costmap_max_cost = INSCRIBED_INFLATED_OBSTACLE
	def __init__(self):
		MarkerVisualization.__init__(self)
		rospy.init_node('coverage_planner_server')
		self._action_name = 'coverage_planner_server'

		self.global_frame = rospy.get_param("~global_frame", "map")
		self.base_frame = rospy.get_param("~base_frame", "base_link")

		self.def_max_cost = rospy.get_param("~default_max_cost", 96)
		self.def_lane_width = rospy.get_param("~default_lane_width", 0.3)
		self.def_edge_offset = rospy.get_param("~default_edge_offset", 0.5)
		self.def_target_approach_offset = rospy.get_param("~default_target_approach_offset", 1.0)
		self.def_costmap_max_non_lethal = rospy.get_param("~default_costmap_max_non_lethal", 70)
		self.def_follow_polygon_border = rospy.get_param("~default_follow_polygon_border", False)
		self.def_follow_inner_polygon = rospy.get_param("~default_follow_inner_polygon", True)
		self.def_boustrophedon_decomposition = rospy.get_param("~default_boustrophedon_decomposition", True)

		self.lane_width = self.def_lane_width
		self.edge_offset = self.def_edge_offset
		self.target_approach_offset = self.def_target_approach_offset
		self.costmap_max_non_lethal = self.def_costmap_max_non_lethal
		self.follow_polygon_border = self.def_follow_polygon_border
		self.follow_inner_polygon = self.def_follow_inner_polygon
		self.boustrophedon_decomposition = self.def_boustrophedon_decomposition

		custom_poly_topic = rospy.get_param("~polygon_topic", "/polygon")
		marker_topic = rospy.get_param("~marker_topic", "/clicked_point")
		local_costmap_topic = rospy.get_param("~local_costmap_topic", "/move_base/local_costmap/costmap")
		global_costmap_topic = rospy.get_param("~global_costmap_topic", "/move_base/global_costmap/costmap")

		self.hvt_db = rospy.ServiceProxy('/hvt_database/get', HvtInfoGet)
		self.move_base = actionlib.SimpleActionClient("move_base", MoveBaseAction)
		rospy.loginfo("Waiting for the move_base action server to come up")
		self.move_base.wait_for_server()
		self.move_base_plan = rospy.ServiceProxy('/move_base/make_plan', GetPlan)
		rospy.loginfo("Got move_base action server")

		self.tfBuffer = tf2_ros.Buffer()
		self.br = tf.TransformBroadcaster()
		self.tfListener = tf.TransformListener()
		listener = tf2_ros.TransformListener(self.tfBuffer)

		rospy.Subscriber(marker_topic, PointStamped, self.rvizPointReceived)
		rospy.Subscriber(custom_poly_topic, PolygonWithHolesStamped, self.polygonReceived)
		rospy.Subscriber(global_costmap_topic, OccupancyGrid, self.globalCostmapReceived)
		rospy.Subscriber(local_costmap_topic, OccupancyGrid, self.localCostmapReceived)
		self.target_pub = rospy.Publisher("test_target", PoseStamped, queue_size=10)

		self.costmap_max_cost = self.def_max_cost
		rospy.on_shutdown(self.on_shutdown)
		self.rate = rospy.Rate(10.0)
		self._as = actionlib.SimpleActionServer(self._action_name, CoveragePlannerAction, self.execute, False)
		self._as.start()
		rospy.loginfo("Running..")

	def __del__(self):
		rospy.loginfo("Cleaning up class CoveragePlannerServer()...")
		self.on_shutdown()
	def on_shutdown(self):
		rospy.loginfo("Canceling all goals")
		self.visualization_cleanup()
		self.stop_thread = True
		self.move_base.cancel_all_goals()

	def rvizPointReceived(self, point):
		pnt = [point.point.x, point.point.y]
		localcost = getCostmapCellCost(self.local_costmap, pnt)
		globalcost = getCostmapCellCost(self.global_costmap, pnt)
		rospy.loginfo("Local Costmap cost = %s at X,Y = %.3f, %.3f" % (str(localcost), pnt[0], pnt[1]))
		rospy.loginfo("Global Costmap cost = %s at X,Y = %.3f, %.3f" % (str(globalcost), pnt[0], pnt[1]))

	def polygonReceived(self, msg):
		dt = msg.header.stamp - rospy.Time.now()
		if(fabs(dt.to_sec()) > 0.5):
			rospy.loginfo("Received polygon is old (%.3f sec diff), not using" % (dt.to_sec()))
			return
		points = [(p.x, p.y) for p in msg.polygon.hull.points]
		self.global_frame = msg.header.frame_id
		if len(points) > 2: # All points must have same frame_id
			self.executeCoverage(points)
			return
	def globalCostmapReceived(self, costmap):
		if(self.global_costmap is None): rospy.logdebug("Received Global Costmap")
		self.global_costmap = costmap
	def localCostmapReceived(self, costmap):
		if(self.global_costmap is None): rospy.logdebug("Received Local Costmap")
		self.local_costmap = costmap
		self.local_costmap_width = costmap.info.width*costmap.info.resolution
		self.local_costmap_height = costmap.info.height*costmap.info.resolution

	def move_base_active_cb(self):
		rospy.loginfo("Goal pose "+str(self._curGoalCnt+1)+" is now being processed by the Action Server...")
	def move_base_feedback_cb(self, feedback):
		rospy.loginfo("Feedback for goal pose "+str(self._curGoalCnt+1)+" received")
	def move_base_done_cb(self, status, result):
		if status == 2:
			rospy.loginfo("Goal pose %s received a cancel request after it started executing, completed execution!" % (str(self._curGoalCnt)) )
		if status == 3:
			rospy.loginfo("Goal pose %s reached" % (str(self._curGoalCnt)) )
			return
		if status == 4:
			rospy.loginfo("Goal pose %s was aborted by the Action Server" % (str(self._curGoalCnt)) )
			# rospy.signal_shutdown("Goal pose %s aborted, shutting down!" % (str(self._curGoalCnt)) )
			return
		if status == 5:
			rospy.loginfo("Goal pose %s has been rejected by the Action Server" % (str(self._curGoalCnt)) )
			# rospy.signal_shutdown("Goal pose %s rejected, shutting down!" % (str(self._curGoalCnt)) )
			return
		if status == 8:
			rospy.loginfo("Goal pose %s received a cancel request before it started executing, successfully cancelled!" % (str(self._curGoalCnt)) )
	def send_move_base_goal(self, goal, timeout = None, cancel_on_target=False):
		success = False
		start = rospy.Time.now()

		rospy.loginfo("Sending goal for found target at (%f, %f)." % (goal.target_pose.pose.position.x, goal.target_pose.pose.position.y))
		# self.move_base.send_goal(goal, self.move_base_done_cb, self.move_base_active_cb, self.move_base_feedback_cb)
		self.move_base.send_goal(goal)
		while(True):
			if self._as.is_preempt_requested():
				rospy.loginfo('%s: Preempted' % self._action_name)
				self._force_cancel = True

			if(self._force_cancel):
				rospy.logerr("User-sent force cancel signal")
				self.move_base.cancel_all_goals()
				break

			target_info = None
			is_tag_found = False
			dt = rospy.Time.now() - start
			state = self.move_base.get_state()
			if(cancel_on_target or self.look_for_target):
				is_tag_found, target_info = self.get_target_object_info(self.target_tag_id)
			if(self.look_for_target):
				self.target_found = is_tag_found
				self.target_info = target_info

			if(cancel_on_target and is_tag_found):
				rospy.logwarn("Target Found, cancelling move_base goal early")
				self.move_base.cancel_all_goals()
				success = True
				break
			if( (timeout is not None) and (dt.to_sec() > timeout) ):
				rospy.logerr("Unable to reach goal in timeout force cancelling")
				self.move_base.cancel_all_goals()
				break
			if(state == GoalStatus.SUCCEEDED):
				rospy.loginfo("Successfully reached target at (%f, %f)" % (goal.target_pose.pose.position.x, goal.target_pose.pose.position.y))
				success = True
				break
			elif(state == GoalStatus.ABORTED):
				rospy.logerr("Unable to reach goal from abortion")
				self.move_base.cancel_all_goals()
				break
			# elif(state != GoalStatus.ACTIVE): printActionServerState(state)

			rospy.Rate(10.0).sleep()

		self.move_base.wait_for_result()
		self.move_base.get_result()
		rospy.logwarn("CoveragePlannerServer::send_move_base_goal() --- Result = %s " % (str(success)) )
		if self._as.is_preempt_requested(): self._as.set_preempted()
		return success
	def next_polygon_goal(self, x, y, angle, timeout = None):
		cost = getCostmapCellCost(self.local_costmap, [x, y])
		rospy.loginfo("Moving to next goal in polygon path (%f, %f, %.0f) w/ cost = %s" % (x, y, angle*180/pi, str(cost)))
		goal = MoveBaseGoal()
		angle_quat = tf.transformations.quaternion_from_euler(0, 0, angle)
		goal.target_pose.header.frame_id = self.global_frame
		goal.target_pose.header.stamp = rospy.Time.now()
		goal.target_pose.pose.position.x = x
		goal.target_pose.pose.position.y = y
		goal.target_pose.pose.orientation.x = angle_quat[0]
		goal.target_pose.pose.orientation.y = angle_quat[1]
		goal.target_pose.pose.orientation.z = angle_quat[2]
		goal.target_pose.pose.orientation.w = angle_quat[3]
		return self.send_move_base_goal(goal, timeout = timeout, cancel_on_target=self.look_for_target)

	def get_target_object_info(self, target_id = None):
		if( (target_id is None) or (target_id == -1)): return False, None
		resp = self.hvt_db(tag_id=target_id)
		if(resp.success):
			if(resp.info.found):
				dt = rospy.Time.now() - resp.info.last_seen
				rospy.loginfo_throttle(5.0, "Requested TargetObject \'%s\' (id = %d) has been found previously. [last_seen %.3f secs ago]" % (resp.info.name, resp.info.tag_id, dt.to_sec()) )
			return resp.info.found, resp.info
		return False, None

	def init_server_params(self, goal, verbose = True):
		self.target_tag_id = -1
		self.search_poly_pts = None

		if(goal.lane_width != self.def_lane_width):
			if(verbose): rospy.loginfo("CoveragePlannerServer::init_server_params() --- Overwriting default lane_width (%s) with provided goal value %s." % (str(self.def_lane_width), str(goal.lane_width)))
			self.lane_width = goal.lane_width
		else: self.lane_width = self.def_lane_width

		if(goal.edge_offset != self.def_edge_offset):
			if(verbose): rospy.loginfo("CoveragePlannerServer::init_server_params() --- Overwriting default edge_offset (%s) with provided goal value %s." % (str(self.def_edge_offset), str(goal.edge_offset)))
			self.edge_offset = goal.edge_offset
		else: self.edge_offset = self.def_edge_offset

		if(goal.target_approach_offset != self.def_target_approach_offset):
			if(verbose): rospy.loginfo("CoveragePlannerServer::init_server_params() --- Overwriting default target_approach_offset (%s) with provided goal value %s." % (str(self.def_target_approach_offset), str(goal.target_approach_offset)))
			self.target_approach_offset = goal.target_approach_offset
		else: self.target_approach_offset = self.def_target_approach_offset

		if(goal.costmap_max_non_lethal != self.def_costmap_max_non_lethal):
			if(verbose): rospy.loginfo("CoveragePlannerServer::init_server_params() --- Overwriting default costmap_max_non_lethal (%s) with provided goal value %s." % (str(self.def_costmap_max_non_lethal), str(goal.costmap_max_non_lethal)))
			self.costmap_max_non_lethal = goal.costmap_max_non_lethal
		else: self.costmap_max_non_lethal = self.def_costmap_max_non_lethal

		if(goal.follow_polygon_border != self.def_follow_polygon_border):
			if(verbose): rospy.loginfo("CoveragePlannerServer::init_server_params() --- Overwriting default follow_polygon_border (%s) with provided goal value %s." % (str(self.def_follow_polygon_border), str(goal.follow_polygon_border)))
			self.follow_polygon_border = goal.follow_polygon_border
		else: self.follow_polygon_border = self.def_follow_polygon_border

		if(goal.follow_inner_polygon != self.def_follow_inner_polygon):
			if(verbose): rospy.loginfo("CoveragePlannerServer::init_server_params() --- Overwriting default follow_inner_polygon (%s) with provided goal value %s." % (str(self.def_follow_inner_polygon), str(goal.follow_inner_polygon)))
			self.follow_inner_polygon = goal.follow_inner_polygon
		else: self.follow_inner_polygon = self.def_follow_inner_polygon

		if(goal.boustrophedon_decomposition != self.def_boustrophedon_decomposition):
			if(verbose): rospy.loginfo("CoveragePlannerServer::init_server_params() --- Overwriting default boustrophedon_decomposition (%s) with provided goal value %s." % (str(self.def_boustrophedon_decomposition), str(goal.boustrophedon_decomposition)))
			self.boustrophedon_decomposition = goal.boustrophedon_decomposition
		else: self.boustrophedon_decomposition = self.def_boustrophedon_decomposition

		if(goal.tag_id != -1):
			if(verbose): rospy.loginfo("CoveragePlannerServer::init_server_params() --- Overwriting default target_tag_id (%s) with provided goal value %s." % (str(-1), str(goal.tag_id)))
			self.target_tag_id = goal.tag_id
		else: self.target_tag_id = -1

		if(len(goal.search_region.points) > 2):
			points = [(p.x, p.y) for p in goal.search_region.points]
			self.search_poly_pts = points
		else: self.search_poly_pts = None
	def exit(self):
		self._result.success = False
		self._as.set_aborted(self._result)

	def execute(self, goal):
		skip_to_end = False
		failed = False
		rospy.loginfo("CoveragePlannerServer::execute() ---- Beginning Coverage Planner Action...")

		rospy.logwarn("CoveragePlannerServer::execute() ---- Previous Coverage Planner Action still running, forcing it to stop...")
		if(self.running_coverage):
			self._force_cancel = True
			try:
				while(self.running_coverage):
					self.rate.sleep()
			except KeyboardInterrupt:
				print("[INFO] CoveragePlannerServer::execute() --- KeyboardInterrupt catched...")
				if(not failed): self.exit()
				else: failed = True
				skip_to_end = True
			self._force_cancel = False

		if(not skip_to_end):
			rospy.loginfo("CoveragePlannerServer::execute() ---- Re-initializing ActionServer goal parameters...")
			self.init_server_params(goal)

		# If target object's tag id is given assume we need to look for it
		finished = False
		target_info = None
		is_tag_found = False
		if(not skip_to_end):
			if(self.target_tag_id >= 0):
				self.look_for_target = True
				is_tag_found, target_info = self.get_target_object_info(self.target_tag_id)
				if(not is_tag_found):
					self.target_found = False
					self.coverage_requested = True
					rospy.logwarn("CoveragePlannerServer --- Target id given, but not found. Requesting coverage.")
				else:
					rospy.logwarn("CoveragePlannerServer --- Target id given, and found. Sending to target.")
					self.target_found = True
					self.coverage_requested = False
					goal = MoveBaseGoal()
					goal.target_pose.header.frame_id = self.global_frame
					goal.target_pose.header.stamp = rospy.Time.now()
					goal.target_pose.pose = target_info.pose.pose
					self._result.success = self.send_move_base_goal(goal,timeout = None, cancel_on_target=False)
					self._as.set_succeeded(self._result)
					rospy.loginfo("CoveragePlannerServer::execute() --- Finished.")
					skip_to_end = True
			else:
				rospy.logwarn("CoveragePlannerServer --- No target id given. Requesting coverage w/o target.")
				self.target_found = False
				self.look_for_target = False
				self.coverage_requested = False
				self._result.success = self.executeCoverage(self.search_poly_pts)
				self._as.set_succeeded(self._result)
				rospy.loginfo("CoveragePlannerServer::execute() --- Finished.")
				skip_to_end = True

		# If the target has already been found send the goal for movement,
		# Otherwise, search for it or just test drive the coverage search planner
		finished = False
		if(self.coverage_requested and not skip_to_end):
			if(self.search_poly_pts is None):
				rospy.logerr("To run coverage planner a polygon needs to be defined for the search to cover.")
				if(not failed): self.exit()
				else: failed = True
				skip_to_end = True

			if(not skip_to_end):
				is_tag_found, target_info = self.get_target_object_info(self.target_tag_id)
				if(not is_tag_found):
					rospy.logwarn("CoveragePlannerServer --- Target id given, but not found. Performing coverage.")
					finished = self.executeCoverage(self.search_poly_pts)
					is_tag_found, target_info = self.get_target_object_info(self.target_tag_id)
					if(is_tag_found):
						rospy.logwarn("CoveragePlannerServer --- Target id found during coverage. Sending to target.")
						goal = MoveBaseGoal()
						goal.target_pose.header.frame_id = self.global_frame
						goal.target_pose.header.stamp = rospy.Time.now()
						goal.target_pose.pose = target_info.pose.pose
						self._result.success = self.send_move_base_goal(goal,timeout = None, cancel_on_target=False)
						self._as.set_succeeded(self._result)
						rospy.loginfo("CoveragePlannerServer::execute() --- Finished.")
					else:
						rospy.logerr("Unable to find given target, even after performing coverage.")
						if(not failed): self.exit()
						else: failed = True
						skip_to_end = True

		if(not skip_to_end):
			self.on_shutdown()
			self._as.set_succeeded(self._result)
		else:
			self.on_shutdown()
			if(not failed): self.exit()
		rospy.loginfo("CoveragePlannerServer::execute() --- END.")
	def executeCoverage(self, points):
		rospy.loginfo("Creating polygon %s" % (str(points)))
		self.running_coverage = True
		self.visualize_area(points, close=True)
		if self.boustrophedon_decomposition: self.do_boustrophedon(Polygon(points), self.global_costmap)
		else: self.drive_polygon(Polygon(points))
		self.visualize_area(points, close=True, show=False)
		self.lClickPoints = []
		self.running_coverage = False
		rospy.loginfo("Finishing Coverage execution")
		return True

	def start(self):
		while not rospy.is_shutdown():
			# if(self.coverage_requested and self.search_poly_pts is not None):
			# 	print("coverage needed")
			# 	self.executeCoverage(self.search_poly_pts)
			#
			# if(self.received_target_request):
			# 	print("received new request")
			# 	# Update global variables to hold newer data if they weren't there prior
			# 	if(self.coverage_requested): self.target_found, self.current_target = self.get_target_object_info(self.target_tag_id)
			#
			# 	reached_target = False
			# 	if(self.target_found and self.current_target is not None):
			# 		try:
			# 			self.target_pub.publish(self.current_target["tag_pose"])
			# 			reached_target = self.send_to_target(self.current_target["tag_pose"])
			# 		except: rospy.warn("Something wrong occurred during the process of going towards the target object.")
			#
			# 	# If we've reached the area where the target object was last seen,
			# 	# Check if the target object was actually seen recently in the process
			# 	# of going towards it
			# 	if(reached_target and not self.coverage_requested):
			# 		prevTarget = self.current_target
			# 		is_tag_available, curTarget = self.get_target_object_info(prevTarget["tag_id"])
			# 		dt = curTarget["last_seen"] - prevTarget["last_seen"]
			# 		if( (dt.to_sec() >= 30.0) or (dt.to_sec() == 0)):
			# 			rospy.logwarn("Target object \'%s\' (id = %d) might not actually be where we previously thought it was. [%.3f secs difference b/w reported sightings]" % (prevTarget["name"], prevTarget["tag_id"], dt.to_sec() ))
			# 		if(dt.to_sec() == 0):
			# 			rospy.logwarn("Previously found object \'%s\' (id = %d) has been LOST, or it isn\'t not in the same position it was when we last saw it. Resetting its stored values back to null." % (prevTarget["name"], prevTarget["tag_id"]) )
			# 			self.lock.acquire()
			# 			for obj in self.targets:
			# 				if(obj["tag_id"] == prevTarget["tag_id"]):
			# 					obj["found"] = False
			# 					obj["tag_pose"] = None
			# 					obj["last_seen"] = None
			# 			self.lock.release()
			#
			# 			rospy.logfatal("TODO: Now that we\'ve reset it\'s stored values, Perform re-acquirement procedure here. [NOTE: try a targetted coverage search in the room polygon that the last reported position of the target\'s tag falls into]")
			# 			rospy.logfatal("TODO: Keep a local version of the last reported tag pose to use for finding the polygon region to use in a targetted coverage search before we do a FULL_BLOWN search")
			self.rate.sleep()

	def drive_path(self, path, timeout = 60.0):
		self.visualize_path(path)
		initial_pos = self.tfBuffer.lookup_transform(self.global_frame, self.base_frame, rospy.Time(0), rospy.Duration(0.0))
		path.insert(0, (initial_pos.transform.translation.x, initial_pos.transform.translation.y))

		for pos_last,pos_next in pairwise(path):
			if rospy.is_shutdown(): return
			if self.target_found: continue
			pos_diff = np.array(pos_next)-np.array(pos_last)
			angle = atan2(pos_diff[1], pos_diff[0]) # angle from last to current position

			if abs(pos_diff[0]) < self.local_costmap_width/2.0 and abs(pos_diff[1]) < self.local_costmap_height/2.0:
				# goal is visible in local costmap, check path is clear
				tolerance = min(pos_diff[0], pos_diff[1])
				closest = self.get_closes_possible_goal(pos_last, pos_next, angle, tolerance)
				if closest is None: continue
				pos_next = closest

			getCostmapCellCost(self.local_costmap, pos_last)
			getCostmapCellCost(self.local_costmap, pos_next)
			success = self.next_polygon_goal(pos_last[0], pos_last[1], angle, timeout=timeout) # rotate in direction of next goal
			success2 = self.next_polygon_goal(pos_next[0], pos_next[1], angle, timeout=timeout)
		self.visualize_path(path, False)
	def do_boustrophedon(self, poly, costmap):
		# Cut polygon area from costmap
		(minx, miny, maxx, maxy) = poly.bounds
		rospy.loginfo("Converting costmap at x=%.2f..%.2f, y=%.2f %.2f for Boustrophedon Decomposition" % (minx, maxx, miny, maxy))

		# Convert to costmap coordinate
		minx = round((minx-costmap.info.origin.position.x)/costmap.info.resolution)
		maxx = round((maxx-costmap.info.origin.position.x)/costmap.info.resolution)
		miny = round((miny-costmap.info.origin.position.y)/costmap.info.resolution)
		maxy = round((maxy-costmap.info.origin.position.y)/costmap.info.resolution)

		# Check min/max limits
		if minx < 0: minx = 0
		if maxx > costmap.info.width: maxx = costmap.info.width
		if miny < 0: miny = 0
		if maxy > costmap.info.height: maxy = costmap.info.height

		# Transform costmap values to values expected by boustrophedon_decomposition script
		rows = []
		for ix in range(int(minx), int(maxx)):
			column = []
			for iy in range(int(miny), int(maxy)):
				x = ix*costmap.info.resolution+costmap.info.origin.position.x
				y = iy*costmap.info.resolution+costmap.info.origin.position.y
				data = costmap.data[int(iy*costmap.info.width+ix)]
				if data == -1 or not poly.contains(Point([x,y])): column.append(0) # Unknown or not inside polygon: Treat as obstacle
				elif data <= self.costmap_max_non_lethal: column.append(-1) # Freespace (non-lethal)
				else: column.append(0) # Obstacle
			rows.append(column)
		polygons = []
		with tempfile.NamedTemporaryFile(delete=False) as ftmp:
			ftmp.write(json.dumps(rows))
			ftmp.flush()
			boustrophedon_script = os.path.join(self.rospack.get_path('path_coverage'), "scripts/boustrophedon_decomposition.rb")
			with os.popen("%s %s" % (boustrophedon_script, ftmp.name)) as fscript:
				polygons = json.loads(fscript.readline())

		for poly in polygons:
			if(self.target_found):
				rospy.loginfo("Boustrophedon Decomposition Stopped b/c a requested target was found")
				return
			points = [((point[0]+minx)*costmap.info.resolution+costmap.info.origin.position.x,
					   (point[1]+miny)*costmap.info.resolution+costmap.info.origin.position.y
			) for point in poly]
			rospy.logdebug("Creating polygon from Boustrophedon Decomposition %s" % (str(points)))
			self.drive_polygon(Polygon(points))
		rospy.loginfo("Boustrophedon Decomposition done")
	def drive_polygon(self, polygon):
		self.visualize_cell(polygon.exterior.coords[:])

		# Align longest side of the polygon to the horizontal axis
		angle = get_angle_of_longest_side_to_horizontal(polygon)
		if angle == None:
			rospy.logwarn("Can not return polygon")
			return
		angle+=pi/2 # up/down instead of left/right
		poly_rotated = rotate_polygon(polygon, angle)
		rospy.logdebug("Rotated polygon by %.0f: %s" % (angle*180/pi, str(poly_rotated.exterior.coords[:])))

		if self.follow_polygon_border:
			path_rotated = border_calc_path(poly_rotated, self.lane_width, edge_offset=self.edge_offset)
			path = rotate_points(path_rotated, -angle)
			self.drive_path(path)
		if self.follow_inner_polygon:
			path_rotated = trapezoid_calc_path(poly_rotated, self.lane_width, edge_offset=self.edge_offset)
			path = rotate_points(path_rotated, -angle)
			self.drive_path(path)

		# cleanup
		self.visualize_cell(polygon.exterior.coords[:], False)
		rospy.logdebug("Polygon done")
	def get_closes_possible_goal(self, pos_last, pos_next, angle, tolerance):
		costmap = self.global_costmap
		angle_quat = tf.transformations.quaternion_from_euler(0, 0, angle)
		start = PoseStamped()
		start.header.frame_id = self.global_frame
		start.pose.position.x = pos_last[0]
		start.pose.position.y = pos_last[1]
		start.pose.orientation.x = angle_quat[0]
		start.pose.orientation.y = angle_quat[1]
		start.pose.orientation.z = angle_quat[2]
		start.pose.orientation.w = angle_quat[3]
		goal = PoseStamped()
		goal.header.frame_id = self.global_frame
		goal.pose.position.x = pos_next[0]
		goal.pose.position.y = pos_next[1]
		goal.pose.orientation.x = angle_quat[0]
		goal.pose.orientation.y = angle_quat[1]
		goal.pose.orientation.z = angle_quat[2]
		goal.pose.orientation.w = angle_quat[3]
		try: plan = self.move_base_plan(start, goal, tolerance).plan
		except: return None
		if len(plan.poses) == 0: return None

		closest = None
		for pose in plan.poses:
			pose.header.stamp = rospy.Time(0) # time for lookup does not need to be exact since we are stopped
			local_pose = self.tfBuffer.transform(pose, costmap.header.frame_id)

			cellx = round((local_pose.pose.position.x-costmap.info.origin.position.x)/costmap.info.resolution)
			celly = round((local_pose.pose.position.y-costmap.info.origin.position.y)/costmap.info.resolution)
			cellidx = int(celly*costmap.info.width+cellx)
			if cellidx < 0 or cellidx >= len(costmap.data):
				rospy.logwarn("get_closes_possible_goal landed outside costmap, returning original goal.")
				return pos_next
			cost = costmap.data[cellidx]
			if(cost >= self.costmap_max_cost):
				rospy.loginfo("costmap cost %s at X,Y = (%.3f, %.3f) exceeds max cost = %s" % (str(cost), local_pose.pose.position.x, local_pose.pose.position.y, str(self.costmap_max_cost)) )
				break
			else: rospy.loginfo("costmap cost = %s at X,Y  = (%.3f, %.3f)" % (str(cost), local_pose.pose.position.x, local_pose.pose.position.y) )

			closest = pose
		if(closest is not None): return (closest.pose.position.x, closest.pose.position.y)
		else: return None

	"""
	def send_to_target(self, pose):
		success = False
		goal = MoveBaseGoal()
		goal.target_pose.header.frame_id = self.global_frame
		goal.target_pose.header.stamp = rospy.Time.now()
		goal.target_pose.pose.position.x = pose.pose.position.x
		goal.target_pose.pose.position.y = pose.pose.position.y
		goal.target_pose.pose.orientation = pose.pose.orientation
		while(self.force_cancelling_prior_goals):
			rospy.loginfo("Waiting for prior move_base goals to be cancelled before sending new goal for found target...")
			self.rate.sleep()
		rospy.loginfo("Sending goal for found target at (%f, %f)." % (goal.target_pose.pose.position.x, goal.target_pose.pose.position.y))
		self.move_base.send_goal(goal, self.move_base_done_cb, self.move_base_active_cb, self.move_base_feedback_cb))

		self.pursuing_target = True
		self.move_base.wait_for_result()

		if self.move_base.get_state() == GoalStatus.SUCCEEDED:
			rospy.loginfo("Successfully reached target at (%f, %f)" % (pose.pose.position.x, pose.pose.position.y))
			self.target_tag_id = None
			success = True
			self.pursuing_target = False
		else:
			quats = (pose.pose.orientation.x, pose.pose.orientation.y,pose.pose.orientation.z,pose.pose.orientation.w)
			roll,pitch,yaw = tf.transformations.euler_from_quaternion(quats)
			rospy.logerr("The base failed moving to RPY = %.3f, %.3f, %.3f" % (np.rad2deg(roll),np.rad2deg(pitch),np.rad2deg(yaw)))
		return success

	def search_callback(self, req):
		# rospy.loginfo("Received service request for TargetObject:\r\n\t%s" % str(req))
		self.target_tag_id = req.tag_id
		self.received_target_request = True

		is_tag_available, target = self.get_target_object_info(req.tag_id)

		if(is_tag_available and target is not None):
			# move_base = actionlib.SimpleActionClient("move_base", MoveBaseAction)
			# move_base.send_goal(MoveBaseGoal())
			# self.moveBaseCanceller.publish(GoalID())
			self.move_base.cancel_all_goals()
			# self.force_cancelling_prior_goals = True
			dt = rospy.Time.now() - target["last_seen"]
			self.current_target = target
			self.target_found = True
			self.search_poly_pts = None
			self.coverage_requested = False
			return TargetObjectResponse(1, "target object \'%s\' (tag_id = %d) has previously been seen w/ valid pose %.3f secs ago" % (target["name"], target["tag_id"], dt.to_sec()) )
		else:
			rospy.loginfo("Requested target object either hasn\'t been found yet, or we were unable to get a valid pose from it")
			if(len(req.search_region.points) > 2):
				points = [(p.x, p.y) for p in req.search_region.points]
				self.target_found = False
				self.current_target = None
				self.search_poly_pts = points
				self.coverage_requested = True
				return TargetObjectResponse(2, "Target object hasn\'t been found, performing coverage search in an attempt to find it in the search region provided.")

		self.received_target_request = False
		self.target_tag_id = None
		self.target_found = False
		self.current_target = None
		self.search_poly_pts = None
		self.coverage_requested = False
		return TargetObjectResponse(-2, "Unable to process request for given target object. Reason = Object has not been previously seen and provided search region polygon was not valid (either empty or doesn\'t have enough points)")

	"""

if __name__ == "__main__":
	p = CoveragePlannerServer()
	p.start()
