#!/usr/bin/env python

import threading
from math import *
import numpy as np
import pprint, time, copy
import pyquaternion as pyquat
import os, json, pdb, tempfile
import rospy, rospkg, tf, actionlib
from shapely.geometry import Polygon, Point
from threading import Thread

import tf2_ros, tf2_geometry_msgs
from nav_msgs.srv import GetPlan
from std_srvs.srv import Empty
from nav_msgs.msg import OccupancyGrid
from actionlib_msgs.msg import GoalStatus, GoalID
from apriltag_ros.msg import AprilTagDetectionArray
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from polygon_coverage_msgs.msg import PolygonWithHolesStamped
from geometry_msgs.msg import PointStamped, PoseStamped, PolygonStamped
from tf.transformations import euler_from_quaternion, quaternion_from_euler

from libs.list_helper import *
from libs.marker_visualization import MarkerVisualization
from common.common import trapezoid_calc_path, border_calc_path

from path_coverage.srv import SearchPolygon, StdDataType
from path_coverage.srv import TargetObject, TargetObjectRequest, TargetObjectResponse

pp = pprint.PrettyPrinter(indent=4, width=1)
INSCRIBED_INFLATED_OBSTACLE = 30

def get_tf_frame_pose(child_frame = 'entry_target_pose',base_frame = 'odom', listener=None, now=None, past=None, global_frame="map"):
    if(listener is None): _listener = tf.TransformListener()
    else: _listener = listener

    trans = None; rot = None
    if(now is None):
        _listener.waitForTransform(base_frame,child_frame, rospy.Time(0), rospy.Duration(1.0))
        (trans,rot) = _listener.lookupTransform(base_frame,child_frame, rospy.Time(0))
    elif(now is not None and past is None):
        _listener.waitForTransform(base_frame,child_frame, now, rospy.Duration(1.0))
        (trans,rot) = _listener.lookupTransform(base_frame,child_frame, now)
    elif(now is not None and past is not None):
    	_listener.waitForTransformFull(base_frame, now, child_frame, past, global_frame, rospy.Duration(1.0))
    	(trans,rot) = _listener.lookupTransformFull(base_frame, now, child_frame, past, global_frame)

    roll,pitch,yaw = tf.transformations.euler_from_quaternion(rot)

    pose = np.array(trans+[np.rad2deg(yaw)])
    Tmat = np.array(trans).T
    Rmat = tf.transformations.euler_matrix(roll,pitch,yaw,axes='sxyz')
    return pose, Rmat, Tmat, rot, [roll,pitch,yaw]
def buildPoseMsg(posList, quatList, flatnav=False):
	qNorm = pyquat.Quaternion(w=quatList[3],x=quatList[0],y=quatList[1],z=quatList[2])
	if(flatnav):
		qRot  = pyquat.Quaternion(array=np.array([1.0, 0.0, 0.0, 0.0]))
		qNorm = qNorm.rotate(qRot)
	msg = PoseStamped()
	msg.pose.position.x = posList[0]
	msg.pose.position.y = posList[1]
	msg.pose.position.z = posList[2]
	msg.pose.orientation.x = qNorm.normalised.x
	msg.pose.orientation.y = qNorm.normalised.y
	msg.pose.orientation.z = qNorm.normalised.z
	msg.pose.orientation.w = qNorm.normalised.w
	return msg
def printGoalState(state):
	if(state == GoalStatus.ABORTED): print("ABORTED")
	elif(state == GoalStatus.ACTIVE): print("ACTIVE")
	elif(state == GoalStatus.LOST): print("LOST")
	elif(state == GoalStatus.PENDING): print("PENDING")
	elif(state == GoalStatus.PREEMPTED): print("PREEMPTED")
	elif(state == GoalStatus.PREEMPTING): print("PREEMPTING")
	elif(state == GoalStatus.RECALLED): print("RECALLED")
	elif(state == GoalStatus.REJECTED): print("REJECTED")
	elif(state == GoalStatus.SUCCEEDED): print("SUCCEEDED")

class MapDrive(MarkerVisualization):
	rospack = rospkg.RosPack()
	lClickPoints = []
	local_costmap = None
	global_costmap = None
	following_path = False
	target_found = False
	pursuing_target = False
	coverage_requested = False
	beerDict = {
	    "name": "Fridge w/ BEER!",
	    "tag_id": 0,
	    "frame": "beer",
	    "found": False,
	    "tag_pose": None,
		"last_seen": None
	}
	washerDict = {
	    "name": "Washer w/ Laundry!",
	    "tag_id": 1,
	    "frame": "washer",
	    "found": False,
	    "tag_pose": None,
		"last_seen": None
	}
	trashDict = {
	    "name": "Trash Can w/ Trash!",
	    "tag_id": 2,
	    "frame": "trash",
	    "found": False,
	    "tag_pose": None,
		"last_seen": None
	}
	targets = [beerDict, washerDict, trashDict]
	target_pose = None
	current_target = None
	search_poly_pts = None
	target_tag_id = None
	last_recvd_polygon_time = None
	tag_expiration_limit = 10.0
	received_target_request = False
	qRef  = pyquat.Quaternion(array=np.array([1.0, 0.0, 0.0, 0.0]))
	force_cancelling_prior_goals = False
	lock = threading.Lock()
	def __init__(self):
		rospy.init_node('map_drive')
		MarkerVisualization.__init__(self)

		self.global_frame = rospy.get_param("~global_frame", "map")
		self.base_frame = rospy.get_param("~base_frame", "base_link")
		self.robot_width = rospy.get_param("~robot_width", 0.3)
		self.edge_offset = rospy.get_param("~edge_offset", 0.5)
		self.target_offset = rospy.get_param("~target_offset", 1.0)
		self.costmap_max_non_lethal = rospy.get_param("~costmap_max_non_lethal", 70)
		self.border_drive = rospy.get_param("~border_drive", False)
		self.poly_drive = rospy.get_param("~poly_drive", True)
		self.boustrophedon_decomposition = rospy.get_param("~boustrophedon_decomposition", True)
		local_costmap_topic = rospy.get_param("~local_costmap_topic", "/move_base/local_costmap/costmap")
		global_costmap_topic = rospy.get_param("~global_costmap_topic", "/move_base/global_costmap/costmap")
		marker_topic = rospy.get_param("~marker_topic", "/clicked_point")
		custom_poly_topic = rospy.get_param("~polygon_topic", "/polygon")
		tag_topic = rospy.get_param('~tag_topic', "tag_detections")

		self.move_base = actionlib.SimpleActionClient("move_base", MoveBaseAction)
		rospy.loginfo("Waiting for the move_base action server to come up")
		self.move_base.wait_for_server()
		self.move_base_plan = rospy.ServiceProxy('/move_base/make_plan', GetPlan)
		rospy.loginfo("Got move_base action server")

		self.tfBuffer = tf2_ros.Buffer()
		self.br = tf.TransformBroadcaster()
		self.tfListener = tf.TransformListener()
		listener = tf2_ros.TransformListener(self.tfBuffer)

		self.target_pub = rospy.Publisher("test_target", PoseStamped, queue_size=10)
		rospy.Subscriber(marker_topic, PointStamped, self.rvizPointReceived)
		rospy.Subscriber(custom_poly_topic, PolygonWithHolesStamped, self.polygonReceived)
		rospy.Subscriber(global_costmap_topic, OccupancyGrid, self.globalCostmapReceived)
		rospy.Subscriber(local_costmap_topic, OccupancyGrid, self.localCostmapReceived)
		rospy.Subscriber(tag_topic, AprilTagDetectionArray, self.tagCallback, queue_size=1)
		self.exec_srv = rospy.Service("map_drive/target", TargetObject, self.search_callback)
		self.reset_db_srv = rospy.Service("map_drive/reset", Empty, self.reset_callback)
		self.moveBaseCanceller = rospy.Publisher("move_base/cancel", GoalID, queue_size=10)

		rospy.on_shutdown(self.on_shutdown)
		self.rate = rospy.Rate(10.0)
		rospy.loginfo("Running..")
		self.stop_thread = False
		self.flagThread = Thread(target=self.goalCanceller,args=())
		self.flagThread.start()
	def goalCanceller(self):
		move_base = actionlib.SimpleActionClient("move_base", MoveBaseAction)
		while not rospy.is_shutdown():
			if(self.force_cancelling_prior_goals):
				state = self.move_base.get_state()
				printGoalState(state)
				while( True):
					if(self.stop_thread): break
					self.moveBaseCanceller.publish(GoalID())
					printGoalState(state)
					move_base.send_goal(MoveBaseGoal())
					self.move_base.cancel_all_goals()
					self.rate.sleep()
					if(not self.following_path): break
				self.force_cancelling_prior_goals = False


	def globalCostmapReceived(self, costmap):
		if(self.global_costmap is None): rospy.loginfo("Received Global Costmap")
		self.global_costmap = costmap

	def localCostmapReceived(self, costmap):
		if(self.global_costmap is None): rospy.loginfo("Received Local Costmap")
		self.local_costmap = costmap
		self.local_costmap_width = costmap.info.width*costmap.info.resolution
		self.local_costmap_height = costmap.info.height*costmap.info.resolution

	def on_shutdown(self):
		rospy.loginfo("Canceling all goals")
		self.visualization_cleanup()
		self.stop_thread = True
		self.move_base.cancel_all_goals()

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

	def next_pos(self, x, y, angle, timeout = None):
		rospy.loginfo("Moving to (%f, %f, %.0f�)" % (x, y, angle*180/pi))

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
		self.move_base.send_goal(goal)

		if(timeout is not None): finished_within_time = self.move_base.wait_for_result(rospy.Duration(timeout))
		else: finished_within_time = self.move_base.wait_for_result()

		if self.move_base.get_state() == GoalStatus.SUCCEEDED: rospy.loginfo("The base moved to (%f, %f)" % (x, y))
		else: rospy.logerr("The base failed moving to (%f, %f)" % (x, y))

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
		self.move_base.send_goal(goal)

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

	def get_closes_possible_goal(self, pos_last, pos_next, angle, tolerance):
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
			local_pose = self.tfBuffer.transform(pose, self.local_costmap.header.frame_id)

			cellx = round((local_pose.pose.position.x-self.local_costmap.info.origin.position.x)/self.local_costmap.info.resolution)
			celly = round((local_pose.pose.position.y-self.local_costmap.info.origin.position.y)/self.local_costmap.info.resolution)
			cellidx = int(celly*self.local_costmap.info.width+cellx)
			if cellidx < 0 or cellidx >= len(self.local_costmap.data):
				rospy.logwarn("get_closes_possible_goal landed outside costmap, returning original goal.")
				return pos_next
			cost = self.local_costmap.data[cellidx]
			if (cost >= INSCRIBED_INFLATED_OBSTACLE): break

			closest = pose
		if(closest is not None): return (closest.pose.position.x, closest.pose.position.y)
		else: return None

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

			self.next_pos(pos_last[0], pos_last[1], angle, timeout=timeout) # rotate in direction of next goal
			self.next_pos(pos_next[0], pos_next[1], angle, timeout=timeout)
		self.visualize_path(path, False)

	def drive_polygon(self, polygon):
		self.visualize_cell(polygon.exterior.coords[:])

		# Align longest side of the polygon to the horizontal axis
		angle = get_angle_of_longest_side_to_horizontal(polygon)
		if angle == None:
			rospy.logwarn("Can not return polygon")
			return
		angle+=pi/2 # up/down instead of left/right
		poly_rotated = rotate_polygon(polygon, angle)
		rospy.logdebug("Rotated polygon by %.0f�: %s" % (angle*180/pi, str(poly_rotated.exterior.coords[:])))

		if self.border_drive and not self.target_found:
			path_rotated = border_calc_path(poly_rotated, self.robot_width, edge_offset=self.edge_offset)
			path = rotate_points(path_rotated, -angle)
			self.drive_path(path)

		# run
		if self.poly_drive and not self.target_found:
			path_rotated = trapezoid_calc_path(poly_rotated, self.robot_width, edge_offset=self.edge_offset)
			path = rotate_points(path_rotated, -angle)
			self.drive_path(path)

		# cleanup
		self.visualize_cell(polygon.exterior.coords[:], False)
		rospy.logdebug("Polygon done")

	def get_target_object_info(self, target_id):
		if(target_id is None): return False, None

		self.lock.acquire()
		current_target_list = copy.deepcopy(self.targets)
		self.lock.release()

		for obj in current_target_list:
			if(obj["tag_id"] == target_id):
				if(obj["found"] and obj["tag_pose"] is not None):
					dt = rospy.Time.now() - obj["last_seen"]
					rospy.loginfo("Requested TargetObject \'%s\' (id = %d) has been found previously. [last_seen %.3f secs ago]" % (obj["name"], obj["tag_id"], dt.to_sec()) )
					return True, obj
		return False, None

	def executeCoverage(self, points):
		# last point is close to maximum, construct polygon
		rospy.loginfo("Creating polygon %s" % (str(points)))
		self.visualize_area(points, close=True)
		self.following_path = True
		if self.boustrophedon_decomposition: self.do_boustrophedon(Polygon(points), self.global_costmap)
		else: self.drive_polygon(Polygon(points))
		self.visualize_area(points, close=True, show=False)
		self.lClickPoints = []
		self.following_path = False
		rospy.loginfo("Finishing Coverage execution")

		return

	def rvizPointReceived(self, point):
		self.lClickPoints.append(point)
		points = [(p.point.x, p.point.y) for p in self.lClickPoints]
		self.global_frame = point.header.frame_id
		if len(self.lClickPoints) > 2: # All points must have same frame_id
			if len(set([p.header.frame_id for p in self.lClickPoints])) != 1: raise
			points_x = [p.point.x for p in self.lClickPoints]
			points_y = [p.point.y for p in self.lClickPoints]
			avg_x_dist = list_avg_dist(points_x)
			avg_y_dist = list_avg_dist(points_y)
			dist_x_first_last = abs(points_x[0] - points_x[-1])
			dist_y_first_last = abs(points_y[0] - points_y[-1])
			if dist_x_first_last < avg_x_dist/10.0 and dist_y_first_last < avg_y_dist/10.0:
				self.coverage_requested = True
				self.search_poly_pts = points
				return
		self.visualize_area(points, close=False)
	def polygonReceived(self, msg):
		dt = msg.header.stamp - rospy.Time.now()
		if(fabs(dt.to_sec()) > 0.5):
			rospy.loginfo("Received polygon is old (%.3f sec diff), not using" % (dt.to_sec()))
			return
		points = [(p.x, p.y) for p in msg.polygon.hull.points]
		self.global_frame = msg.header.frame_id
		if len(points) > 2: # All points must have same frame_id
			self.coverage_requested = True
			self.search_poly_pts = points
			return

	def reset_callback(self, req):
		rospy.loginfo("Reset service request...")
		self.move_base.cancel_all_goals()
		self.lClickPoints = []
		self.following_path = False
		self.target_found = False
		self.pursuing_target = False
		self.coverage_requested = False
		self.beerDict = {
		    "name": "Fridge w/ BEER!",
		    "tag_id": 0,
		    "frame": "beer",
		    "found": False,
		    "tag_pose": None,
			"last_seen": None
		}
		self.washerDict = {
		    "name": "Washer w/ Laundry!",
		    "tag_id": 1,
		    "frame": "washer",
		    "found": False,
		    "tag_pose": None,
			"last_seen": None
		}
		self.trashDict = {
		    "name": "Trash Can w/ Trash!",
		    "tag_id": 2,
		    "frame": "trash",
		    "found": False,
		    "tag_pose": None,
			"last_seen": None
		}
		self.targets = [beerDict, washerDict, trashDict]
		self.target_pose = None
		self.current_target = None
		self.search_poly_pts = None
		self.target_tag_id = None
		self.last_recvd_polygon_time = None
		self.received_target_request = False
		return []
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

	def tagCallback(self, msg):
		verbose = False
		self.lock.acquire()
		current_target_list = copy.deepcopy(self.targets)
		self.lock.release()
		now = rospy.Time.now()
		temp_target_found = False
		for tag in msg.detections:
			tagPose = tag.pose.pose.pose
			tag_pos = [tagPose.position.x,tagPose.position.y,tagPose.position.z]
			tag_or = (tagPose.orientation.x,tagPose.orientation.y,tagPose.orientation.z,tagPose.orientation.w)
			tag_angs = euler_from_quaternion(tag_or)
			# rospy.loginfo("Tag RPY = %.3f, %.3f, %.3f" % (np.rad2deg(tag_angs[0]),np.rad2deg(tag_angs[1]),np.rad2deg(tag_angs[2])))

			negQy = 1.5717 - tag_angs[1]
			negQz = -1*(1.5717 - tag_angs[2])
			negQuats = quaternion_from_euler( 3.14, negQy, negQz)

			# rospy.loginfo("Tag Quats = %.3f, %.3f, %.3f, %.3f" % (quats[0],quats[1],quats[2], quats[3]))
			for obj in current_target_list:
				needs_updating = False
				if(obj["tag_id"] == tag.id[0]):
					needs_update = not obj["found"]
					if(obj["last_seen"] is not None):
						dt = now - obj["last_seen"]
						needs_update = needs_update or (dt.to_sec() >= self.tag_expiration_limit)

					if(needs_update):
						rospy.loginfo("Updating Object: %s (frame = %s)" % (obj["name"], obj["frame"]) )
						try:
							self.br.sendTransform((0.0, 0.0, 1.0), tuple(negQuats), now, obj["frame"]+"/target", obj["frame"])
							# rospy.Rate(3.0).sleep()
							posf, _, _, rotf, _ = get_tf_frame_pose(child_frame = obj["frame"]+"/target", base_frame="map", now=now,listener=self.tfListener)
							tmpPose = buildPoseMsg(posf, rotf, True)
							tmpPose.header.stamp = now
							tmpPose.header.frame_id = "map"
							obj["found"] = True
							obj["last_seen"] = now
							obj["tag_pose"] = tmpPose
						except: pass

			if(self.received_target_request and self.target_tag_id is not None):
				if((not self.target_found) and (tag.id[0] == self.target_tag_id)):
					rospy.loginfo("tagCallback() --- Updating target object found, when it wasn\'t found at the time of request")
					self.target_found = True
					if( (self.coverage_requested or self.following_path) and not self.pursuing_target):
							self.force_cancelling_prior_goals = True
							rospy.loginfo("tagCallback() --- Target was found while coverage search was being performed. Cancelling all current move_base goals for coverage search.")
							# self.move_base.cancel_all_goals()
							# self.move_base.send_goal(MoveBaseGoal())
							# self.rate.sleep()
							# self.force_cancelling_prior_goals = False


		self.lock.acquire()
		self.targets = copy.deepcopy(current_target_list)
		self.lock.release()

	def start(self):
		while not rospy.is_shutdown():
			if(self.coverage_requested and self.search_poly_pts is not None):
				print("coverage needed")
				self.executeCoverage(self.search_poly_pts)

			if(self.received_target_request):
				print("received new request")
				# Update global variables to hold newer data if they weren't there prior
				if(self.coverage_requested): self.target_found, self.current_target = self.get_target_object_info(self.target_tag_id)

				reached_target = False
				if(self.target_found and self.current_target is not None):
					try:
						self.target_pub.publish(self.current_target["tag_pose"])
						reached_target = self.send_to_target(self.current_target["tag_pose"])
					except: rospy.warn("Something wrong occurred during the process of going towards the target object.")

				# If we've reached the area where the target object was last seen,
				# Check if the target object was actually seen recently in the process
				# of going towards it
				if(reached_target and not self.coverage_requested):
					prevTarget = self.current_target
					is_tag_available, curTarget = self.get_target_object_info(prevTarget["tag_id"])
					dt = curTarget["last_seen"] - prevTarget["last_seen"]
					if( (dt.to_sec() >= 30.0) or (dt.to_sec() == 0)):
						rospy.logwarn("Target object \'%s\' (id = %d) might not actually be where we previously thought it was. [%.3f secs difference b/w reported sightings]" % (prevTarget["name"], prevTarget["tag_id"], dt.to_sec() ))
					if(dt.to_sec() == 0):
						rospy.logwarn("Previously found object \'%s\' (id = %d) has been LOST, or it isn\'t not in the same position it was when we last saw it. Resetting its stored values back to null." % (prevTarget["name"], prevTarget["tag_id"]) )
						self.lock.acquire()
						for obj in self.targets:
							if(obj["tag_id"] == prevTarget["tag_id"]):
								obj["found"] = False
								obj["tag_pose"] = None
								obj["last_seen"] = None
						self.lock.release()

						rospy.logfatal("TODO: Now that we\'ve reset it\'s stored values, Perform re-acquirement procedure here. [NOTE: try a targetted coverage search in the room polygon that the last reported position of the target\'s tag falls into]")
						rospy.logfatal("TODO: Keep a local version of the last reported tag pose to use for finding the polygon region to use in a targetted coverage search before we do a FULL_BLOWN search")

			self.coverage_requested = False
			self.search_poly_pts = None
			self.current_target = None
			self.target_found = False
			self.received_target_request = False

			self.rate.sleep()

if __name__ == "__main__":
	p = MapDrive()
	p.start()
