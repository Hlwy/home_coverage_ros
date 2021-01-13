#!/usr/bin/env python

import threading
from math import *
import numpy as np
import pprint, time, copy
import pyquaternion as pyquat

import rospy, rospkg, tf, actionlib

from std_srvs.srv import Empty
from apriltag_ros.msg import AprilTagDetectionArray
from geometry_msgs.msg import PointStamped, PoseStamped
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from common.common import get_tf_frame_pose, buildPoseMsg, printActionServerState

from home_coverage_ros.msg import HighValueTarget
from home_coverage_ros.srv import HvtInfoGet, HvtInfoGetRequest, HvtInfoGetResponse
from home_coverage_ros.srv import HvtInfoAdd, HvtInfoAddRequest, HvtInfoAddResponse
from home_coverage_ros.srv import HvtInfoDel, HvtInfoDelRequest, HvtInfoDelResponse

pp = pprint.PrettyPrinter(indent=4, width=1)

defBeerTarget = HighValueTarget(name="beer", tag_id=0, label="Fridge w/ BEER!", frame_id="beer")
defWasherTarger = HighValueTarget(name="washer", tag_id=1, label="Washer w/ Laundry!", frame_id="washer")
defTrashTarger = HighValueTarget(name="trash", tag_id=2, label="Trash Can w/ Trash!", frame_id="trash")

class HvtDatabaseNode():
	objects = [defBeerTarget, defWasherTarger,defTrashTarger]
	tag_expiration_limit = 2.0
	qRef  = pyquat.Quaternion(array=np.array([1.0, 0.0, 0.0, 0.0]))
	lock = threading.Lock()
	def __init__(self):
		rospy.init_node('hvt_database_node')

		self.global_frame = rospy.get_param("~global_frame", "map")
		self.target_offset = rospy.get_param("~approach_offset", 1.0)
		tag_topic = rospy.get_param('~tag_topic', "tag_detections")

		self.br = tf.TransformBroadcaster()
		self.tfListener = tf.TransformListener()
		# self.target_pub = rospy.Publisher("test_target", PoseStamped, queue_size=10)

		rospy.Service("hvt_database/reset", Empty, self.resetCallback)
		rospy.Service("hvt_database/add", HvtInfoAdd, self.addHvtCallback)
		rospy.Service("hvt_database/delete", HvtInfoDel, self.delHvtCallback)
		rospy.Service("hvt_database/get", HvtInfoGet, self.getHvtCallback)
		rospy.Subscriber(tag_topic, AprilTagDetectionArray, self.tagCallback, queue_size=1)
		self.rate = rospy.Rate(10.0)
		rospy.loginfo("Running..")

	def tagCallback(self, msg):
		verbose = False
		self.lock.acquire()
		now = rospy.Time.now()
		temp_target_found = False
		for tag in msg.detections:
			tagPose = tag.pose.pose.pose
			tag_pos = [tagPose.position.x,tagPose.position.y,tagPose.position.z]
			tag_or = (tagPose.orientation.x,tagPose.orientation.y,tagPose.orientation.z,tagPose.orientation.w)
			tag_angs = euler_from_quaternion(tag_or)
			# rospy.loginfo("Tag RPY = %.3f, %.3f, %.3f" % (np.rad2deg(tag_angs[0]),np.rad2deg(tag_angs[1]),np.rad2deg(tag_angs[2])))
			# negQx = 3.14 - tag_angs[0]
			negQy = 1.5717 - tag_angs[1]
			negQz = -1*(1.5717 - tag_angs[2])
			negQuats = quaternion_from_euler( 3.14, negQy, negQz)
			# negQuats = quaternion_from_euler( 3.14, 1.5717, negQz)
			# rospy.loginfo("Tag Quats = %.3f, %.3f, %.3f, %.3f" % (quats[0],quats[1],quats[2], quats[3]))
			for obj in self.objects:
				needs_updating = False
				if(obj.tag_id == tag.id[0]):
					dt = now - obj.last_seen
					needs_update = not obj.found
					needs_update = needs_update or (dt.to_sec() >= self.tag_expiration_limit)

					if(needs_update):
						rospy.logdebug("Updating Object: %s (frame = %s)" % (obj.name, obj.frame_id) )
						self.br.sendTransform((0.0, 0.0, self.target_offset), tuple(negQuats), now, obj.frame_id+"/target", obj.frame_id)
						posf, _, _, rotf, _ = get_tf_frame_pose(child_frame = obj.frame_id+"/target", base_frame=self.global_frame, now=now,listener=self.tfListener)
						if(posf is not None):
							tmpPose = buildPoseMsg(posf, rotf, True)
							tmpPose.header.stamp = now
							tmpPose.header.frame_id = self.global_frame
							obj.found = True
							obj.times_seen += 1
							obj.last_seen = now
							obj.pose = tmpPose
		self.lock.release()

	def resetCallback(self,req):
		self.lock.acquire()
		rospy.loginfo("Resetting info for all HVT\'s currently stored in database")
		for obj in self.objects:
			obj.found = False
			obj.pose = PoseStamped()
			obj.last_seen = rospy.Time(0.0)
			obj.times_seen = 0
		rospy.logdebug("Database reset complete")
		self.lock.release()
		return []
	def addHvtCallback(self, req=HvtInfoAddRequest(name="", tag_id=-1)):
		resp = HvtInfoAddResponse(False, "")

		self.lock.acquire()
		for idx, item in enumerate(self.objects):
			if (item.tag_id == req.tag_id) or (item.name == req.name.lower()):
				resp.success = False
				resp.status = "Unable to add requested object \'%s\' (tag_id = %d) to database. Reason = object already exists in the database w/ same tag_id at index = %d." % (req.name, req.tag_id, idx)
				self.lock.release()
				return resp
		self.objects.append(HighValueTarget(name=req.name, tag_id=req.tag_id, label=req.label, frame_id=req.frame_id))
		resp.status = "Successfully added HVT \'%s\' w/ tag_id = %d, frame_id = %s to current database at index %d" % (req.name, req.tag_id, req.frame_id, idx+1)
		resp.success = True
		self.lock.release()
		return resp
	def delHvtCallback(self, req=HvtInfoDelRequest(name="", tag_id=-1)):
		resp = HvtInfoDelResponse(False, "")
		rmIdcs = []
		self.lock.acquire()
		for idx, item in enumerate(self.objects):
			if (item.tag_id == req.tag_id) or (item.name == req.name.lower()):
				rmIdcs.append(idx)

		idxOffset = 0
		if(len(rmIdcs) > 0):
			for i in rmIdcs:
				del self.objects[i-idxOffset]
				idxOffset += 1
			resp.status = "Successfully removed %d instances of the requested object \'%s\' (tag_id = %d) at indices = %s" % (int(len(rmIdcs)), req.name, req.tag_id, str(rmIdcs))
			resp.success = True
		else:
			resp.status = "No instances of the requested object \'%s\' (tag_id = %d) currently exist in the database. Nothing removed." % (req.name, req.tag_id)
			resp.success = False
		self.lock.release()
		return resp
	def getHvtCallback(self, req=HvtInfoGetRequest("", -1)):
		self.lock.acquire()
		db_copy = copy.deepcopy(self.objects)
		self.lock.release()
		resp = HvtInfoGetResponse(False, HighValueTarget())
		infoFound = False

		for item in db_copy:
			if (req.tag_id != -1) and (item.tag_id == req.tag_id):
				resp.success = True
				resp.info = item
				return resp
			elif (req.name != "") and (item.name == req.name.lower()):
				resp.success = True
				resp.info = item
				return resp
		return HvtInfoGetResponse(False, HighValueTarget())

	def start(self):
		while not rospy.is_shutdown():
			self.rate.sleep()

if __name__ == "__main__":
	p = HvtDatabaseNode()
	p.start()
