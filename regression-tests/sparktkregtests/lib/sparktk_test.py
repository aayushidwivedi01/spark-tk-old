"""Setup up tests for regression """
import unittest
import uuid
import datetime

import sparktk as stk

import config
from threading import Lock

lock = Lock()
global_tc = None


def get_cluster_spark_conf():
    return {'spark.conf.properties.spark.executor.extrajavaoptions': "-Xmx4224m",
            'spark.conf.properties.spark.driver.maxPermSize': "512m",
            'spark.conf.properties.spark.yarn.driver.memoryOverhead': "384",
            'spark.conf.properties.spark.driver.memory': "3712m",
            #'auto-partitioner.broadcast-join-threshold': "2048MB",
            'spark.conf.properties.spark.driver.cores': "1",
            'spark.conf.properties.spark.yarn.executor.memoryOverhead': "384",
            'spark.conf.properties.spark.shuffle.service.enabled': "true",
            'spark.conf.properties.spark.dynamicAllocation.maxExecutors': "38",
            'spark.conf.properties.spark.driver.maxResultSize': "2g",
            'spark.conf.properties.spark.executor.cores': "1",
            'spark.conf.properties.spark.dynamicAllocation.minExecutors': "1",
            'spark.conf.properties.spark.shuffle.io.preferDirectBufs': "false",
            'spark.conf.properties.spark.yarn.am.waitTime': "1000000",
            'spark.conf.properties.spark.executor.memory': "5248m",
            'spark.conf.properties.spark.driver.extraJavaOptions': "Xmx3072m",
            'spark.conf.properties.spark.dynamicAllocation.enabled': "true",
            'spark.conf.properties.spark.eventLog.enabled': "false",
}

def get_context():
    global global_tc
    with lock:
        if global_tc is None:
            global_tc = stk.TkContext(master='yarn-client', extra_conf=get_cluster_spark_conf())
    return global_tc


class SparkTKTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Build the context for use"""
        cls.context = get_context()

    def setUp(self):
        pass

    def tearDown(self):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def get_file(self, filename):
        """Return the hdfs path to the given file"""

        # Note this is an HDFS path, not a userspace path. os.path library
        # may be wrong
        placed_path = "/user/" + config.user + "/qa_data/" + filename
        return placed_path

    def get_name(self, prefix):
        """build a guid hardened unique name """
        datestamp = datetime.datetime.now().strftime("%m_%d_%H_%M_")
        name = prefix + datestamp + uuid.uuid1().hex

        return name

    def assertFramesEqual(self, frame1, frame2):
        frame1_take = frame1.take(frame1.count()).data
        frame2_take = frame2.take(frame2.count()).data

        self.assertItemsEqual(frame1_take, frame2_take)
