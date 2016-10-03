# vim: set encoding=utf-8

#  Copyright (c) 2016 Intel Corporation 
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#


import random
import math
import numpy

def data_generate(users, items):
    user_factors = numpy.array([[random.random()*5 for i in range(3)] for i in range(users)])
    item_factors = numpy.array([[random.random()*5 for j in range(items)] for i in range(3)])

    values = numpy.dot(user_factors, item_factors)
    for u in range(users):
        for i in range(items):
            print ",".join(["user-"+str(u), "item-"+str(i), str(values[int(u)][int(i)])])




if __name__ == "__main__":
    data_generate(40, 40)
