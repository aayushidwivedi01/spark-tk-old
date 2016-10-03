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

from itertools import cycle

def generate_mode(num_rows):
    choice = cycle([10, 20, 30, 40, 50])
    with open("../datasets/mode_stats2.csv", 'w') as f:
        for i in xrange(0, num_rows):
            f.write(str(next(choice)) + "," + str(4) + "\n")

if __name__ == "__main__":
    generate_mode(50)
