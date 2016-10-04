/**
 *  Copyright (c) 2016 Intel Corporation 
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */
package org.trustedanalytics.sparktk.frame.internal.ops.cumulativedist

import org.apache.commons.lang.StringUtils
import org.trustedanalytics.sparktk.frame.{ Column, DataTypes }
import org.trustedanalytics.sparktk.frame.internal.{ FrameState, FrameTransform, BaseFrame }

trait CumulativeSumTransform extends BaseFrame {
  /**
   * Add column to frame with cumulative percent.
   *
   * A cumulative sum is computed by sequentially stepping through the rows, observing the column values and keeping
   * track of the cumulative sum for each value.
   *
   * @note This method applies only to columns containing numerical data.
   *
   * @param sampleCol The name of the column from which to compute the cumulative percent.
   */
  def cumulativeSum(sampleCol: String): Unit = {
    execute(CumulativeSum(sampleCol))
  }
}

case class CumulativeSum(sampleCol: String) extends FrameTransform {
  require(StringUtils.isNotEmpty(sampleCol), "column name for sample is required")

  override def work(state: FrameState): FrameState = {
    // run the operation
    val cumulativeDistRdd = CumulativeDistFunctions.cumulativeSum(state, sampleCol)
    val updatedSchema = state.schema.addColumnFixName(Column(sampleCol + "_cumulative_sum", DataTypes.float64))

    // return result
    FrameState(cumulativeDistRdd, updatedSchema)
  }
}