""" Tests Logistic Regression Model against known values, calculated in R"""
import unittest
import sys
import os
from sparktkregtests.lib import sparktk_test
from numpy import array

class LogisticRegression(sparktk_test.SparkTKTestCase):

    def setUp(self):
        """Build the frames needed for the tests."""
        super(LogisticRegression, self).setUp()

        binomial_dataset = self.get_file("small_logit_binary.csv")

        schema = [("vec0", float),
                  ("vec1", float),
                  ("vec2", float),
                  ("vec3", float),
                  ("vec4", float),
                  ("res", int),
                  ("count", int),
                  ("actual", int)]
        self.binomial_frame = self.context.frame.import_csv(
            binomial_dataset, schema=schema, header=True)

    @unittest.skip("")
    def test_predict(self):
        """test predict works as expected"""
        log_model = self.context.models.classification.logistic_regression.train(
            self.binomial_frame, ["vec0", "vec1", "vec2", "vec3", "vec4"],
            'res')
        log_model.predict(self.binomial_frame,
            ["vec0", "vec1", "vec2", "vec3", "vec4"])
        frame = self.binomial_frame.copy(["actual", "predicted_label"])
        labels = frame.download(frame.row_count)
        for _, row in labels.iterrows():
            self.assertEqual(row["actual"], row["predicted_label"])

    @unittest.skip("")
    def test_bad_predict_frame_type(self):
        """test invalid frame on predict"""
        with self.assertRaisesRegexp(Exception, ".*"):
            log_model = self.context.models.classification.logistic_regression.train(
                self.binomial_frame, ["vec0", "vec1", "vec2", "vec3", "vec4"],
                'res', num_classes=2)
            log_model.predict(7)

    def test_bad_predict_observation(self):
        """test invalid observations on predict"""
        log_model = self.context.models.classification.logistic_regression.train(
                self.binomial_frame, ["vec0", "vec1", "vec2", "vec3", "vec4"],
                'res', num_classes=2)
        with self.assertRaisesRegexp(Exception,
            "java.lang.Integer cannot be cast to scala.collection.LinearSeqOptimized"):
            log_model.predict(self.binomial_frame, 7)

    def test_bad_predict_observation_value(self):
        """test invalid observation value on predict"""
        log_model = self.context.models.classification.logistic_regression.train(
                self.binomial_frame, ["vec0", "vec1", "vec2", "vec3", "vec4"],
                'res', num_classes=2)
        with self.assertRaisesRegexp(Exception, 
            "Number of columns for train and predict should be same"):
            log_model.predict(self.binomial_frame, ["err"])

    def test_bad_feature_column_type(self):
        """test invalid feature column type"""
        with self.assertRaisesRegexp(TypeError, "\'int\' object is not iterable"):
            log_model = self.context.models.classification.logistic_regression.train(
                self.binomial_frame, 7, 'res', num_classes=2)

    def test_no_such_feature_column_train(self):
        """test invalid feature column name while training"""
        with self.assertRaisesRegexp(Exception, "Invalid column name blah provided"):
            log_model = self.context.models.classification.logistic_regression.train(
                self.binomial_frame, ["vec0", "vec1", "blah", "vec3", "vec4"], 
                'res', num_classes=2 )

    def test_no_feature_column_test(self):
        """test invalid feature column name in test"""
        log_model = self.context.models.classification.logistic_regression.train(
                self.binomial_frame, ["vec0", "vec1", "vec2", "vec3", "vec4"],
                'res', num_classes=2)
        with self.assertRaisesRegexp(
                Exception,
                "Invalid column name blah provided"):
            test_result = log_model.test(self.binomial_frame, 'res',
                                         ["vec0", "vec1", "blah", "vec3", "vec4"])

    @unittest.skip("")
    def test_label_column_type_train(self):
        """test invalid label column type name in train"""
        with self.assertRaises(Exception):
            log_model = self.context.models.classification.logistic_regression.train(
                self.binomial_frame, ["vec0", "vec1", "vec2", "vec3", "vec4"],
                7, num_classes=2)

    #def test_observation_column_invalid(self):
    #    """test invalid observation column name in train"""
    #    with self.assertRaises(Exception):
    #        log_model = self.context.models.classification.logistic_regression.train()
    #        log_model.train(
    #            self.binomial_frame,
    #            "res", ["err", "vec1", "blah", "vec3", "vec4"])

    @unittest.skip("")
    def test_incorrect_num_classes(self):
        """test wrong num_classes type in train"""
        with self.assertRaises(Exception):
            log_model = self.context.models.classification.logistic_regression.train(
                self.binomial_frame, ["vec0", "vec1", "vec2", "vec3", "vec4"],
                'res', num_classes=7)


    @unittest.skip("")
    def test_bad_optimizer_type(self):
        """test invalid optimizer type train"""
        with self.assertRaises(Exception):
            log_model = self.context.models.classification.logistic_regression.train(
                self.binomial_frame, ["vec0", "vec1", "vec2", "vec3", "vec4"],
                'res', num_classes=2, optimizer=7)

    @unittest.skip("")
    def test_bad_optimizer_value(self):
        """test invalid optmizer value train"""
        with self.assertRaises(Exception):
            log_model = self.context.models.classification.logistic_regression.train(
                self.binomial_frame, ["vec0", "vec1", "vec2", "vec3", "vec4"],
                'res', num_classes=2, optimizer="err")

    @unittest.skip("")
    def test_bad_num_classes_type(self):
        """test bad num_classes data type in train"""
        with self.assertRaises(Exception):
            log_model = self.context.models.classification.logistic_regression.train(
                self.binomial_frame, ["vec0", "vec1", "vec2", "vec3", "vec4"], 
                'res', num_classes="err")


    @unittest.skip("")
    def test_bad_frame_type(self):
        """test invalid frame type train"""
        with self.assertRaises(Exception):
            log_model = self.context.models.classification.logistic_regression.train(
                "ERR", ["vec0", "vec1", "vec2", "vec3", "vec4"],
                "res", num_classes=2)


    def test_optimizer_bad_config(self):
        """test invalid sgd configuraton"""
        with self.assertRaisesRegexp(Exception, "multinomial logistic regression not supported for SGD"):
            log_model = self.context.models.classification.logistic_regression.train(
                self.binomial_frame, ["vec0", "vec1", "vec2", "vec3", "vec4"],
                "count", num_classes=4, optimizer="SGD")

    @unittest.skip("")
    def test_covariance_type(self):
        """test invalid covariance type"""
        with self.assertRaises(Exception):
            log_model = self.context.models.classification.logistic_regression.train(
                self.binomial_frame, ["vec0", "vec1", "vec2", "vec3", "vec4"],
                "res", num_classes=2, compute_covariance=7)

    @unittest.skip("")
    def test_covariance_false(self):
        """test not computing covariance"""
        log_model = self.context.models.classification.logistic_regression.train(
            self.binomial_frame, ["vec0", "vec1", "vec2", "vec3", "vec4"],
            "res", compute_covariance=False)
        self.assertIsNone(log_model.training_summary.covariance_matrix)

    @unittest.skip("")
    def test_intercept_false(self):
        """test not having an intercept"""
        log_model = self.context.models.classification.logistic_regression.train(
            self.binomial_frame, ["vec0", "vec1", "vec2", "vec3", "vec4"],
            "res", intercept=False)
        self.assertNotIn("intercept", log_model.training_summary.coefficients.keys())

    @unittest.skip("")
    def test_intercept_type(self):
        """test invalid intercept type"""
        with self.assertRaises(Exception):
            log_model = self.context.models.classification.logistic_regression.train(
                self.binomial_frame, ["vec0", "vec1", "vec2", "vec3", "vec4"],
                "res", intercept=7)

    @unittest.skip("")
    def test_feature_scaling_type(self):
        """test feature scaling type"""
        with self.assertRaises(Exception):
            log_model = self.context.models.classification.logistic_regression.train(
                self.binomial_frame, ["vec0", "vec1", "vec2", "vec3", "vec4"],
                "res", feature_scaling=7)

    @unittest.skip("")
    def test_num_corrections_type(self):
        """test invalid num corrections type"""
        with self.assertRaises(Exception):
            log_model = self.context.models.classification.logistic_regression.train(
                self.binomial_frame, ["vec0", "vec1", "vec2", "vec3", "vec4"],
                "res", num_corrections="ERR")

    @unittest.skip("")
    def test_threshold_type(self):
        """test invalid threshold type"""
        with self.assertRaises(Exception):
            log_model = self.context.models.classification.logistic_regression.train(
                self.binomial_frame, ["vec0", "vec1", "vec2", "vec3", "vec4"],
                "res", threshold="ERR")

    @unittest.skip("")
    def test_regularization_invalid(self):
        """test regularization parameter"""
        with self.assertRaises(Exception):
            log_model = self.context.models.classification.logistic_regression.train(
                self.binomial_frame, ["vec0", "vec1", "vec2", "vec3", "vec4"],
                "res", reg_type="ERR")

    @unittest.skip("")
    def test_num_iterations_type(self):
        """test invalid num iterations type"""
        with self.assertRaises(Exception):
            log_model = self.context.models.classification.logistic_regression.train(
                self.binomial_frame, ["vec0", "vec1", "vec2", "vec3", "vec4"],
                "res", num_iterations="ERR")

    @unittest.skip("")
    def test_regularization_value_type(self):
        """test invalid regularization value"""
        with self.assertRaises(Exception):
            log_model = self.context.models.classification.logistic_regression.train(
                self.binomial_frame, ["vec0", "vec1", "vec2", "vec3", "vec4"],
                "res", reg_param="ERR")

    @unittest.skip("")
    def test_convergence_type(self):
        """test invalid convergence type"""
        with self.assertRaises(Exception):
            log_model = self.context.models.classification.logistic_regression.train(
                self.binomial_frame, ["vec0", "vec1", "vec2", "vec3", "vec4"],
                "res", convergence_tolerance="ERR")

    @unittest.skip("")
    def test_regularization_type(self):
        """test invalid reg type type"""
        with self.assertRaises(Exception):
            log_model = self.context.models.classification.logistic_regression.train(
                self.binomial_frame, ["vec0", "vec1", "vec2", "vec3", "vec4"],
                "res", reg_type=7)

    @unittest.skip("")
    def test_step_size_type(self):
        """test invalid step size type"""
        with self.assertRaises(Exception):
            log_model = self.context.models.classification.logistic_regression.train(
                self.binomial_frame, ["vec0", "vec1", "vec2", "vec3", "vec4"],
                "res", step_size="ERR")

    @unittest.skip("")
    def test_logistic_regression_test(self):
        """test the default happy path of logistic regression test"""
        log_model = self.context.models.classification.logistic_regression.train(
            self.binomial_frame, ["vec0", "vec1", "vec2", "vec3", "vec4"], "res")
        values = log_model.test(
            self.binomial_frame, "res",
            ["vec0", "vec1", "vec2", "vec3", "vec4"])

        tp_f = self.binomial_frame.copy()
        tp_f.filter(lambda x: x['res'] == 1 and x['actual'] == 1)
        tp = float(tp_f.row_count)

        tn_f = self.binomial_frame.copy()
        tn_f.filter(lambda x: x['res'] == 0 and x['actual'] == 0)
        tn = float(tn_f.row_count)

        fp_f = self.binomial_frame.copy()
        fp_f.filter(lambda x: x['res'] == 0 and x['actual'] == 1)
        fp = float(fp_f.row_count)

        fn_f = self.binomial_frame.copy()
        fn_f.filter(lambda x: x['res'] == 1 and x['actual'] == 0)
        fn = float(fn_f.row_count)

        # manually build the confusion matrix and derivitive properties
        precision = tp/(tp+fp)
        recall = tp/(tp+fn)

        self.assertAlmostEqual(values.precision, precision)
        self.assertAlmostEqual(values.recall, recall)
        self.assertAlmostEqual(
            values.f_measure, 2*(precision*recall)/(precision+recall))
        self.assertAlmostEqual(values.accuracy, (tp+tn)/(tp+fn+tn+fp))

        self.assertAlmostEqual(
            values.confusion_matrix["Predicted_Pos"]["Actual_Pos"], tp)
        self.assertAlmostEqual(
            values.confusion_matrix["Predicted_Pos"]["Actual_Neg"], fp)
        self.assertAlmostEqual(
            values.confusion_matrix["Predicted_Neg"]["Actual_Neg"], tn)
        self.assertAlmostEqual(
            values.confusion_matrix["Predicted_Neg"]["Actual_Pos"], fn)

    @unittest.skip("")
    def test_logistic_regression_train(self):
        """test logistic regression train"""
        log_model = self.context.models.classification.logistic_regression.train(
            self.binomial_frame, ["vec0", "vec1", "vec2", "vec3", "vec4"],
            "res")
        self._standard_summary(log_model.training_summary, False)

    @unittest.skip("")
    def test_logistic_regression_train_sgd(self):
        """test logistic regression train with sgd"""
        log_model = self.context.models.classification.logistic_regression.train(
            self.binomial_frame, ["vec0", "vec1", "vec2", "vec3", "vec4"], 
            "res", num_classes=2, optimizer="SGD", step_size=12)
        self._standard_summary(log_model.taining_summary, False)

    @unittest.skip("")
    def test_logistic_regression_train_count_column(self):
        """test logistic regression train with count"""
        log_model = self.context.models.classification.logistic_regression.train(
            self.binomial_frame, ["vec0", "vec1", "vec2", "vec3", "vec4"],
            "res", "count")
        self._standard_summary(log_model.training_summary, True)

    def _standard_summary(self, summary, coefficients_only):
        """verfies the summary object against calculated values from R"""
        self.assertEqual(summary.num_features, 5)
        self.assertEqual(summary.num_classes, 2)
        self.assertAlmostEqual(
            summary.coefficients["intercept"],
            0.9, delta=0.01)
        self.assertAlmostEqual(
            summary.coefficients["vec0"], 0.4, delta=0.01)
        self.assertAlmostEqual(
            summary.coefficients["vec1"], 0.7, delta=0.01)
        self.assertAlmostEqual(
            summary.coefficients["vec2"], 0.9, delta=0.01)
        self.assertAlmostEqual(
            summary.coefficients["vec3"], 0.3, delta=0.01)
        self.assertAlmostEqual(
            summary.coefficients["vec4"], 1.4, delta=0.01)

        if not coefficients_only:
            self.assertAlmostEqual(
                summary.degrees_freedom["intercept"], 1)
            self.assertAlmostEqual(
                summary.standard_errors["intercept"],
                0.012222, delta=0.01)
            self.assertAlmostEqual(
                summary.wald_statistic["intercept"], 73.62,
                delta=0.01)
            self.assertAlmostEqual(
                summary.p_value["intercept"], 0,
                delta=0.01)

            self.assertAlmostEqual(
                summary.degrees_freedom["vec0"], 1)
            self.assertAlmostEqual(
                summary.standard_errors["vec0"], 0.005307,
                delta=0.01)
            self.assertAlmostEqual(
                summary.wald_statistic["vec0"], 75.451032,
                delta=0.01)
            self.assertAlmostEqual(
                summary.p_value["vec0"], 0,
                delta=0.01)

            self.assertAlmostEqual(
                summary.degrees_freedom["vec1"], 1)
            self.assertAlmostEqual(
                summary.standard_errors["vec1"], 0.006273,
                delta=0.01)
            self.assertAlmostEqual(
                summary.wald_statistic["vec1"], 110.938600,
                delta=0.01)
            self.assertAlmostEqual(
                summary.p_value["vec1"], 0,
                delta=0.01)

            self.assertAlmostEqual(
                summary.degrees_freedom["vec2"], 1)
            self.assertAlmostEqual(
                summary.standard_errors["vec2"], 0.007284,
                delta=0.01)
            self.assertAlmostEqual(
                summary.wald_statistic["vec2"], 124.156836,
                delta=0.01)
            self.assertAlmostEqual(
                summary.p_value["vec2"], 0,
                delta=0.01)

            self.assertAlmostEqual(
                summary.degrees_freedom["vec3"], 1)
            self.assertAlmostEqual(
                summary.standard_errors["vec3"], 0.005096,
                delta=0.01)
            self.assertAlmostEqual(
                summary.wald_statistic["vec3"], 58.617307,
                delta=0.01)
            self.assertAlmostEqual(
                summary.p_value["vec3"], 0,
                delta=0.01)

            self.assertAlmostEqual(
                summary.degrees_freedom["vec4"], 1)
            self.assertAlmostEqual(
                summary.standard_errors["vec4"], 0.009784,
                delta=0.01)
            self.assertAlmostEqual(
                summary.wald_statistic["vec4"], 143.976980,
                delta=0.01)
            self.assertAlmostEqual(
                summary.p_value["vec4"], 0,
                delta=0.01)

            r_cov = [
                [1.495461e-04, 1.460983e-05, 2.630870e-05,
                 3.369595e-05, 9.938721e-06, 5.580564e-05],
                [1.460983e-05, 2.816412e-05, 1.180398e-05,
                 1.477251e-05, 6.008541e-06, 2.206105e-05],
                [2.630870e-05, 1.180398e-05, 3.935554e-05,
                 2.491004e-05, 8.871815e-06, 4.022959e-05],
                [3.369595e-05, 1.477251e-05, 2.491004e-05,
                 5.305153e-05, 1.069435e-05, 5.243129e-05],
                [9.938721e-06, 6.008541e-06, 8.871815e-06,
                 1.069435e-05, 2.596849e-05, 1.776645e-05],
                [5.580564e-05, 2.206105e-05, 4.022959e-05,
                 5.243129e-05, 1.776645e-05, 9.572358e-05]]

            summ_cov = summary.covariance_matrix.take(
                summary.covariance_matrix.row_count).data

            for (r_list, summ_list) in zip(r_cov, summ_cov):
                for (r_val, summ_val) in zip(r_list, summ_list):
                    self.assertAlmostEqual(r_val, summ_val)


if __name__ == '__main__':
    unittest.main()
