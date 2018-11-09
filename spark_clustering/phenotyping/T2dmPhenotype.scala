package edu.gatech.cse6250.phenotyping

import edu.gatech.cse6250.model.{ Diagnostic, LabResult, Medication }
import org.apache.spark.rdd.RDD

/**
 * @author Hang Su <hangsu@gatech.edu>,
 * @author Sungtae An <stan84@gatech.edu>,
 */
object T2dmPhenotype {

  /** Hard code the criteria */
  val T1DM_DX = Set("250.01", "250.03", "250.11", "250.13", "250.21", "250.23", "250.31", "250.33", "250.41", "250.43",
    "250.51", "250.53", "250.61", "250.63", "250.71", "250.73", "250.81", "250.83", "250.91", "250.93")

  val T2DM_DX = Set("250.3", "250.32", "250.2", "250.22", "250.9", "250.92", "250.8", "250.82", "250.7", "250.72", "250.6",
    "250.62", "250.5", "250.52", "250.4", "250.42", "250.00", "250.02")

  val T1DM_MED = Set("lantus", "insulin glargine", "insulin aspart", "insulin detemir", "insulin lente", "insulin nph", "insulin reg", "insulin,ultralente")

  val T2DM_MED = Set("chlorpropamide", "diabinese", "diabanase", "diabinase", "glipizide", "glucotrol", "glucotrol xl",
    "glucatrol ", "glyburide", "micronase", "glynase", "diabetamide", "diabeta", "glimepiride", "amaryl",
    "repaglinide", "prandin", "nateglinide", "metformin", "rosiglitazone", "pioglitazone", "acarbose",
    "miglitol", "sitagliptin", "exenatide", "tolazamide", "acetohexamide", "troglitazone", "tolbutamide",
    "avandia", "actos", "actos", "glipizide")

  val DM_RELATED_DX = Set("790.21", "790.22", "790.2", "790.29", "648.81", "648.82", "648.83", "648.84", "648",
    "648.01", "648.02", "648.03", "648.04", "791.5", "277.7", "V77.1", "256.4")

  /**
   * Transform given data set to a RDD of patients and corresponding phenotype
   *
   * @param medication medication RDD
   * @param labResult  lab result RDD
   * @param diagnostic diagnostic code RDD
   * @return tuple in the format of (patient-ID, label). label = 1 if the patient is case, label = 2 if control, 3 otherwise
   */
  def transform(medication: RDD[Medication], labResult: RDD[LabResult], diagnostic: RDD[Diagnostic]): RDD[(String, Int)] = {
    /**
     * Remove the place holder and implement your code here.
     * Hard code the medication, lab, icd code etc. for phenotypes like example code below.
     * When testing your code, we expect your function to have no side effect,
     * i.e. do NOT read from file or write file
     *
     * You don't need to follow the example placeholder code below exactly, but do have the same return type.
     *
     * Hint: Consider case sensitivity when doing string comparisons.
     */

    val sc = medication.sparkContext

    /** Hard code the criteria */
    // val type1_dm_dx = Set("code1", "250.03")
    // val type1_dm_med = Set("med1", "insulin nph")
    // use the given criteria above like T1DM_DX, T2DM_DX, T1DM_MED, T2DM_MED and hard code DM_RELATED_DX criteria as well
    val patients = diagnostic.map(_.patientID).union(labResult.map(_.patientID)).union(medication.map(_.patientID)).distinct()

    val T1_Diag_true = diagnostic.filter(x => T1DM_DX.contains(x.code)).map(_.patientID).distinct()
    val T1_Diag_false = patients.subtract(T1_Diag_true)

    val T2_Diag_true = diagnostic.filter(x => T2DM_DX.contains(x.code)).map(_.patientID).distinct()
    val T2_Diag_false = patients.subtract(T2_Diag_true)

    val T1_Med_true = medication.filter(x => T1DM_MED.contains(x.medicine.toLowerCase)).map(_.patientID).distinct()
    val T1_Med_false = patients.subtract(T1_Med_true)

    val T2_Med_true = medication.filter(x => T2DM_MED.contains(x.medicine.toLowerCase)).map(_.patientID).distinct()
    val T2_Med_false = patients.subtract(T2_Med_true)

    val casePatient_one = T1_Diag_false.intersection(T2_Diag_true).intersection(T1_Med_false)
    val casePatient_two = T1_Diag_false.intersection(T2_Diag_true).intersection(T1_Med_true).intersection(T2_Med_false)

    val casePatient_three_candidate = T1_Diag_false.intersection(T2_Diag_true).intersection(T1_Med_true).intersection(T2_Med_true).collect.toSet

    //    printf("count of case 1 patient: %d", casePatient_one.count())
    //    println()
    //    printf("count of case 2 patient: %d",casePatient_two.count())
    //    println()

    val tmp = medication.filter(x => casePatient_three_candidate.contains(x.patientID)).map(x => (x.patientID, x))
    val tmp2 = tmp.map(x => Medication(x._2.patientID, x._2.date, x._2.medicine))
    val tmp2_T1_med = tmp2.filter(x => T1DM_MED.contains(x.medicine.toLowerCase)).map(x => (x.patientID, x.date.getTime())).reduceByKey(Math.min)
    val tmp2_T2_med = tmp2.filter(x => T2DM_MED.contains(x.medicine.toLowerCase)).map(x => (x.patientID, x.date.getTime())).reduceByKey(Math.min)
    val tmp3 = tmp2_T1_med.join(tmp2_T2_med)
    val casePatient_three = tmp3.filter(x => x._2._1 > x._2._2).map(_._1)
    //    println(casePatient_three.count())

    /** Find CASE Patients */
    val casePatients = sc.union(casePatient_one, casePatient_two, casePatient_three)

    /** Find CONTROL Patients */
    val patients_with_glucose = labResult.filter(x => x.testName.toLowerCase.contains("glucose")).map(_.patientID).collect.toSet

    //    printf("number of patients with glucose: %d", patients_with_glucose.size)
    //    println()

    val lab_patients_with_glucose = labResult.filter(x => patients_with_glucose.contains(x.patientID))
    val patients_with_ab_lab_value = filter_patients_with_abnormal_lab_value(lab_patients_with_glucose)
    val patients_without_ab_lab_value = lab_patients_with_glucose.map(_.patientID).distinct().subtract(patients_with_ab_lab_value)

    //    printf("number of patients without abnormal lab value: %d", patients_without_ab_lab_value.count())
    //    println()

    val patients_without_ab_lab_value_set = patients_without_ab_lab_value.collect.toSet
    val controlPatients = patients_without_ab_lab_value.subtract(diagnostic.filter(x => patients_without_ab_lab_value_set.contains(x.patientID) && (DM_RELATED_DX.contains(x.code) || x.code.startsWith("250."))).map(_.patientID).distinct())

    //    printf("number of control patients: %d", controlPatients.count())
    //    println()

    /** Find OTHER Patients */
    val others = patients.subtract(casePatients).subtract(controlPatients)

    /** Once you find patients for each group, make them as a single RDD[(String, Int)] */
    val phenotypeLabel = sc.union(casePatients.map(x => (x, 1)), controlPatients.map(x => (x, 2)), others.map(x => (x, 3)))

    /** Return */
    phenotypeLabel
  }

  def filter_patients_with_abnormal_lab_value(labResult_g: RDD[LabResult]): RDD[String] = {
    val sc = labResult_g.sparkContext
    labResult_g.filter(x => (x.testName.equals("hba1c") && x.value >= 6.0) ||
      (x.testName.equals("hemoglobin a1c") && x.value >= 6.0) ||
      (x.testName.equals("fasting glucose") && x.value >= 110) ||
      (x.testName.equals("fasting blood glucose") && x.value >= 110) ||
      (x.testName.equals("fasting plasma glucose") && x.value >= 110) ||
      (x.testName.equals("glucose") && x.value > 110) ||
      (x.testName.equals("glucose, serum") && x.value > 110)).map(_.patientID).distinct()
  }

}