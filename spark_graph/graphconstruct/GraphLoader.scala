/**
 * @author Hang Su <hangsu@gatech.edu>.
 */

package edu.gatech.cse6250.graphconstruct

import edu.gatech.cse6250.model._
import org.apache.spark.graphx._
import org.apache.spark.rdd.RDD

object GraphLoader {
  /**
   * Generate Bipartite Graph using RDDs
   *
   * @input: RDDs for Patient, LabResult, Medication, and Diagnostic
   * @return: Constructed Graph
   *
   */
  def load(patients: RDD[PatientProperty], labResults: RDD[LabResult],
    medications: RDD[Medication], diagnostics: RDD[Diagnostic]): Graph[VertexProperty, EdgeProperty] = {

    val sc = patients.sparkContext
    /** HINT: See Example of Making Patient Vertices Below */
    val vertexPatient: RDD[(VertexId, VertexProperty)] = patients
      .map(patient => (patient.patientID.toLong, patient.asInstanceOf[VertexProperty]))

    val num_patient = patients.map(_.patientID).distinct().count()

    val latest_diag = diagnostics.groupBy(x => (x.patientID, x.icd9code)).mapValues(_.maxBy(_.date)).values
    val diag_index_map = latest_diag.map(_.icd9code).distinct().zipWithIndex().map(x => (x._1, x._2 + num_patient + 1))
    val diagVertexId = diag_index_map.collect.toMap

    val vertexDiagnostic: RDD[(VertexId, VertexProperty)] = diag_index_map.map(x => (x._2, DiagnosticProperty(x._1)))
    val num_diag = diag_index_map.count()

    //labresult vertex
    val latest_lab = labResults.groupBy(x => (x.patientID, x.labName)).mapValues(_.maxBy(_.date)).values
    val lab_index_map = latest_lab.map(_.labName).distinct().zipWithIndex().map(x => (x._1, x._2 + num_patient + 1 + num_diag))
    val labVertexId = lab_index_map.collect.toMap
    val vertexLabresult: RDD[(VertexId, VertexProperty)] = lab_index_map.map(x => (x._2, LabResultProperty(x._1)))
    val num_lab = lab_index_map.count()

    //medication vertex
    val latest_med = medications.groupBy(x => (x.patientID, x.medicine)).mapValues(_.maxBy(_.date)).values
    val med_index_map = latest_med.map(_.medicine).distinct().zipWithIndex().map(x => (x._1, x._2 + num_patient + 1 + num_diag + num_lab))
    val medVertexId = med_index_map.collect.toMap
    val vertexMedication: RDD[(VertexId, VertexProperty)] = med_index_map.map(x => (x._2, MedicationProperty(x._1)))

    /**
     * HINT: See Example of Making PatientPatient Edges Below
     *
     * This is just sample edges to give you an example.
     * You can remove this PatientPatient edges and make edges you really need
     */

    /**patient-diag-edge*/
    val edge_patient_diag = latest_diag.map(x => Edge(x.patientID.toLong, diagVertexId(x.icd9code), PatientDiagnosticEdgeProperty(x).asInstanceOf[EdgeProperty]))
    val edge_diag_patient = latest_diag.map(x => Edge(diagVertexId(x.icd9code), x.patientID.toLong, PatientDiagnosticEdgeProperty(x).asInstanceOf[EdgeProperty]))
    val patient_diag_bi_edges = sc.union(edge_patient_diag, edge_diag_patient)

    /**patient-lab-edge*/
    val edge_patient_lab = latest_lab.map(x => Edge(x.patientID.toLong, labVertexId(x.labName), PatientLabEdgeProperty(x).asInstanceOf[EdgeProperty]))
    val edge_lab_patient = latest_lab.map(x => Edge(labVertexId(x.labName), x.patientID.toLong, PatientLabEdgeProperty(x).asInstanceOf[EdgeProperty]))
    val patient_lab_bi_edges = sc.union(edge_patient_lab, edge_lab_patient)

    /**patient-med-edge*/
    val edge_patient_med = latest_med.map(x => Edge(x.patientID.toLong, medVertexId(x.medicine), PatientMedicationEdgeProperty(x).asInstanceOf[EdgeProperty]))
    val edge_med_patient = latest_med.map(x => Edge(medVertexId(x.medicine), x.patientID.toLong, PatientMedicationEdgeProperty(x).asInstanceOf[EdgeProperty]))
    val patient_med_bi_edges = sc.union(edge_patient_med, edge_med_patient)

    // Making Graph
    val vertices = sc.union(vertexPatient, vertexDiagnostic, vertexLabresult, vertexMedication)
    val edges = sc.union(patient_diag_bi_edges, patient_lab_bi_edges, patient_med_bi_edges)
    val graph: Graph[VertexProperty, EdgeProperty] = Graph(vertices, edges)

    graph
  }
}
