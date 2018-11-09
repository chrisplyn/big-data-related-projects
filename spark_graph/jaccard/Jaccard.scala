/**
 *
 * students: please put your implementation in this file!
 */
package edu.gatech.cse6250.jaccard

import edu.gatech.cse6250.model._
import edu.gatech.cse6250.model.{ EdgeProperty, VertexProperty }
import org.apache.spark.graphx._
import org.apache.spark.rdd.RDD

object Jaccard {

  def jaccardSimilarityOneVsAll(graph: Graph[VertexProperty, EdgeProperty], patientID: Long): List[Long] = {
    /**
     * Given a patient ID, compute the Jaccard similarity w.r.t. to all other patients.
     * Return a List of top 10 patient IDs ordered by the highest to the lowest similarity.
     * For ties, random order is okay. The given patientID should be excluded from the result.
     */

    val patientVertexSet = graph.subgraph(vpred = (id, attr) => attr.isInstanceOf[PatientProperty]).vertices.map(_._1).collect().toSet
    val neighbors = graph.collectNeighborIds(EdgeDirection.Out)
    val patientNeighbors = neighbors.filter(x => patientVertexSet.contains(x._1) && x._1 != patientID)
    val patientIDNeighbors = neighbors.filter(x => x._1 == patientID).map(x => x._2).flatMap(x => x).collect().toSet
    val patientJaccard = patientNeighbors.map(x => (x._1, jaccard(patientIDNeighbors, x._2.toSet)))
    patientJaccard.takeOrdered(10)(Ordering[Double].reverse.on(x => x._2)).map(_._1).toList
  }

  def jaccardSimilarityAllPatients(graph: Graph[VertexProperty, EdgeProperty]): RDD[(Long, Long, Double)] = {
    /**
     * Given a patient, med, diag, lab graph, calculate pairwise similarity between all
     * patients. Return a RDD of (patient-1-id, patient-2-id, similarity) where
     * patient-1-id < patient-2-id to avoid duplications
     */

    val patientVertexSet = graph.subgraph(vpred = (id, attr) => attr.isInstanceOf[PatientProperty]).vertices.map(_._1).collect().toSet
    val patientNeighbors = graph.collectNeighborIds(EdgeDirection.Out).filter(x => patientVertexSet.contains(x._1))
    val cartesian_neighbors = patientNeighbors.cartesian(patientNeighbors).filter(x => x._1._1 < x._2._1)
    cartesian_neighbors.map(x => (x._1._1, x._2._1, jaccard(x._1._2.toSet, x._2._2.toSet)))
  }

  def jaccard[A](a: Set[A], b: Set[A]): Double = {
    /**
     * Helper function
     *
     * Given two sets, compute its Jaccard similarity and return its result.
     * If the union part is zero, then return 0.
     */

    /** Remove this placeholder and implement your code */
    if (a.isEmpty || b.isEmpty) { return 0.0 }
    a.intersect(b).size.toDouble / a.union(b).size.toDouble
  }
}
