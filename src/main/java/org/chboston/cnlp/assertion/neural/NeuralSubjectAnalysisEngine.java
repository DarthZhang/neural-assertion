package org.chboston.cnlp.assertion.neural;

import org.apache.ctakes.typesystem.type.textsem.IdentifiedAnnotation;

public class NeuralSubjectAnalysisEngine extends
    NeuralAssertionStatusAnalysisEngine {

  @Override
  public String getLabel(IdentifiedAnnotation ent) {
    return ent.getSubject();
  }

  @Override
  public void applyLabel(IdentifiedAnnotation ent, String label) {
   ent.setSubject(label);
  }

}
