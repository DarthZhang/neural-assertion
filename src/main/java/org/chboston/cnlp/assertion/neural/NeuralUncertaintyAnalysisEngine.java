package org.chboston.cnlp.assertion.neural;

import org.apache.ctakes.typesystem.type.textsem.IdentifiedAnnotation;

public class NeuralUncertaintyAnalysisEngine extends
    NeuralAssertionStatusAnalysisEngine {

  @Override
  public String getLabel(IdentifiedAnnotation ent) {
    return String.valueOf(ent.getUncertainty());
  }

  @Override
  public void applyLabel(IdentifiedAnnotation ent, String label) {
    ent.setUncertainty(Integer.parseInt(label));
  }

}
